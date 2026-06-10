# chronos/search.py
# Responsibility: BM25 full-text retrieval over the memories_fts index,
# snapshot ranking for time-travel, and pairwise similarity for duplicate
# detection. Replaces the v3.x in-memory TF-IDF index (chronos/tfidf.py).
#
# Design:
#   - The live index is the memories_fts FTS5 table (see db.py), maintained
#     by SQLite triggers. Index updates commit atomically with the memories
#     rows they mirror, so there is no in-memory state and no startup load.
#   - Ranking uses SQLite's built-in bm25() with porter stemming, so
#     'running' matches 'run'. Scores are returned as -bm25() (higher =
#     better). They are relative ranking values, not normalised [0,1].
#   - Untrusted free text is never spliced into MATCH syntax: match_query()
#     reduces input to quoted alphanumeric terms joined by OR, which makes
#     FTS5 query-syntax injection ('"', 'NEAR(', column filters) impossible.

import math
import re
import sqlite3
from collections import Counter
from typing import List, Optional, Tuple

# Rough token cost per word — budget hint, not a precise tokenizer count.
TOKENS_PER_WORD = 0.75

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common English words excluded from similarity math and related-memory
# queries. NOT applied to recall queries — BM25's IDF already down-weights
# ubiquitous terms there, and dropping user terms would change query meaning.
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "this", "that", "these",
    "those", "i", "we", "you", "he", "she", "they", "it", "its", "my",
    "our", "your", "his", "her", "their", "not", "no", "so", "if", "as",
    "up", "out", "about", "into", "than", "then", "when", "where", "who",
    "which", "what", "how", "all", "any", "each", "also", "just", "more",
})


def _tokens(text: str) -> List[str]:
    """Lowercase alphanumeric tokens, no other processing."""
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> List[str]:
    """Tokens suitable for similarity math: stop words and 1-char tokens removed."""
    return [t for t in _tokens(text) if len(t) >= 2 and t not in _STOP_WORDS]


def estimate_tokens(text: str) -> int:
    """Rough token count estimate for a text string."""
    words = len(text.split())
    return max(1, round(words * TOKENS_PER_WORD))


# ---------------------------------------------------------------------------
# MATCH query construction
# ---------------------------------------------------------------------------

def match_query(text: str, max_terms: int = 32) -> str:
    """
    Convert free text into a safe FTS5 MATCH expression.

    Each term is double-quoted (a one-token phrase), which neutralises all
    FTS5 query syntax in the input. Terms are joined with OR so partial
    matches rank rather than vanish; BM25 rewards multi-term hits anyway.
    When the text has more than max_terms distinct terms (e.g. a whole
    memory passed by related_memories), the most frequent terms are kept.

    Returns '' when no usable terms exist — callers must treat that as
    'no results', not pass it to MATCH.
    """
    toks = [t for t in _tokens(text) if len(t) >= 2]
    if not toks:
        return ""
    counts = Counter(toks)
    if len(counts) > max_terms:
        terms = [t for t, _ in counts.most_common(max_terms)]
    else:
        # Preserve first-appearance order for short queries (deterministic)
        seen = set()
        terms = [t for t in toks if not (t in seen or seen.add(t))]
    return " OR ".join(f'"{t}"' for t in terms)


# ---------------------------------------------------------------------------
# Live search over memories_fts
# ---------------------------------------------------------------------------

def search_memories(
    db,
    query: str,
    project: Optional[str] = None,
    k: int = 5,
    exclude_id: Optional[str] = None,
) -> List[sqlite3.Row]:
    """
    BM25-ranked search over non-forgotten memories.

    db: open connection from get_db() (Row factory assumed).
    Returns up to k Rows with columns: id, project, content, created_at,
    confidence, stability, last_reviewed, source, score (higher = better).
    """
    mq = match_query(query)
    if not mq:
        return []

    sql = (
        "SELECT m.id, m.project, m.content, m.created_at, m.confidence, "
        "       m.stability, m.last_reviewed, m.source, "
        "       -bm25(memories_fts) AS score "
        "FROM memories_fts "
        "JOIN memories m ON m.id = memories_fts.memory_id "
        "WHERE memories_fts MATCH ? AND m.forgotten = 0"
    )
    params: list = [mq]
    if project:
        sql += " AND m.project = ?"
        params.append(project)
    if exclude_id:
        sql += " AND m.id != ?"
        params.append(exclude_id)
    sql += " ORDER BY bm25(memories_fts) LIMIT ?"
    params.append(k)

    return db.execute(sql, params).fetchall()


# ---------------------------------------------------------------------------
# Snapshot ranking — used by time-travel queries
# ---------------------------------------------------------------------------

def rank_snapshot(
    docs: List[Tuple[str, str]],
    query: str,
    k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Rank an ephemeral (doc_id, content) snapshot against a query using an
    in-memory FTS5 table, so time-travel results use the same BM25 + porter
    ranking as live recall.

    Returns [(doc_id, score)] sorted best-first.
    """
    mq = match_query(query)
    if not mq or not docs:
        return []

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE snap USING fts5("
            "doc_id UNINDEXED, content, tokenize='porter unicode61')"
        )
        conn.executemany("INSERT INTO snap VALUES (?, ?)", docs)
        rows = conn.execute(
            "SELECT doc_id, -bm25(snap) FROM snap WHERE snap MATCH ? "
            "ORDER BY bm25(snap) LIMIT ?",
            (mq, k),
        ).fetchall()
        return [(r[0], float(r[1])) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Pairwise similarity — used by consolidation duplicate detection
# ---------------------------------------------------------------------------

def token_counter(text: str) -> Counter:
    """Counter of content tokens, for reuse across many pairwise comparisons."""
    return Counter(content_tokens(text))


def cosine_similarity(a: Counter, b: Counter) -> float:
    """
    Cosine similarity between two token-count vectors, in [0, 1].
    Stop words are already excluded by token_counter(), so boilerplate
    phrasing does not inflate the score.
    """
    if not a or not b:
        return 0.0
    # Iterate over the smaller counter for the dot product
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    dot = sum(count * large.get(tok, 0) for tok, count in small.items())
    if dot == 0:
        return 0.0
    norm_a = math.sqrt(sum(c * c for c in a.values()))
    norm_b = math.sqrt(sum(c * c for c in b.values()))
    return dot / (norm_a * norm_b)
