# chronos/tfidf.py
# Responsibility: TF-IDF content indexer for free-text memory retrieval.
# Zero external dependencies — pure Python + numpy only.
#
# Design:
#   - Maintains an in-memory term-document matrix rebuilt on load/update
#   - Persists raw document text in the memories table (db layer)
#   - query() returns ranked (doc_id, score) pairs for injection into recall
#
# Limitations (acknowledged):
#   - No cross-vocabulary generalisation (synonyms don't match)
#   - English stop-word list only
#   - Adequate for project memory where vocabulary is controlled

import math
import re
from collections import Counter
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Stop words — common English words that carry no retrieval signal
# ---------------------------------------------------------------------------

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


def _tokenise(text: str) -> List[str]:
    """
    Lowercase, strip punctuation, split on whitespace, remove stop words.
    Returns list of stemmed-ish tokens (simple suffix stripping).
    """
    text   = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    result = []
    for tok in tokens:
        if tok in _STOP_WORDS or len(tok) < 2:
            continue
        # Minimal suffix stripping: remove trailing 'ing', 'ed', 's'
        if len(tok) > 5 and tok.endswith("ing"):
            tok = tok[:-3]
        elif len(tok) > 4 and tok.endswith("ed"):
            tok = tok[:-2]
        elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        result.append(tok)
    return result


class TFIDFIndex:
    """
    In-memory TF-IDF index over free-text memory documents.

    Persistence model:
      - Raw text is stored in the DB (memories table) by the memory module
      - This index is rebuilt from DB on server startup via load_documents()
      - Updated incrementally via add_document() / remove_document()

    Token cost estimate:
      - Assumes ~0.75 tokens per word (GPT/Claude tokenisation approximation)
    """

    TOKENS_PER_WORD = 0.75

    def __init__(self) -> None:
        # doc_id -> raw text
        self._docs: Dict[str, str] = {}
        # doc_id -> Counter of token frequencies
        self._tf: Dict[str, Counter] = {}
        # token -> number of documents containing it
        self._df: Counter = Counter()
        # cached IDF values, invalidated on doc add/remove
        self._idf_cache: Dict[str, float] = {}
        self._idf_dirty: bool = True

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_document(self, doc_id: str, text: str) -> None:
        """Index a new document. Replaces any existing entry for doc_id.
        If the new text produces zero tokens (all stop words), the old entry
        is preserved rather than silently deleting it from the index."""
        tokens = _tokenise(text)
        if not tokens:
            # FIX: Don't remove old entry if new text is un-indexable.
            # Caller still sees the doc via get_text() with old content.
            return

        if doc_id in self._docs:
            self.remove_document(doc_id)

        self._docs[doc_id]  = text
        self._tf[doc_id]    = Counter(tokens)
        for tok in set(tokens):
            self._df[tok] += 1
        self._idf_dirty = True

    def remove_document(self, doc_id: str) -> None:
        """Remove a document from the index."""
        if doc_id not in self._docs:
            return
        for tok in set(self._tf[doc_id].keys()):
            self._df[tok] -= 1
            if self._df[tok] <= 0:
                del self._df[tok]
        del self._docs[doc_id]
        del self._tf[doc_id]
        self._idf_dirty = True

    def load_documents(self, docs: List[Tuple[str, str]]) -> None:
        """
        Bulk-load (doc_id, text) pairs on server startup.
        Replaces the entire index — only call at init time.
        """
        self._docs.clear()
        self._tf.clear()
        self._df.clear()
        self._idf_cache.clear()
        self._idf_dirty = True
        for doc_id, text in docs:
            tokens = _tokenise(text)
            if not tokens:
                continue
            self._docs[doc_id]  = text
            self._tf[doc_id]    = Counter(tokens)
            for tok in set(tokens):
                self._df[tok] += 1

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _idf(self, token: str) -> float:
        """Smoothed IDF: log((N+1) / (df+1)) + 1"""
        if self._idf_dirty:
            self._rebuild_idf()
        return self._idf_cache.get(token, math.log(len(self._docs) + 1) + 1)

    def _rebuild_idf(self) -> None:
        n = len(self._docs)
        self._idf_cache = {
            tok: math.log((n + 1) / (df + 1)) + 1
            for tok, df in self._df.items()
        }
        self._idf_dirty = False

    def query(self, text: str, k: int = 5,
              exclude: set = None) -> List[Tuple[str, float]]:
        """
        Return top-k (doc_id, score) pairs ranked by TF-IDF cosine similarity.
        Excludes tombstoned doc_ids passed in `exclude`.
        Returns empty list if index is empty.
        """
        if not self._docs:
            return []

        # Rebuild IDF cache once at query start, not lazily inside the scoring loop.
        # This prevents any edge case where the cache could be partially stale
        # when multiple add/remove ops occur before the first query call.
        if self._idf_dirty:
            self._rebuild_idf()

        exclude   = exclude or set()
        q_tokens  = _tokenise(text)
        if not q_tokens:
            return []

        q_counts  = Counter(q_tokens)
        q_len     = len(q_tokens)

        scores: Dict[str, float] = {}
        for doc_id, tf_counts in self._tf.items():
            if doc_id in exclude:
                continue
            doc_len = sum(tf_counts.values())
            score   = 0.0
            for tok, q_count in q_counts.items():
                if tok not in tf_counts:
                    continue
                tf_d  = tf_counts[tok] / doc_len
                tf_q  = q_count / q_len
                idf   = self._idf(tok)
                score += tf_d * tf_q * (idf ** 2)
            if score > 0:
                scores[doc_id] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def estimate_tokens(self, text: str) -> int:
        """Rough token count estimate for a text string."""
        words = len(text.split())
        return max(1, round(words * self.TOKENS_PER_WORD))

    def doc_count(self) -> int:
        return len(self._docs)

    def doc_ids(self) -> set:
        """Return the set of all currently indexed document IDs."""
        return set(self._docs.keys())

    def get_text(self, doc_id: str) -> str:
        """Return raw stored text for a doc_id, or empty string if missing."""
        return self._docs.get(doc_id, "")
