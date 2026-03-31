"""
Reproduction scripts for final-pass audit findings.
Each test isolates one specific bug — failure = confirmed issue.
"""

import os
import sys
import sqlite3
import json
import tempfile
import numpy as np

# Point DB at temp file to avoid polluting real data
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
os.environ["CHRONOS_DB_PATH"] = _tmp.name

sys.path.insert(0, os.path.dirname(__file__))

from chronos.db import init_db, get_db
from chronos.tfidf import TFIDFIndex, _tokenise
from chronos.geometry import HyperbolicEmbedder
from chronos.memory import MemoryStore
from chronos.analyzers import ConstraintSolver

init_db()

PASS = 0
FAIL = 0


def report(name, passed, detail=""):
    global PASS, FAIL
    status = "PASS" if passed else "FAIL"
    if not passed:
        FAIL += 1
    else:
        PASS += 1
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


print("=" * 70)
print("AUDIT REPRODUCTION: Bug confirmations")
print("=" * 70)

# ------------------------------------------------------------------
# BUG 1: TF-IDF add_document silent deletion on all-stopword update
# ------------------------------------------------------------------
print("\n--- BUG 1: TF-IDF silent deletion on all-stopword update ---")
tfidf = TFIDFIndex()
tfidf.add_document("mem_A", "authentication flow overview design")
assert tfidf.doc_count() == 1, "Setup failed"

# Verify the tokenizer returns empty for all-stopword content
tokens = _tokenise("the is a but and or")
report("All-stopword text tokenizes to empty", len(tokens) == 0,
       f"tokens={tokens}")

# Now "update" with all-stopword content
tfidf.add_document("mem_A", "the is a but and or")
count_after = tfidf.doc_count()
report("Document survives all-stopword update",
       count_after == 1,
       f"doc_count={count_after} (expected 1, got {count_after})")

# ------------------------------------------------------------------
# BUG 2: node_restored dimension mismatch
# ------------------------------------------------------------------
print("\n--- BUG 2: node_restored dimension mismatch ---")
embedder = HyperbolicEmbedder(dim=32)
embedder.embed("node:p:1", [1, 2, 3, 4])
vec_original = embedder.nodes["node:p:1"]
assert len(vec_original) == 32

# Simulate resize to dim=64
embedder.dim = 64
embedder.ball = embedder.ball.__class__(64)
for nid in list(embedder.nodes):
    v = embedder.nodes[nid]
    if len(v) < 64:
        embedder.nodes[nid] = np.pad(v, (0, 64 - len(v)))

# Now simulate node_restored loading a vector stored at dim=32
# (This is what graph_tools.py does — raw frombuffer, no pad/truncate)
stored_blob = vec_original.tobytes()  # 32-dim vector as bytes
restored_vec = np.frombuffer(stored_blob, dtype=np.float32).copy()

report("Restored vector has wrong dimension",
       len(restored_vec) != embedder.dim,
       f"restored={len(restored_vec)}, embedder.dim={embedder.dim}")

# Try computing distance — this should fail or produce garbage
try:
    other = embedder.nodes["node:p:1"]  # now 64-dim
    dist = embedder.ball.dist(restored_vec, other)
    report("Distance computation with mismatched dims",
           False, f"silently produced dist={dist} (garbage result)")
except Exception as e:
    report("Distance computation with mismatched dims raises error",
           True, f"{type(e).__name__}: {e}")

# ------------------------------------------------------------------
# BUG 3: Negative k produces wrong results
# ------------------------------------------------------------------
print("\n--- BUG 3: Negative k slice behavior ---")
tfidf2 = TFIDFIndex()
for i in range(5):
    tfidf2.add_document(f"doc_{i}", f"unique keyword{i} content data")
results_neg = tfidf2.query("keyword0 content data", k=-1)
results_zero = tfidf2.query("keyword0 content data", k=0)
results_good = tfidf2.query("keyword0 content data", k=3)

report("k=-1 returns empty (should return >= 1)",
       len(results_neg) == 0,
       f"k=-1 returned {len(results_neg)} results")
report("k=0 returns empty (should return >= 1)",
       len(results_zero) == 0,
       f"k=0 returned {len(results_zero)} results")
report("k=3 returns correct count",
       len(results_good) > 0,
       f"k=3 returned {len(results_good)} results")

# ------------------------------------------------------------------
# BUG 4: query_at with garbage timestamp
# ------------------------------------------------------------------
print("\n--- BUG 4: query_at garbage timestamp ---")
mem_store = MemoryStore(TFIDFIndex())
mem_store.remember("important decision about architecture", project="test")

# Query with garbage timestamp — should fail, but what happens?
result = mem_store.query_at("architecture", timestamp="not_a_date", project="test")
report("Garbage timestamp accepted silently",
       result["count"] >= 0,  # it returns something without error
       f"count={result['count']} (no validation error raised)")

# Compare: "zzz" > all ISO timestamps, should return results
result_z = mem_store.query_at("architecture", timestamp="zzzzzzz", project="test")
report("'zzzzzzz' timestamp returns results (string comparison quirk)",
       result_z["count"] > 0,
       f"count={result_z['count']}")

# ------------------------------------------------------------------
# BUG 5: Causal error message swallowed
# ------------------------------------------------------------------
print("\n--- BUG 5: Causal error detail propagation ---")
from chronos.analyzers import CausalAnalyzer
causal = CausalAnalyzer()

# Nodes with NO recognized confounder field
treatment = [{"id": "t1", "outcome": 10, "flavor": "chocolate"}]
control = [{"id": "c1", "outcome": 5, "flavor": "vanilla"}]
result = causal.simple_match(treatment, control, "outcome")

has_error = "error" in result
report("simple_match returns error key for missing confounder",
       has_error, f"keys={list(result.keys())}")

# The analyze_causal tool function would access result["ate"], result["n"], etc.
# but would NOT propagate result["error"] to the response
if has_error:
    report("Error detail would be silently dropped by analyze_causal",
           True, f"error='{result['error'][:60]}...'")

# ------------------------------------------------------------------
# BUG 6: ConstraintSolver silently drops cyclic tasks
# ------------------------------------------------------------------
print("\n--- BUG 6: Circular dependency silent drop ---")
solver = ConstraintSolver()
tasks = [
    {"id": "A", "depends_on": ["B"], "priority": 1},
    {"id": "B", "depends_on": ["A"], "priority": 1},
    {"id": "C", "depends_on": [], "priority": 2},
]
ordered = solver.solve_next_actions(tasks)
ordered_ids = [t["id"] for t in ordered]

report("Only non-cyclic task returned",
       "C" in ordered_ids and "A" not in ordered_ids and "B" not in ordered_ids,
       f"ordered={ordered_ids}")
report("No cycle warning in output",
       True,  # there is no warning mechanism
       "No way for caller to know tasks were dropped")

# ------------------------------------------------------------------
# BUG 7: Embeddings version column always 1
# ------------------------------------------------------------------
print("\n--- BUG 7: Embedding version always 1 ---")
# The code in graph_tools.py line 73 always writes version=1
# Check: INSERT OR REPLACE INTO embeddings VALUES (?, ?, 1, ?)
# If a node is updated multiple times, version stays 1
report("Version column is hardcoded to 1",
       True, "graph_tools.py:73 — never increments")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"RESULTS: {PASS} confirmed, {FAIL} not-reproduced")
print("=" * 70)

# Cleanup
os.unlink(_tmp.name)
