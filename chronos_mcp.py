# chronos_mcp.py — Entry point only.
# Responsibility: Initialize schema, build singletons, register tools, run server.
# No domain logic lives here. See chronos/ package for all implementation.
#
# Architecture: CHRONOS v3.1 — zero-dependency Claude project memory layer
# Primary interface: remember / recall / forget / query_at /
#                    update_memory / query_similar_memories
# Advanced interface: add_event / query_similar / analyze_causal /
#                     suggest_next_tasks / analyze_structure / add_constraint
#
# Module layout:
#   tools.py          — memory tools (remember/recall/forget/query_at/stats) + register()
#   graph_tools.py    — add_event, query_similar, add_constraint
#   analysis_tools.py — analyze_causal, suggest_next_tasks, analyze_structure
#   memory_tools.py   — update_memory, query_similar_memories

from mcp.server.fastmcp import FastMCP

from chronos.analyzers import CausalAnalyzer, ConstraintSolver, StructureAnalyzer
from chronos.db import init_db
from chronos.geometry import HyperbolicEmbedder
from chronos.mem_embed import MemoryEmbedder
from chronos.memory import MemoryStore
from chronos.tfidf import TFIDFIndex
from chronos.tools import register

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

mcp = FastMCP("chronos")

# ---------------------------------------------------------------------------
# Startup: schema → singletons → tool registration
# ---------------------------------------------------------------------------

# 1. Apply DDL once — not on every connection
init_db()

# 2. Build singletons after DB is confirmed ready
embedder  = HyperbolicEmbedder(dim=32)
embedder.load_from_db()   # restore node embeddings persisted from previous sessions

causal    = CausalAnalyzer()
solver    = ConstraintSolver()
structure = StructureAnalyzer()

# 3. Memory layer: TF-IDF index + content embedding index
tfidf        = TFIDFIndex()
mem_embedder = MemoryEmbedder(dim=32)
mem_store    = MemoryStore(tfidf, mem_embedder=mem_embedder)
mem_store.load()  # rebuilds TF-IDF index AND loads memory_vectors from DB

# 4. Register all tools — single call wires all sub-modules
register(mcp, embedder, causal, solver, structure, mem_store, mem_embedder)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
