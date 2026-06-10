# chronos_mcp.py — Entry point only.
# Responsibility: Initialize schema, build singletons, register tools, run server.
# No domain logic lives here. See chronos/ package for all implementation.
#
# Architecture: CHRONOS v4.0 — temporal memory layer for Claude
#   Core interface:      remember / recall / get_memory / forget /
#                        restore_memory / purge_memory / update_memory /
#                        related_memories / query_at
#   Cognitive interface: set_confidence / boost_confidence / weaken_confidence /
#                        review_memory / get_memory_health / log_search_feedback /
#                        search_effectiveness / consolidate_memories
#   Graph interface:     add_event / query_similar / add_constraint /
#                        analyze_causal / suggest_next_tasks / analyze_structure
#                        — OPT-IN via CHRONOS_ENABLE_GRAPH=1. Quarantined in
#                        v4.0: structural-feature similarity proved a weak
#                        signal in practice, and the layer is experimental.
#
# Search is SQLite FTS5/BM25, trigger-synced with content writes — there is
# no in-memory index and no startup rebuild (see chronos/db.py, search.py).

import os

from mcp.server.fastmcp import FastMCP

from chronos.beliefs import BeliefEngine
from chronos.consolidation import ConsolidationEngine
from chronos.db import init_db
from chronos.memory import MemoryStore
from chronos.tools import register


def _graph_enabled() -> bool:
    return os.environ.get("CHRONOS_ENABLE_GRAPH", "").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

mcp = FastMCP("chronos")

# ---------------------------------------------------------------------------
# Startup: schema → singletons → tool registration
# ---------------------------------------------------------------------------

# 1. Apply DDL + migrations + FTS index/triggers once — not per connection
init_db()

# 2. Core singletons (stateless — all persistence lives in SQLite)
mem_store            = MemoryStore()
belief_engine        = BeliefEngine()
consolidation_engine = ConsolidationEngine(belief_engine)

# 3. Optional graph layer — experimental, off by default
embedder = None
if _graph_enabled():
    from chronos.analysis_tools import register_analysis_tools
    from chronos.analyzers import CausalAnalyzer, ConstraintSolver, StructureAnalyzer
    from chronos.geometry import HyperbolicEmbedder
    from chronos.graph_tools import register_graph_tools

    embedder = HyperbolicEmbedder(dim=32)
    embedder.load_from_db()
    register_graph_tools(mcp, embedder)
    register_analysis_tools(
        mcp, CausalAnalyzer(), ConstraintSolver(), StructureAnalyzer()
    )

# 4. Register core tools — single call wires all always-on sub-modules
register(
    mcp,
    mem_store,
    belief_engine=belief_engine,
    consolidation_engine=consolidation_engine,
    embedder=embedder,
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
