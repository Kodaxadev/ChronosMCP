# chronos/consolidation_config.py
# Responsibility: Tunable thresholds for memory consolidation.

# Minimum TF-IDF cosine similarity to consider two memories as near-duplicates.
# 0.85 is deliberately conservative: only clearly redundant content merges.
DUPLICATE_THRESHOLD = 0.85

# Confidence decay applied per consolidation pass to unreviewed memories.
STALE_DECAY_DELTA = 0.05
STALE_DAYS_THRESHOLD = 30

# FSRS retention below this triggers a "needs review" flag.
RETENTION_WARNING_THRESHOLD = 0.3

# Memories below BOTH prune thresholds are auto-forgotten when auto_prune=True.
PRUNE_CONFIDENCE_THRESHOLD = 0.10
PRUNE_RETENTION_THRESHOLD = 0.15
