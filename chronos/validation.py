# chronos/validation.py
# Responsibility: Event schema validation — whitelist enforcement, format checks.
# Strict mode: no silent coercion. Raises ValueError on any violation.

import re

# ---------------------------------------------------------------------------
# Spec §2.2 — full event taxonomy
# ---------------------------------------------------------------------------

VALID_EVENT_TYPES: frozenset = frozenset({
    "node_created",
    "node_updated",
    "node_deleted",
    "node_restored",
    "relation_added",
    "relation_removed",
    "relation_updated",
    "snapshot_created",
    "embedding_recomputed",
    "dimension_changed",
})

# Spec §2.5 — aggregate_id must be {type}:{tenant}:{id}
# Uses \S (non-whitespace, non-colon) to reject spaces, tabs, newlines in segments.
_AGGREGATE_RE = re.compile(r"^(node|sprint|team):[^\s:]+:[^\s:]+\Z")


def validate_event(aggregate_id: str, event_type: str, payload: dict) -> None:
    """
    Minimal local port of the Event Validation Service (§3).
    Raises ValueError with a descriptive message on any violation.
    Strict mode: no silent coercion.
    """
    # Required fields non-empty
    if not aggregate_id or not aggregate_id.strip():
        raise ValueError("aggregate_id must be a non-empty string")
    if not event_type or not event_type.strip():
        raise ValueError("event_type must be a non-empty string")
    if not isinstance(payload, dict) or not payload:
        raise ValueError("payload must be a non-empty dict (§3.1)")

    # aggregate_id format (§2.5 CHECK constraint)
    if not _AGGREGATE_RE.match(aggregate_id):
        raise ValueError(
            f"aggregate_id '{aggregate_id}' must match "
            r"^(node|sprint|team):[^:]+:[^:]+$  "
            "Example: 'node:myproject:task_001'"
        )

    # Event type whitelist (§2.2)
    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(
            f"Unknown event_type '{event_type}'. "
            f"Valid types: {sorted(VALID_EVENT_TYPES)}"
        )
