# CHRONOS v2.3 Technical Specification - Part 1

Source split from CHRONOS_v2.3_Technical_Specification.md to keep files under 400 lines.

# CHRONOS v2.3 â€” Complete Technical Documentation Suite

**Status**: Implementation-Ready  
**Last Validated**: 2026-03-27  
**Architecture Version**: 2.3  

---

## Table of Contents

1. [Technical Architecture Specification](#1-technical-architecture-specification)
2. [Data Model & Event Sourcing](#2-data-model--event-sourcing)
3. [Event Validation Pipeline](#3-event-validation-pipeline)
4. [Hyperbolic Embedding Engine](#4-hyperbolic-embedding-engine)
5. [Causal Analysis Engine](#5-causal-analysis-engine)
6. [Constraint Solver](#6-constraint-solver)
7. [TDA/Mapper Engine](#7-tdamapper-engine)
8. [API & Integration Specification](#8-api--integration-specification)
9. [LISTEN/NOTIFY IPC Contract](#9-listennotify-ipc-contract)
10. [Operational Runbook](#10-operational-runbook)
11. [Security & Compliance](#11-security--compliance)
12. [Naming & Canonical Reference](#12-naming--canonical-reference)

---

## 1. Technical Architecture Specification

### 1.1 Runtime Architecture

CHRONOS implements a **4-layer runtime** with **4 analytical subsystems**:

| Layer | Responsibilities | Key Technologies |
|-------|-----------------|------------------|
| **Gateway** | Auth, rate limiting, idempotency, event ingestion, version routing | FastAPI/Go, OAuth2, Redis Cluster |
| **Materialized Views** | Read-model construction, analytical computation, caching | Python, Gudhi, sklearn, python-constraint |
| **Engine** | Event sourcing, snapshotting, compaction, IPC | PostgreSQL, LISTEN/NOTIFY, asyncpg |
| **Storage** | Persistent state, embeddings, snapshots, cache | PostgreSQL, pgvector, S3/MinIO, Redis |

**Consistency Model**: Eventually consistent (5-minute SLA for embedding updates).  
**CQRS Pattern**: Command (write) path via Gateway â†’ Event Store; Query path via materialized views.

### 1.2 Subsystem Interaction Matrix

| Source | Target | Trigger | Data Format |
|--------|--------|---------|-------------|
| Gateway | Event Store | HTTP POST | Event Envelope v2.3 |
| Event Store | Snapshotter | Every 100 events | Aggregate State |
| Event Store | Hyperbolic Engine | 5-min cron + threshold | Graph Batch |
| Hyperbolic Engine | Embedding Store | Embedding completion | Vector + Metadata |
| Causal Engine | Constraint Solver | Hypothesis validation | Constraint Score |
| Query API | All Views | User Request | Query + Version Header |
| Engine | Views (all) | LISTEN/NOTIFY | JSON notification payload |

### 1.3 Data Retention Policy

| Data Type | Retention | Notes |
|-----------|-----------|-------|
| Raw Events | 90 days | Soft-deleted after snapshot compaction |
| Snapshots | Indefinite | S3/MinIO versioned buckets |
| Tombstones | **Permanent** | Required for causal validity |
| Audit Logs | 7 years | WORM storage, separate instance |

---

## 2. Data Model & Event Sourcing

### 2.1 Event Envelope Schema

Every event **MUST** conform to this envelope:

```json
{
  "event_id": "018e1234-5678-7abc-8def-0123456789ab",
  "aggregate_id": "node:proj_123:task_456",
  "event_type": "node_created",
  "timestamp": "2026-03-27T09:03:00.000Z",
  "payload": {
    "node_type": "task",
    "attributes": {
      "title": "Implement authentication",
      "status": "open",
      "priority": "high"
    },
    "relations": [
      {"target": "node:proj_123:user_789", "type": "assigned_to"}
    ]
  },
  "schema_version": "2.3",
  "metadata": {
    "source": "github_webhook",
    "ingestion_id": "req_abc123",
    "tenant_id": "tenant_xyz"
  }
}
```

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `event_id` | UUIDv7 | PK, Time-sortable | App-generated; lexicographically sortable by time |
| `aggregate_id` | String | Format: `{type}:{tenant}:{id}` | Entity identifier with type prefix |
| `event_type` | Enum | See Â§2.2 | Domain event classification |
| `timestamp` | ISO8601 | UTC, millisecond precision | Event occurrence time |
| `schema_version` | SemVer | Major.Minor only | Spec version for migration logic |

### 2.2 Event Taxonomy

**Node Lifecycle**: `node_created`, `node_updated`, `node_deleted`, `node_restored`  
**Relation Events**: `relation_added`, `relation_removed`, `relation_updated`  
**System Events**: `snapshot_created`, `embedding_recomputed`, `dimension_changed`

### 2.3 UUIDv7 Generation (Application Layer)

`event_id` is generated at the application layer. No database default.

```python
import time
import secrets

def generate_uuidv7_pure() -> str:
    """RFC 9562 UUIDv7 with CSPRNG. No external dependencies."""
    # time_ns() avoids float precision loss vs time.time()
    ts_ms = time.time_ns() // 1_000_000

    # CSPRNG â€” unpredictable, safe for security-sensitive IDs
    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)

    # RFC 9562 bit layout
    time_hi      = (ts_ms >> 16) & 0xFFFFFFFF   # 32-bit high timestamp
    time_mid     = ts_ms & 0xFFFF                # 16-bit low timestamp
    ver_rand_a   = 0x7000 | (rand_a & 0x0FFF)   # version=7 + 12-bit rand_a
    var_rand_b_hi = 0x8000 | ((rand_b >> 48) & 0x3FFF)  # variant=10 + 14-bit rand_b
    rand_b_lo    = rand_b & 0xFFFFFFFFFFFF       # 48-bit rand_b low

    return (
        f"{time_hi:08x}-{time_mid:04x}-"
        f"{ver_rand_a:04x}-{var_rand_b_hi:04x}-{rand_b_lo:012x}"
    )
```

> **Note**: When the `uuid7` PyPI package is available (`pip install uuid7>=0.1.0`), prefer it:  
> `from uuid7 import uuid7 as generate_uuidv7`

### 2.4 Tombstone Policy

Tombstones are **permanent** â€” they are never deleted. This preserves causal validity of historical analyses.

```json
{
  "event_type": "node_deleted",
  "payload": {
    "node_id": "node:proj_123:task_456",
    "reason": "manual_delete|merge_duplicate|gdpr_request|automated_cleanup",
    "deleted_by": "user_id",
    "merged_to": "node:proj_123:task_789"
  }
}
```

**Behavioral Semantics**:
- **Query Side**: Tombstoned nodes excluded from `current_nodes`, retained in `all_nodes`
- **Embedding Side**: Node removed from graph structure; vector retained in `archived_embeddings`
- **Causal Side**: Tombstone treated as censoring event (survival analysis)

### 2.5 PostgreSQL Schema

```sql
-- Dependencies
CREATE EXTENSION IF NOT EXISTS "pgvector";

-- Event Store (Append-only, no triggers, no updates)
CREATE TABLE events (
    event_id        UUID PRIMARY KEY,  -- App-generated UUIDv7
    aggregate_id    TEXT NOT NULL,
    event_type      TEXT NOT NULL CHECK (event_type IN (
        'node_created','node_updated','node_deleted','node_restored',
        'relation_added','relation_removed','relation_updated',
        'snapshot_created','embedding_recomputed','dimension_changed'
    )),
    timestamp       TIMESTAMPTZ NOT NULL,
    payload         JSONB NOT NULL,
    schema_version  TEXT NOT NULL DEFAULT '2.3',
    metadata        JSONB,
    sequence_number BIGSERIAL UNIQUE,
    CONSTRAINT valid_aggregate CHECK (aggregate_id ~ '^(node|sprint|team):[^:]+:[^:]+$')
);

CREATE INDEX idx_events_aggregate  ON events(aggregate_id, sequence_number);
CREATE INDEX idx_events_timestamp  ON events(timestamp);
CREATE INDEX idx_events_type       ON events(event_type);
CREATE INDEX idx_events_payload_gin ON events USING GIN (payload jsonb_path_ops);

-- Snapshots (no FK to events â€” aggregate_id not unique in events)
CREATE TABLE snapshots (
    snapshot_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregate_id    TEXT NOT NULL,
    sequence_number BIGINT NOT NULL,
    state           JSONB NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    event_count     INTEGER NOT NULL,
    schema_version  TEXT NOT NULL,
    CONSTRAINT valid_sequence CHECK (sequence_number > 0)
);
CREATE INDEX idx_snapshots_aggregate ON snapshots(aggregate_id, sequence_number DESC);

-- Tombstones â€” PERMANENT, no retention_until
CREATE TABLE tombstones (
    node_id         TEXT PRIMARY KEY,
    event_id        UUID REFERENCES events(event_id),
    deleted_at      TIMESTAMPTZ DEFAULT NOW(),
    reason          TEXT,
    gdpr_request_id TEXT   -- Audit trail for GDPR requests
);

-- Embeddings (pgvector)
CREATE TABLE embeddings (
    node_id              TEXT PRIMARY KEY,
    embedding            vector(128),
    version              INTEGER NOT NULL,
    updated_at           TIMESTAMPTZ DEFAULT NOW(),
    dimensions           INTEGER NOT NULL,
    is_stale             BOOLEAN DEFAULT FALSE,
    dim_change_in_progress BOOLEAN DEFAULT FALSE
);
CREATE INDEX idx_embeddings_version ON embeddings(version);
CREATE INDEX idx_embeddings_vector  ON embeddings USING ivfflat (embedding vector_cosine_ops);

-- Embedding Version Control
CREATE TABLE embedding_versions (
    version_id           SERIAL PRIMARY KEY,
    dimension            INTEGER NOT NULL,
    node_count           INTEGER NOT NULL,
    reconstruction_loss  FLOAT,
    created_at           TIMESTAMPTZ DEFAULT NOW(),
    is_active            BOOLEAN DEFAULT TRUE,
    validation_edge_count INTEGER
);
-- Only one active version at a time
CREATE UNIQUE INDEX one_active_embedding_version
    ON embedding_versions(is_active) WHERE is_active = true;

-- Causal Hypotheses
CREATE TABLE causal_hypotheses (
    hypothesis_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    treatment_node   TEXT,
    outcome_node     TEXT,
    status           TEXT CHECK (status IN (
        'hypothesis','observational','counterfactual_validated','paused'
    )),
    propensity_model JSONB,
    matched_samples  JSONB,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    validated_at     TIMESTAMPTZ
);

-- Persistence Diagrams (TDA)
CREATE TABLE persistence_diagrams (
    diagram_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id            TEXT NOT NULL,
    computed_at           TIMESTAMPTZ DEFAULT NOW(),
    filter_function       TEXT NOT NULL,
    cover_intervals       INTEGER NOT NULL,
    features              JSONB NOT NULL,
    n_connected_components INTEGER,
    n_loops               INTEGER,
    max_persistence       FLOAT,
    entropy               FLOAT
);
CREATE INDEX idx_persistence_project ON persistence_diagrams(project_id, computed_at);
```

### 2.6 Snapshot & Compaction

```python
def compact_aggregate(aggregate_id: str):
    events = get_events_since_last_snapshot(aggregate_id)
    if len(events) >= 100:
        current_state = fold_events(events)
        snapshot = create_snapshot(
            aggregate_id=aggregate_id,
            state=current_state,
            sequence_number=events[-1].sequence_number,
            event_count=len(events)
        )
        schedule_archival(aggregate_id, events[0].timestamp + timedelta(days=90))
        return snapshot

def rehydrate_aggregate(aggregate_id: str) -> State:
    snapshot = get_latest_snapshot(aggregate_id)
    base_state = snapshot.state if snapshot else {}
    from_seq   = snapshot.sequence_number if snapshot else 0
    recent_events = get_events_after(aggregate_id, from_seq)
    return fold_events(base_state, recent_events)
```

---

## 3. Event Validation Pipeline

The Event Validation Service is a **dedicated microservice** between ingestion adapters and the Event Store.

**Pipeline Order**: Raw Event â†’ Schema Validation â†’ Auth Check â†’ Idempotency Check â†’ Enrichment (parallel) â†’ Sanitization â†’ Event Store

### 3.1 JSON Schema (Draft 2020-12, Strict Mode)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "chronos-event-v2.3",
  "type": "object",
  "required": ["event_id","aggregate_id","event_type","timestamp","payload","schema_version"],
  "properties": {
    "event_id": {
      "type": "string",
      "pattern": "^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
    },
    "aggregate_id": {
      "type": "string",
      "pattern": "^(node|sprint|team):[^:]+:[^:]+$"
    },
    "event_type": {
      "enum": [
        "node_created","node_updated","node_deleted","node_restored",
        "relation_added","relation_removed","relation_updated"
      ]
    },
    "timestamp":      { "type": "string", "format": "date-time" },
    "payload":        { "type": "object", "minProperties": 1 },
    "schema_version": { "const": "2.3" },
    "metadata": {
      "type": "object",
      "properties": {
        "source":       { "type": "string" },
        "ingestion_id": { "type": "string" },
        "tenant_id":    { "type": "string" },
        "geo_region":   { "type": "string" }
      },
      "additionalProperties": false
    }
  },
  "additionalProperties": false
}
```

**Strict Mode Rules**: No type coercion. `additionalProperties: false` at root. Pattern validation fails without normalization.

### 3.2 Idempotency

**Redis Key**: `idempotency:{tenant_id}:{idempotency_key}`  
**TTL**: 24 hours  
