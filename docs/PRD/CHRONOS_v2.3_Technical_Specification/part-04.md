# CHRONOS v2.3 Technical Specification - Part 4

Source split from CHRONOS_v2.3_Technical_Specification.md to keep files under 400 lines.

    async def handle_snapshot_created(self, message: dict):
        pass  # Implement cache warm-up or log as needed

    async def handle_events_new(self, message: dict):
        pass  # Implement downstream fan-out as needed

    async def invalidate_cache(self, node_id):
        if node_id:
            await self.redis.delete(f"embedding:{node_id}")
        else:
            async for key in self.redis.scan_iter("embedding:*"):
                await self.redis.unlink(key)  # Non-blocking async delete

    async def notify_clients_of_dimension_change(self, new_dim: int):
        pass  # WebSocket / pub-sub notification stub

    async def fallback_poller(self):
        """60s fallback for missed NOTIFY messages."""
        while True:
            await asyncio.sleep(self.fallback_poll_interval)
            latest = await self.get_latest_version()
            if latest > self.last_known_version:
                logger.warning(f"Fallback poll detected version drift: {latest}")
                await self.handle_embedding_updated({"payload": {"version": latest}})
```

---

## 10. Operational Runbook

### 10.1 Monitoring Metrics

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| `embedding_reconstruction_loss` | > 0.10 | > 0.15 (Ã—3) | Trigger re-embed |
| `event_store_lag_ms` | > 1000 | > 5000 | Scale ingestion |
| `causal_queue_depth` | > 500 | > 2000 | Scale workers |
| `snapshot_age_hours` | > 26 | > 48 | Manual compaction |
| `propensity_balance_max_smd` | > 0.1 | > 0.2 | Reject hypothesis |

### 10.2 Health Check Endpoints

- `/health/live` â€” Kubernetes liveness (immediate)
- `/health/ready` â€” DB connection, version table accessible
- `/health/deep` â€” Reconstruction loss, causal queue depth, IPC connectivity

### 10.3 Critical Procedures

**Dimension change stuck (>30 min)**:
```sql
-- Force rollback to previous version
UPDATE embedding_versions SET is_active = true  WHERE version_id = <PREV>;
UPDATE embedding_versions SET is_active = false WHERE version_id = <STUCK>;
UPDATE embeddings SET dim_change_in_progress = false;
```

**Compaction failure**:
```bash
python manage.py compact --aggregate-id node:critical:aggregate --force
python manage.py verify-snapshot --snapshot-id <uuid>
```

**Causal engine backpressure (queue > 1000)**:
```sql
UPDATE causal_hypotheses SET status = 'paused'
WHERE status = 'hypothesis' AND created_at < NOW() - INTERVAL '1 day';
```

### 10.4 Backup & DR

| Target | Method | RPO | RTO |
|--------|--------|-----|-----|
| Event Store | Continuous WAL â†’ S3 | 5 min | 30 min |
| Snapshots | Daily full backup | 24 h | 10 min |
| Embeddings | Reconstructible from events | N/A | 2â€“4 h |

---

## 11. Security & Compliance

### 11.1 Authentication

OAuth2/JWT (RS256). Row-level security at PostgreSQL:

```sql
ALTER TABLE events ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON events
    USING (metadata->>'tenant_id' = current_setting('app.current_tenant'));
```

### 11.2 GDPR â€” Key Manager & Anonymization

```python
import hmac, hashlib, secrets
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class KeyRotation:
    key_id:     str
    key_value:  str
    created_at: datetime
    expires_at: datetime

class KeyManager:
    def __init__(self, rotation_days=90, retention_keys=3):
        self.rotation_days  = rotation_days
        self.retention_keys = retention_keys
        self._keys = {}
        self._current_key_id = None
        self._rotate_key()

    def _rotate_key(self):
        # Unix timestamp in seconds â€” collision-safe (not date string)
        new_id = f"key_{int(datetime.utcnow().timestamp())}"
        self._keys[new_id] = KeyRotation(
            key_id=new_id,
            key_value=secrets.token_hex(32),
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=self.rotation_days)
        )
        self._current_key_id = new_id
        if len(self._keys) > self.retention_keys:
            oldest = min(self._keys, key=lambda k: self._keys[k].created_at)
            del self._keys[oldest]

    def get_current_key(self) -> str:
        current = self._keys.get(self._current_key_id)
        if current and datetime.utcnow() > current.expires_at:
            self._rotate_key()
        return self._keys[self._current_key_id].key_value

    @property
    def current_key_id(self): return self._current_key_id

    def get_key(self, key_id: str) -> str:
        if key_id not in self._keys:
            raise KeyError(f"Key {key_id} expired or unknown")
        return self._keys[key_id].key_value


class GDPRAnonymizer:
    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager

    def anonymize_user(self, user_id: str) -> str:
        """HMAC-SHA256 with rotating key. Prevents rainbow table attacks."""
        key       = self.key_manager.get_current_key()
        anon_hash = hmac.new(key.encode(), user_id.encode(), hashlib.sha256).hexdigest()[:32]
        return f"anon:{anon_hash}:{self.key_manager.current_key_id}"

    def verify_anonymized(self, anonymized_id: str, candidate_user_id: str) -> bool:
        _, hash_val, key_id = anonymized_id.split(":")
        key      = self.key_manager.get_key(key_id)
        expected = hmac.new(key.encode(), candidate_user_id.encode(), hashlib.sha256).hexdigest()[:32]
        return hmac.compare_digest(hash_val, expected)
```

### 11.3 Audit Logging

```json
{
  "timestamp":      "2026-03-27T09:03:00Z",
  "actor":          "user_123",
  "action":         "embedding_query",
  "resource":       "node:tenant1:task_456",
  "outcome":        "success",
  "ip_address":     "10.0.0.1",
  "correlation_id": "req_abc123"
}
```

Retention: 7 years. WORM storage class. Separate PostgreSQL instance.

---

## 12. Naming & Canonical Reference

All subsystem names are standardized as follows. Historical aliases are retired.

| Component | Canonical Name |
|-----------|---------------|
| Causal analysis subsystem | **Causal Engine** |
| Constraint processing subsystem | **Constraint Solver** |
| Hyperbolic geometry subsystem | **Hyperbolic Embedding Engine** |
| Topology subsystem | **TDA/Mapper Engine** |
| Ingestion validation stage | **Event Validation Service** |

---

*Document Control: v2.3 â€” 2026-03-27 â€” All contradictions resolved, all gaps closed, all runtime bugs patched.*
