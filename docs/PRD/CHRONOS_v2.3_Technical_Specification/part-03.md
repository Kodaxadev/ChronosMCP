# CHRONOS v2.3 Technical Specification - Part 3

Source split from CHRONOS_v2.3_Technical_Specification.md to keep files under 400 lines.

            labels = clusterer.fit_predict(data[indices])
            for local_id in set(labels):
                if local_id == -1:
                    continue
                members = indices[labels == local_id]
                clusters.append({
                    "id": cluster_id,
                    "members": members.tolist(),
                    "level": i,
                    "centroid": data[members].mean(axis=0)
                })
                cluster_id += 1

        edges = []
        for i, c1 in enumerate(clusters):
            for c2 in clusters[i+1:]:
                shared = set(c1["members"]) & set(c2["members"])
                if shared:
                    edges.append({
                        "source": c1["id"],
                        "target": c2["id"],
                        "weight": len(shared),
                        "shared_members": list(shared)
                    })
        return MapperGraph(nodes=clusters, edges=edges)

    def compute_persistence_diagram(
        self, data: np.ndarray,
        max_dimension: int = 2,
        max_edge_length: float = 2.0
    ):
        """
        Persistent homology via Gudhi Rips complex.
        Uses correct persistence algorithm (not manual filtration).
        """
        rips_complex = gudhi.RipsComplex(points=data, max_edge_length=max_edge_length)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
        persistence  = simplex_tree.persistence()

        features = [
            {
                "dimension": dim,
                "birth": birth,
                "death": death if death != float("inf") else None,
                "persistence": (death - birth) if death != float("inf") else max_edge_length
            }
            for dim, (birth, death) in persistence
        ]

        betti = simplex_tree.betti_numbers()
        return PersistenceDiagram(
            features=features,
            statistics={
                "betti_0":               betti[0] if len(betti) > 0 else 0,
                "betti_1":               betti[1] if len(betti) > 1 else 0,
                "n_persistent_components": len([f for f in features if f["dimension"] == 0 and f["death"]]),
                "n_loops":               len([f for f in features if f["dimension"] == 1]),
                "max_persistence":       max((f["persistence"] for f in features), default=0)
            }
        )
```

---

## 8. API & Integration Specification

### 8.1 REST API Endpoints

**Ingest Event**:
```
POST /v1/events
Authorization: Bearer {token}
Idempotency-Key: {uuid}
â†’ 201 Created: { "event_id": "...", "sequence_number": 12345, "status": "accepted" }
```

**Query Embedding**:
```
GET /v1/nodes/{node_id}/embedding?version={optional}
â†’ 200 OK: { "node_id": "...", "embedding": [...], "version": 42,
            "dim_change_in_progress": false, "is_stale": false }
```

**Causal Query**:
```
POST /v1/causal/query
â†’ 200 OK: { "hypothesis_id": "...", "status": "counterfactual_validated",
            "average_treatment_effect": -0.45, "confidence_interval": [-0.67,-0.23],
            "p_value": 0.002, "matched_samples": 45, "balance_metrics": {...} }
```

**Topology Query**:
```
GET /v1/projects/{project_id}/topology
â†’ 200 OK: { "features": {"components": 5, "loops": 2, ...},
            "bottlenecks": [...], "clusters": [...] }
```

### 8.2 GitHub Webhook Handler

```python
async def handle_github_webhook(request, tenant_id):
    payload    = await request.json()
    event_type = request.headers.get("X-GitHub-Event")
    if event_type == "pull_request":
        pr = payload["pull_request"]
        await ingest_event({
            "event_id":      generate_uuidv7(),
            "aggregate_id":  f"node:{tenant_id}:pr_{pr['id']}",
            "event_type":    "node_created" if payload["action"] == "opened" else "node_updated",
            "timestamp":     pr["created_at"],
            "payload": {
                "node_type":  "pull_request",
                "attributes": {
                    "title":     pr["title"],
                    "state":     pr["state"],
                    "additions": pr["additions"],
                    "deletions": pr["deletions"],
                    "author":    pr["user"]["login"]
                },
                "relations": [{
                    "target": f"node:{tenant_id}:repo_{payload['repository']['id']}",
                    "type": "belongs_to"
                }]
            },
            "schema_version": "2.3",
            "metadata": {
                "source":      "github_webhook",
                "delivery_id": request.headers.get("X-GitHub-Delivery")
            }
        })
```

### 8.3 Jira REST Poller

**Strategy**: 30s interval, JQL cursor, 100 issues/page, exponential backoff on 429.

```python
class JiraPoller:
    def __init__(self, config, redis_client):
        self.config        = config
        self.redis         = redis_client
        self.last_poll_time = self.load_cursor()
        self.backoff_delay  = 1

    async def poll(self):
        # Jira expects "YYYY-MM-DD HH:MM" format, not ISO8601 T-separator
        jql = (
            f'updated >= "{self.last_poll_time.strftime("%Y-%m-%d %H:%M")}"'
            ' ORDER BY updated ASC'
        )
        start_at, total = 0, 1
        while start_at < total:
            async with self.session.get(
                f"{self.config.base_url}/rest/api/2/search",
                params={"jql": jql, "startAt": start_at, "maxResults": 100, "expand": "changelog"}
            ) as resp:
                if resp.status == 429:
                    await self.backoff(resp.headers.get("Retry-After"))
                    continue
                if resp.status == 200:
                    self.backoff_delay = 1  # Reset on success
                data  = await resp.json()
                total = data["total"]
                for issue in data["issues"]:
                    revision = f"{issue['id']}:{issue['fields']['updated']}"
                    if await self.is_seen(revision):
                        continue
                    await self.ingest(self.transform_to_chronos(issue))
                    await self.mark_seen(revision)
                start_at += len(data["issues"])
        self.save_cursor(datetime.utcnow())

    async def backoff(self, retry_after=None):
        """Does NOT reset backoff_delay â€” reset happens on success in poll()."""
        delay  = int(retry_after) if retry_after else min(self.backoff_delay, 60)
        if not retry_after:
            self.backoff_delay *= 2  # Exponential increase
        jitter = delay * 0.2 * (2 * random.random() - 1)
        await asyncio.sleep(delay + jitter)

    async def is_seen(self, revision):
        return await self.redis.get(f"jira:seen:{self.config.tenant_id}:{revision}") is not None

    async def mark_seen(self, revision):
        await self.redis.setex(f"jira:seen:{self.config.tenant_id}:{revision}", 604800, "1")

    def transform_to_chronos(self, issue):
        return {
            "event_id":      generate_uuidv7(),
            "aggregate_id":  f"node:{self.config.tenant_id}:jira_{issue['id']}",
            "event_type":    "node_created" if issue["fields"]["created"] == issue["fields"]["updated"] else "node_updated",
            "timestamp":     issue["fields"]["updated"],
            "payload": {
                "node_type":  "jira_issue",
                "attributes": {
                    "key":          issue["key"],
                    "summary":      issue["fields"]["summary"],
                    "status":       issue["fields"]["status"]["name"],
                    "priority":     issue["fields"]["priority"]["name"],
                    "story_points": issue["fields"].get("customfield_10016"),
                    "assignee":     issue["fields"]["assignee"]["displayName"] if issue["fields"]["assignee"] else None
                }
            },
            "schema_version": "2.3",
            "metadata": {"source": "jira_rest", "jira_id": issue["id"]}
        }
```

### 8.4 Slack Events API Handler

```python
class SlackEventHandler:
    def __init__(self, signing_secret: str, tenant_id: str):
        self.signing_secret = signing_secret
        self.tenant_id = tenant_id  # Required for aggregate_id construction

    async def handle(self, request):
        if not self.verify_signature(request):
            raise HTTPException(401, "Invalid signature")
        body  = await request.json()
        if body.get("type") == "url_verification":
            return {"challenge": body["challenge"]}
        event = body.get("event", {})
        if event.get("bot_id") or event.get("subtype"):
            return {"status": "ignored"}
        await ingest_event(self.transform(event))
        return {"status": "processed"}

    def transform(self, event):
        return {
            "event_id":      generate_uuidv7(),
            "aggregate_id":  f"node:{self.tenant_id}:slack_{event['ts']}",
            "event_type":    "relation_added" if event.get("type") == "reaction_added" else "node_created",
            "timestamp":     datetime.fromtimestamp(float(event["ts"])).isoformat(),
            "payload": {
                "node_type":  "slack_message",
                "attributes": {
                    "text":      event.get("text", ""),
                    "user":      event.get("user"),
                    "channel":   event.get("channel"),
                    "reaction":  event.get("reaction"),
                    "thread_ts": event.get("thread_ts")
                },
                "relations": [
                    {"target": f"node:{self.tenant_id}:slack_user_{event.get('user')}", "type": "authored_by"},
                    {"target": f"node:{self.tenant_id}:slack_channel_{event.get('channel')}", "type": "posted_in"}
                ]
            },
            "schema_version": "2.3",
            "metadata": {"source": "slack_events_api", "team_id": event.get("team")}
        }
```

### 8.5 gRPC Service Definition

```protobuf
syntax = "proto3";
package chronos.v1;

service KnowledgeGraphService {
  rpc IngestEvent    (EventRequest)   returns (EventResponse);
  rpc GetEmbedding   (EmbeddingRequest) returns (EmbeddingResponse);
  rpc StreamEvents   (StreamRequest)  returns (stream Event);
  rpc QueryCausal    (CausalRequest)  returns (CausalResponse);
}

message EmbeddingResponse {
  string node_id               = 1;
  repeated float embedding     = 2;
  int32  version               = 3;
  bool   dim_change_in_progress = 4;
  bool   is_stale              = 5;
}
```

---

## 9. LISTEN/NOTIFY IPC Contract

### 9.1 Channels

| Channel | Trigger | Payload |
|---------|---------|---------|
| `events_new` | New event committed | `{event_id, aggregate_id, event_type}` |
| `embedding_updated` | Recomputation complete | `{node_id, version, dimensions}` |
| `snapshot_created` | New snapshot available | `{aggregate_id, snapshot_id, sequence_number}` |
| `dimension_change` | Dimension changing | `{old_dimension, new_dimension, version}` |

### 9.2 Notification Payload Format

```json
{
  "channel":        "embedding_updated",
  "timestamp":      "2026-03-27T09:03:00Z",
  "payload":        { "node_id": "node:tenant1:task_123", "version": 42, "dimensions": 64 },
  "correlation_id": "batch_abc123"
}
```

### 9.3 Consumer Implementation

```python
import asyncpg, asyncio, json, logging

logger = logging.getLogger(__name__)

class IPCConsumer:
    def __init__(self, dsn: str, redis_client):
        self.dsn                  = dsn
        self.redis                = redis_client
        self.channels             = ["events_new", "embedding_updated",
                                     "snapshot_created", "dimension_change"]
        self.fallback_poll_interval = 60
        self.last_known_version   = 0

    async def listen(self):
        conn = await asyncpg.connect(self.dsn)
        for channel in self.channels:
            await conn.add_listener(channel, self.handle_notification)
        asyncio.create_task(self.fallback_poller())
        while True:
            await asyncio.sleep(1)

    def handle_notification(self, connection, pid, channel, payload):
        try:
            msg     = json.loads(payload)
            handler = getattr(self, f"handle_{channel}")
            asyncio.create_task(handler(msg))
        except Exception as e:
            logger.error(f"Notification handling failed: {e}")

    async def handle_embedding_updated(self, message: dict):
        payload = message.get("payload", {})
        version = payload.get("version")
        if version:
            await self.invalidate_cache(payload.get("node_id"))
            self.last_known_version = version

    async def handle_dimension_change(self, message: dict):
        payload = message.get("payload", {})
        version = payload.get("version")
        new_dim = payload.get("new_dimension")
        old_dim = payload.get("old_dimension")
        logger.warning(f"Dimension changing {old_dim} â†’ {new_dim} (v{version})")
        await self.invalidate_cache(None)
        if version:
            self.last_known_version = version  # Prevents fallback poller spam
        await self.notify_clients_of_dimension_change(new_dim)

