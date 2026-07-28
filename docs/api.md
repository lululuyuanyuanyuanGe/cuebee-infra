# Local interface

The local server is intentionally small and dependency-free. It supports Speech-to-Text (STT) integration testing, demos, and trace replay; it is not an internet-facing production gateway. Request and response bodies use JavaScript Object Notation (JSON).

## Routes

| Method | Route | Purpose |
|---|---|---|
| GET | `/healthz` | Liveness |
| GET | `/metrics` | Prometheus-compatible metrics |
| GET | `/v1/sessions/{session_id}` | Transcript, cache, branch, and task snapshot |
| POST | `/v1/sessions/{session_id}/stt` | Apply one normalized STT event |
| POST | `/v1/sessions/{session_id}/tasks` | Create an analysis branch |
| POST | `/v1/scheduler/tick` | Execute one admitted task |
| POST | `/v1/speaker/chunks` | Admit one pulse-code modulation sample chunk |
| POST | `/v1/speaker/flush` | Dispatch ready speaker batches |

## STT event body

```json
{
  "segment_id": "segment-17",
  "revision": 4,
  "type": "revise_partial",
  "text": "budget is one hundred fifty thousand",
  "start_ms": 1200,
  "end_ms": 2600,
  "confidence": 0.88,
  "client_epoch": 2,
  "seq_no": 91
}
```

Partial and Final `text` is the complete current hypothesis for the segment, not a character delta.

## Analysis task body

```json
{
  "kind": "proactive_hint",
  "prompt": "give one short actionable hint",
  "deadline_ms": 300,
  "priority": 2,
  "max_output_tokens": 32
}
```

Task kinds are `user_query`, `proactive_hint`, `search_decision`, `memory_extraction`, and `context_summary`.
