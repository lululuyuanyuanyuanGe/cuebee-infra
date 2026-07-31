# CueBee Inference Infrastructure

CueBee is a real-time inference control plane for long-lived audio conversations. It combines:

- multi-tenant speaker embedding serving with Voice Activity Detection (VAD), length buckets, cross-session micro-batching, stable speaker identities, and an audio-load autoscaler;
- a stateful Large Language Model (LLM) runtime for revisable Speech-to-Text (STT) input, with a committed conversation spine, tentative tail rollback, Copy-on-Write (COW) branches, semantic scheduling, and a final freshness gate.

This repository implements the system semantics around vLLM. It does not rebrand upstream PagedAttention, continuous batching, or Automatic Prefix Caching (APC) as CueBee features.

## Current status

| Area | Implemented now | Hardware validation still required |
|---|---|---|
| Transcript runtime | Idempotent events, token-level diff, stable/tentative state, configurable tokenizer guard | Replay production STT traces after redaction |
| Key-Value (KV) lifecycle | Logical references, COW, branch invalidation, and pinned vLLM physical suffix rollback | Validate hybrid-cache layouts and L4 traces |
| Scheduling | Version, deadline, foreground, age, prefix reuse, cost, overload shedding | Tune weights on NVIDIA L4 traces |
| Output safety | Abort propagation plus freshness gate | Inject abort/output races under vLLM load |
| Speaker serving | VAD, deadline-aware micro-batching, stable Session Store, worker replacement, autoscaler | Replace deterministic worker with the frozen Open Neural Network Exchange (ONNX) model and collect latency |
| Interfaces | JavaScript Object Notation (JSON) over Hypertext Transfer Protocol (HTTP), metrics, demos, trace replay, container | Add production authentication, Transport Layer Security (TLS), and durable state |

No throughput or latency claim in this repository should be treated as an L4 result until the benchmark metadata and raw request records are checked in.

## Quick start

Requires Python 3.11 or newer and has no mandatory third-party runtime dependency.

```bash
make test
make demo
make benchmark-stt
make benchmark-speaker
make serve
```

If the default `python3` is older, pass a newer interpreter explicitly:

```bash
make test PYTHON=/path/to/python3
```

The server listens on `127.0.0.1:8080` by default. A minimal transcript flow is:

```bash
curl -s http://127.0.0.1:8080/v1/sessions/demo/stt \
  -H 'content-type: application/json' \
  -d '{"segment_id":"seg-1","revision":1,"type":"append_partial","text":"budget is fifty"}'

curl -s http://127.0.0.1:8080/v1/sessions/demo/stt \
  -H 'content-type: application/json' \
  -d '{"segment_id":"seg-1","revision":2,"type":"commit_final","text":"budget is one hundred fifty"}'

curl -s http://127.0.0.1:8080/v1/sessions/demo/tasks \
  -H 'content-type: application/json' \
  -d '{"kind":"proactive_hint","prompt":"give one short suggestion","deadline_ms":300}'

curl -s -X POST http://127.0.0.1:8080/v1/scheduler/tick \
  -H 'content-type: application/json' -d '{}'
```

The default model backend and speaker worker are deterministic so the complete state machine runs on a laptop. `OpenAICompatibleEngine` sends final prompts to a compatible endpoint. The `third_party/vllm` fork and `VLLMRevisionBridge` provide the in-process versioned input and physical KV suffix rollback path.

## Architecture

```mermaid
flowchart LR
    Audio[Audio chunks] --> VAD[VAD]
    VAD --> Batcher[Length and deadline batcher]
    Batcher --> Worker[Stateless embedding workers]
    Worker --> Store[Session speaker store]
    Store --> Timeline[Versioned transcript timeline]

    STT[Partial and Final STT events] --> Adapter[Revision adapter]
    Adapter --> Timeline
    Timeline --> Spine[Committed spine and tentative tail]
    Spine --> KV[Logical KV manager]
    KV --> Branch[Shared branch graph]
    Branch --> Scheduler[Version and deadline scheduler]
    Scheduler --> Engine[vLLM or deterministic engine]
    Engine --> Gate[Freshness gate]
    Gate --> Output[Hint, search, memory, summary]
```

The semantic controller runs before the upstream token scheduler. It decides which analysis is valid and worth admitting; vLLM remains responsible for token budgets, continuous batching, and model execution.

## Correctness invariants

- The commit frontier only moves forward.
- Ordinary Partial revisions never roll back committed tokens.
- A physical block is released only after all logical references are gone.
- Tail-dependent tasks are cancelled when their transcript version changes.
- Every user-visible or memory-writing output passes the freshness gate.
- Stable `spk_001` identifiers are separate from user-facing display names.
- `stale_output_escape` must remain zero.

## Repository map

```text
src/cuebee/
  api/                 local HTTP gateway
  speaker/             VAD, batching, workers, Session Store, autoscaler
  event_schema.py      versioned input contracts
  session_manager.py   committed spine and tentative tail
  kv_metadata.py       logical references and reference-counted block model
  branch_graph.py      COW task branches and dependency invalidation
  scheduler.py         semantic admission and overload policy
  freshness_gate.py    final output consistency check
  engine.py            deterministic and vLLM-compatible HTTP backends
  vllm_bridge.py       versioned in-process input stream for the pinned fork
  runtime.py           end-to-end orchestration
benchmarks/             STT trace replay and speaker load generation
demos/                  reproducible revision sequence
docs/                   architecture, integration, experiments, and history
tests/                  unit and integration coverage
third_party/vllm/       pinned vLLM fork branch with physical suffix rollback
```

See [architecture.md](docs/architecture.md), [vllm-integration.md](docs/vllm-integration.md), [benchmark-plan.md](docs/benchmark-plan.md), and [project-history.md](docs/project-history.md) for the implementation contract and evidence boundary.

## License

Apache License 2.0.
