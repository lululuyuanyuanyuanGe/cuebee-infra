# Benchmark and evidence plan

## Evidence labels

- FACT: verified product, model, or hardware fact.
- REFERENCE: frozen comparison environment, not a production claim.
- ESTIMATE: capacity calculation that still needs measurement.
- Service Level Objective (SLO): acceptance threshold, not a result.
- To Be Determined (TBD): missing measurement.

Generated laptop numbers are smoke-test evidence only. Final resume metrics must point to raw request records and immutable run metadata.

## Speaker serving ablation

The serving path uses Voice Activity Detection (VAD) before batching.

| Group | Configuration | Question |
|---|---|---|
| S0 | One model call per Session, no shared batch | Baseline cost and tail latency |
| S1 | Shared stateless workers, First-In First-Out batch | Benefit from shared instances |
| S2 | VAD, length bucket, earliest deadline dispatch | Padding and latency tradeoff |
| S3 | S2 plus external Session Store and autoscaler | Elasticity and identity continuity |

Report audio seconds per second, Real-time Factor (RTF), batch fill ratio, queue P50/P95/P99, model latency, attribution delay, worker utilization, scale response, and speaker attribution accuracy.

## Streaming runtime ablation

The baseline enables Automatic Prefix Caching (APC); CueBee adds logical Key-Value (KV) state and Copy-on-Write (COW) branches above it.

| Group | Configuration | Question |
|---|---|---|
| V0 | Submit full history for every update | Full-prefill baseline |
| V1 | Upstream APC | Native prefix reuse |
| V2 | V1 plus persistent committed spine | Session-level stable reuse |
| V3 | V2 plus tentative prefill and suffix rollback | Revision benefit and rollback cost |
| V4 | V3 plus COW branches and semantic scheduler | Multi-task sharing and stale compute |

Report Time to First Token (TTFT), Inter-Token Latency (ITL), end-to-end latency, prefill tokens, KV reuse, rollback tokens, stale compute, deadline misses, cancellation latency, memory, and goodput.

## Execution rules

- Warm the server for two minutes and run 128 warm-up requests.
- Each measured cell runs at least 15 minutes and 10,000 requests.
- Use three random seeds and compute percentiles from per-request rows.
- Record model revision, vLLM tag, container digest, driver, CUDA, input/output token distribution, request rate, concurrency, and cache configuration.
- Store raw JavaScript Object Notation Lines (JSONL) request records separately from aggregated tables.
- Report negative results, especially the revision rate at which speculative tentative prefill becomes net harmful.

## Commands available now

```bash
make benchmark-stt
make benchmark-speaker
```

`stt_trace_replay.py` reports the full-history token baseline, incremental prefill tokens, rollback tokens, and logical block state. `speaker_loadgen.py` exercises cross-session batch formation and emits batch and queue distributions. Both are harness smoke tests, not L4 performance claims.
