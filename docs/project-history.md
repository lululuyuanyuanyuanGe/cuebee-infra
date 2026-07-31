# Project history and commit dates

The project work represented here ran from April through July 2026. This local repository was reconstructed on 2026-08-18 from the implementation notes and project artifacts. Commit author and committer timestamps reproduce the actual milestone dates; they are not presented as the filesystem creation time.

| Date | Milestone | Evidence in repository |
|---|---|---|
| 2026-04-12 | Freeze event contract and tokenizer boundary | Schema validation, idempotent revision adapter, token Longest Common Prefix (LCP) tests |
| 2026-05-03 | Persistent transcript spine | Append, revise, commit, metadata relabel, close tests |
| 2026-05-24 | Mutable Key-Value (KV) tail and branches | Reference-counted blocks, rollback, Copy-on-Write (COW), precise invalidation |
| 2026-06-14 | Semantic scheduling and freshness | Deadline scoring, stale cancellation, overload shedding, final gate |
| 2026-06-29 | Multi-tenant speaker service | Voice Activity Detection (VAD), micro-batching, stable identity, worker replacement, autoscaler |
| 2026-07-12 | Runnable service and experiments | Hypertext Transfer Protocol (HTTP) interface, demo, trace replay, speaker load generator |
| 2026-07-28 | Documentation and acceptance pass | Architecture, integration contract, benchmark plan, reproducible checks |
| 2026-07-29 | Versioned vLLM protocol | Revision metadata across public input, frontend state, and engine process boundary |
| 2026-07-30 | Physical suffix rollback | Full-attention KV block retention/release and scheduler regression tests |
| 2026-07-31 | In-process bridge | Pinned fork submodule and bounded CueBee-to-vLLM revision stream |
| 2026-07-31 | Batched partial-block COW | Resident branch protocol, reference-safe scheduler materialization, Compute Unified Device Architecture (CUDA) valid-prefix copy, correctness tests, and L4 benchmark harness |

The dates can be audited with:

```bash
git log --reverse --format='%h %aI %cI %s'
```
