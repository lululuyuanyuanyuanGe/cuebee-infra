# vLLM integration boundary

## What is already real

The repository implements the event contract, transcript state machine, token-level rollback decision, logical Key-Value (KV) ownership, branch dependencies, admission policy, cancellation propagation, and final freshness gate. These components are engine-independent and are covered by deterministic tests.

`OpenAICompatibleEngine` can run final prompts against a vLLM completion server today. With Automatic Prefix Caching (APC) enabled, stable prompt prefixes may be reused by upstream vLLM. This path is useful as the V1 baseline, but the Hypertext Transfer Protocol (HTTP) interface does not expose physical block promotion, rollback, or branch reference ownership.

## Pinned target

The integration target is vLLM `v0.23.0` with the V1 engine. The official release page identifies the tag at commit `91df0fa`; an earlier CueBee planning document recorded `0fc695f`. The vendor checkout must verify `git rev-parse v0.23.0` before a patch or benchmark is claimed. The repository intentionally records this discrepancy rather than silently choosing a commit.

Primary upstream references:

- [vLLM v0.23.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.23.0)
- [vLLM scheduler interface](https://docs.vllm.ai/en/v0.23.0/api/vllm/v1/core/sched/interface/)
- [vLLM Automatic Prefix Caching example](https://docs.vllm.ai/en/v0.22.0/examples/features/automatic_prefix_caching/)
- [vLLM KV cache manager design](https://github.com/vllm-project/vllm/blob/main/docs/design/hybrid_kv_cache_manager.md)

## Required in-process bridge

The version-specific bridge must translate control-plane transitions without changing their semantics:

| CueBee transition | vLLM-side operation |
|---|---|
| Prefill stable or tentative suffix | Add or continue the corresponding request tokens |
| Promote tentative tail | Mark logical references committed without recomputing tokens |
| Roll back suffix | Release only divergent request blocks; preserve the valid prefix |
| Fork branch | Share complete prefix blocks and perform COW for a partial last block |
| Cancel stale task | Finish or abort the request and wait for cache cleanup acknowledgement |
| Complete branch | Release private blocks and decrement shared references |

The bridge must not assign one global validity value to a physical block shared by multiple logical consumers. Logical reference state remains in CueBee; physical allocation remains in vLLM.

## Verification gates

1. Pin the model, tokenizer revision, vLLM tag, container digest, driver, and Compute Unified Device Architecture (CUDA) version.
2. Run the same token trace through the local block model and the vLLM bridge; compare allocation, rollback point, and final token stream.
3. Inject revision while a request is waiting, scheduled, executing, and returning output.
4. Assert that every reference count returns to baseline after complete, cancel, and failure.
5. Require `stale_output_escape == 0` before any latency or throughput result is accepted.

Until these gates run on the target checkout, the repository describes the physical bridge as pending rather than claiming a completed vLLM fork.

