# vLLM integration boundary

## What is implemented

The repository implements the event contract, transcript state machine, token-level rollback decision, logical Key-Value (KV) ownership, branch dependencies, admission policy, cancellation propagation, and final freshness gate. These components are engine-independent and are covered by deterministic tests.

The repository now pins a real vLLM fork as `third_party/vllm`. Branch `cuebee-v0.23.0` adds a versioned `StreamingRevision` input contract to the V1 engine, transports it through the engine process boundary, and applies suffix replacement in the scheduler. `KVCacheManager.rollback_request` retains complete valid prefix blocks, releases divergent physical suffix blocks, and recomputes only the partial boundary block plus the new suffix.

`VLLMRevisionBridge` converts CueBee token-level session updates into the fork-native asynchronous input stream. The deterministic backend remains the laptop default, while `OpenAICompatibleEngine` remains the unmodified Hypertext Transfer Protocol (HTTP) baseline.

## Pinned target

The integration target is vLLM `v0.23.0` with the V1 engine. The annotated upstream tag resolves to `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`; both the GitHub reference Application Programming Interface (API) and the vendored Git checkout were verified against that value.

The user's GitHub fork is currently named `agentKV` and remains a fork of `vllm-project/vllm`. The CueBee changes live only on its `cuebee-v0.23.0` branch; the default branch is not rewritten.

Primary upstream references:

- [vLLM v0.23.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.23.0)
- [vLLM scheduler interface](https://docs.vllm.ai/en/v0.23.0/api/vllm/v1/core/sched/interface/)
- [vLLM Automatic Prefix Caching example](https://docs.vllm.ai/en/v0.22.0/examples/features/automatic_prefix_caching/)
- [vLLM KV cache manager design](https://github.com/vllm-project/vllm/blob/main/docs/design/hybrid_kv_cache_manager.md)

## Implemented in-process bridge

The version-specific bridge must translate control-plane transitions without changing their semantics:

| CueBee transition | vLLM-side operation |
|---|---|
| Prefill stable or tentative suffix | Publish a `SessionRevisionUpdate` as vLLM `StreamingInput` tokens |
| Promote tentative tail | Advance the logical frontier; the next content update carries the frontier into vLLM |
| Roll back suffix | Round down to a complete scheduler block, retain prefix block identifiers, and free the physical suffix |
| Fork branch | Reuse complete prefixes through Automatic Prefix Caching (APC); logical Copy-on-Write (COW) remains in CueBee |
| Cancel stale task | Finish or abort the request and wait for cache cleanup acknowledgement |
| Complete branch | Release private blocks and decrement shared references |

The bridge does not assign one global validity value to a physical block shared by multiple logical consumers. Logical reference state remains in CueBee; physical allocation remains in vLLM. Physical suffix rollback currently rejects sliding-window, chunked-local, and recurrent state-cache groups because those layouts may already have discarded earlier blocks. The frozen Qwen3-4B reference uses the supported full-attention path.

## Verification gates

1. Pin the model, tokenizer revision, vLLM tag, container digest, driver, and Compute Unified Device Architecture (CUDA) version.
2. Run the same token trace through the local block model and the vLLM bridge; compare allocation, rollback point, and final token stream.
3. Inject revision while a request is waiting, scheduled, executing, and returning output.
4. Assert that every reference count returns to baseline after complete, cancel, and failure.
5. Require `stale_output_escape == 0` before any latency or throughput result is accepted.

The CPU scheduler suite verifies protocol serialization, monotonic versions, committed-frontier protection, exact prefix block retention, divergent block release, partial-block recomputation, and legacy append streaming. NVIDIA L4 correctness and performance gates remain pending; no latency result is claimed yet.
