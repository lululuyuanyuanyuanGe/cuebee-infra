# vLLM integration boundary

## What is implemented

The repository implements the event contract, transcript state machine, token-level rollback decision, logical Key-Value (KV) ownership, branch dependencies, admission policy, cancellation propagation, and final freshness gate. These components are engine-independent and are covered by deterministic tests.

The repository now pins a real vLLM fork as `third_party/vllm`. Branch `cuebee-v0.23.0` adds versioned `StreamingRevision` and `StreamingFork` input contracts to the V1 engine, transports them through the engine process boundary, and applies suffix replacement or branch materialization in the scheduler. `KVCacheManager.rollback_request` retains complete valid prefix blocks, releases divergent physical suffix blocks, and recomputes only the partial boundary block plus the new suffix.

When a branch forks inside an incomplete cache block, the scheduler shares every complete block by reference, allocates one private destination block, and emits a `PartialBlockCopyPlan` containing the source block, destination block, and valid-token count. The model runner groups plans from concurrent Sessions by cache group. One `batched_partial_block_copy` Compute Unified Device Architecture (CUDA) kernel launch per compatible layout copies the valid Key-Value (KV) prefix across all layers while leaving each destination tail private and untouched.

`VLLMRevisionBridge` converts CueBee token-level session updates and branch forks into the fork-native asynchronous input stream. The deterministic backend remains the laptop default, while `OpenAICompatibleEngine` remains the unmodified Hypertext Transfer Protocol (HTTP) baseline.

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
| Fork at a block boundary | Increment physical references for complete resident source blocks |
| Fork inside a partial block | Share complete blocks, allocate one private block, and batch-copy only the valid KV prefix on the Graphics Processing Unit (GPU) |
| Cancel stale task | Finish or abort the request and wait for cache cleanup acknowledgement |
| Complete branch | Release private blocks and decrement shared references |

The copy has an explicit lifetime barrier. The scheduler pins both source and destination blocks after materialization and releases those extra references only after the model runner returns. A concurrent source revision, branch cancellation, or branch preemption therefore cannot recycle either block while the asynchronous GPU copy is queued.

The bridge does not assign one global validity value to a physical block shared by multiple logical consumers. Logical reference state remains in CueBee; physical allocation remains in vLLM. Physical suffix rollback and partial-block COW currently require resident, tokenized, full-attention requests with the same Low-Rank Adaptation (LoRA) adapter and cache salt. KV connectors, multimodal prompts, per-token-head quantization, NVFP4 KV cache, and recurrent state-cache groups fail closed. The reference deployment uses one Data Parallel (DP) rank; a multi-DP router must colocate source and branch requests before this path is enabled. The frozen Qwen3-4B reference uses the supported full-attention path.

## Kernel and benchmark entry points

The implementation is split across the scheduler, model runner, and stable C++/CUDA extension:

- `vllm/v1/core/single_type_kv_cache_manager.py`: full-block sharing, private partial allocation, and temporary reference pins;
- `vllm/v1/worker/utils.py`: layout discovery and cross-Session plan batching;
- `csrc/libtorch_stable/cache_kernels.cu`: vectorized valid-prefix copy kernel;
- `tests/kernels/test_cache_kernels.py`: GPU correctness check that the prefix is copied and the private tail is unchanged;
- `benchmarks/kernels/benchmark_partial_block_cow.py`: batched kernel versus per-branch tensor-copy launch baseline.

Run the microbenchmark from the vLLM checkout on an NVIDIA GPU:

```bash
.venv/bin/python benchmarks/kernels/benchmark_partial_block_cow.py \
  --batch-size 128 --num-layers 36 --block-size 16
```

The benchmark emits measured median microseconds and speedup. No number is recorded in this document until the kernel is compiled and run on the frozen L4 environment.

## Verification gates

1. Pin the model, tokenizer revision, vLLM tag, container digest, driver, and Compute Unified Device Architecture (CUDA) version.
2. Run the same token trace through the local block model and the vLLM bridge; compare allocation, rollback point, and final token stream.
3. Inject revision while a request is waiting, scheduled, executing, and returning output.
4. Assert that every reference count returns to baseline after complete, cancel, and failure.
5. Require `stale_output_escape == 0` before any latency or throughput result is accepted.

The Central Processing Unit (CPU) scheduler suite verifies protocol serialization, monotonic versions, committed-frontier protection, exact prefix block retention, divergent block release, partial-block recomputation, branch block ownership, copy-plan batching metadata, and legacy append streaming. The CUDA correctness test and L4 performance gate remain pending on GPU hardware; no latency result is claimed yet.
