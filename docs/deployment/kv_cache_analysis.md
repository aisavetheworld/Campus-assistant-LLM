# KV Cache Analysis — Campus Assistant (vLLM 0.8.5, L4 24GB)

## Setup

- Model: Qwen/Qwen2.5-7B-Instruct + DPO LoRA (rank=32), bfloat16
- gpu_memory_utilization: 0.88
- block_size: 16 tokens, enforce_eager: True

## Memory Breakdown (max_seq_len=4096)

| Component | VRAM |
|-----------|------|
| Model weights (7B × BF16) | ~14,000 MiB |
| KV cache pool | ~5,000 MiB |
| LoRA adapter + overhead | ~900 MiB |
| **Total used** | **19,922 MiB / 23,034 MiB (86.5%)** |
| Free | 2,652 MiB |

## KV Cache Pool Size vs max_seq_len

| max_seq_len | KV cache (tokens) | Blocks | VRAM used | Max theoretical concurrency |
|-------------|-------------------|--------|-----------|-----------------------------|
| 1024 | 45,856 | 2,866 | 19,922 MiB | ~44 reqs |
| 2048 | 45,824 | 2,864 | 19,922 MiB | ~22 reqs |
| 4096 | 45,824 | 2,864 | 19,922 MiB | ~11 reqs |
| 8192 | 45,824 | 2,864 | 19,922 MiB | ~5 reqs |

**Key finding: max_seq_len does not affect VRAM usage.** Pool size is fixed by:
```
KV cache pool = gpu_memory_utilization × total_VRAM − model_weights
```

max_seq_len only limits how many blocks a single request can occupy.

## KV Cache Formula

```
Per-token KV cache = num_layers × 2 × num_kv_heads × head_dim × dtype_bytes
                   = 28 × 2 × 8 × 128 × 2
                   = 114,688 bytes ≈ 112 KB / token

Per block (16 tokens) = 112 KB × 16 = 1.75 MB
Pool size = 45,824 tokens × 112 KB ≈ 4.97 GB
```

## Prefix Cache Hit Rate (from load test logs)

| Concurrency | Peak Prefix Cache Hit Rate |
|-------------|---------------------------|
| u=1 | 18–44% |
| u=4 | 60–75% |
| u=8 | 80–89% |
| u=16 | 91–95% |
| u=32 | **95–97%** |

RAG prompts share a large common prefix (system message + retrieved chunks), so vLLM reuses cached KV tensors across requests. At u=32, nearly all prompt computation is skipped.

## Generation Throughput Scaling

| Concurrency | Generation throughput |
|-------------|----------------------|
| u=1 | ~15 tok/s |
| u=8 | ~100 tok/s |
| u=16 | ~200 tok/s |
| u=32 | ~350 tok/s |

Linear scaling due to continuous batching — larger batches utilize GPU more efficiently.

## Peak KV Cache Usage (from load test)

| Concurrency | Peak KV cache usage |
|-------------|---------------------|
| u=1 | 3–4% |
| u=8 | 13–17% |
| u=16 | 20–26% |
| u=32 | 28–32% |

Even at 32 concurrent users, only 32% of the KV cache pool is used. The system is far from KV cache saturation.

## Notes

- Flash Attention eliminates O(n²) temporary memory for attention computation, so long sequences do not cause VRAM spikes during the forward pass.
- Setting max_seq_len beyond pool capacity (e.g., 65536 with 45,824-token pool) causes vLLM startup failure, not extra VRAM usage.
- `--enforce-eager` reduces KV cache pool vs default (45,824 vs 58,160 tokens) because CUDA graph optimization allows more aggressive memory packing.
