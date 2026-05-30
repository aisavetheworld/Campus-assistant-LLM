# Project 3 Final Report — Serving System

## System Architecture

```
User → FastAPI (port 8080) → Redis Cache
                           ↓ (cache miss)
                        RAG Retrieval (FAISS + BM25)
                           ↓
                        vLLM (port 8000, Qwen2.5-7B + DPO LoRA)
                           ↓
                        Safety Validation
                           ↓
                        Response + Latency Breakdown
```

## Components

| Component | Detail |
|-----------|--------|
| Base model | Qwen/Qwen2.5-7B-Instruct |
| Adapter | DPO LoRA rank=32 (outputs/dpo_7b) |
| Inference | vLLM 0.8.5.post1, bfloat16 |
| GPU | NVIDIA L4 24GB |
| RAG | FAISS dense + BM25, alpha=0.8, 4098 chunks |
| Cache | Redis, TTL=3600s, key=SHA256(query+top_k) |

## Smoke Test Results (5/5 passed)

All 5 canonical campus queries answered correctly with `safety_passed=True`.
Avg total latency: 8,962ms (without cache).

## Load Test Results

| Concurrency | RPS  | P50   | P95   | P99   | Failures |
|-------------|------|-------|-------|-------|----------|
| 1           | 0.09 | 11.0s | 13.0s | 13.0s | 0        |
| 4           | 0.32 | 11.0s | 17.0s | 17.0s | 0        |
| 8           | 0.62 | 7.6s  | 18.0s | 18.0s | 0        |
| 16          | 1.09 | 7.9s  | 24.0s | 35.0s | 0        |
| 32          | 1.71 | 14.0s | 27.0s | 37.0s | 0        |

Sweet spot: u=8 (0.62 RPS, P95=18s, zero failures).

## KV Cache Analysis

- Pool size: 45,824 tokens (~5 GB) — fixed regardless of max_seq_len
- Peak usage: 32% at u=32 — far from saturation
- Prefix cache hit rate: up to **97%** at u=32 (RAG prompts share common prefix)
- Generation throughput: ~15 tok/s (u=1) → ~350 tok/s (u=32)

## Redis Cache Results

| Request | cache_hit | Latency |
|---------|-----------|---------|
| Cold (miss) | false | 11,827ms |
| Warm (hit) | true | 35ms |

**338x speedup** for repeated queries.

## Failure Drills

| Failure | Signal | Fix |
|---------|--------|-----|
| Context too long | 400 ValueError in vLLM logs | Query length guard in FastAPI |
| vLLM crash | `vllm_status: unreachable` on /health | Restart vLLM; add process supervisor |
| Latency jitter | High stdev in generation_ms | Cap max_tokens; enable streaming |
| Output truncation | `finish_reason: length` | Increase GEN_MAX_TOKENS |

## Documents

- [docs/deployment/project3_serving_plan.md](project3_serving_plan.md) — architecture
- [docs/deployment/vllm_runbook.md](vllm_runbook.md) — vLLM ops runbook
- [docs/deployment/load_test_results.md](load_test_results.md) — full load test data
- [docs/deployment/kv_cache_analysis.md](kv_cache_analysis.md) — KV cache experiment
- [docs/deployment/failure_drills.md](failure_drills.md) — failure reproduction & fixes
