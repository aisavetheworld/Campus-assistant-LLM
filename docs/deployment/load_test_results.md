# Load Test Results — Campus Assistant (vLLM + LoRA, L4 24GB)

## Setup

- GPU: NVIDIA L4 24GB VRAM
- Model: Qwen/Qwen2.5-7B-Instruct + DPO LoRA (rank=32)
- Backend: vLLM 0.8.5.post1, bfloat16, max_model_len=4096
- FastAPI workers: 1
- Tool: Locust 2.44.0, 60s per run, wait_time=between(1,3)s

## Results

| Concurrency | Requests | RPS  | P50   | P75   | P95   | P99   | Max   | Failures |
|-------------|----------|------|-------|-------|-------|-------|-------|----------|
| 1           | 4        | 0.09 | 11.0s | 13.0s | 13.0s | 13.0s | 13.4s | 0        |
| 4           | 18       | 0.32 | 11.0s | 13.0s | 17.0s | 17.0s | 16.7s | 0        |
| 8           | 37       | 0.62 | 7.6s  | 14.0s | 18.0s | 18.0s | 18.0s | 0        |
| 16          | 65       | 1.09 | 7.9s  | 15.0s | 24.0s | 35.0s | 34.9s | 0        |
| 32          | 102      | 1.71 | 14.0s | 16.0s | 27.0s | 37.0s | 40.2s | 0        |

## Key Findings

1. **Zero failures at all concurrency levels** — system remains stable up to 32 concurrent users.

2. **Throughput scales with concurrency** — vLLM continuous batching processes multiple requests together; throughput grows from 0.09 → 1.71 RPS (19x) as concurrency increases from 1 → 32.

3. **P50 stays low** — even at 16 concurrent users P50 is 7.9s, because vLLM batches tokens from multiple requests into a single GPU kernel pass.

4. **P99 degrades sharply above u=16** — queue builds faster than it drains; P99 jumps from 18s (u=8) to 35s (u=16) to 37s (u=32).

5. **Sweet spot: u=8** — 0.62 RPS with P95=18s and P50=7.6s. Adequate for a campus assistant serving non-real-time queries.

## Interpretation

Generation is the bottleneck (~6–14s per request), not retrieval (17–22ms). Scaling options:
- **Quantization (INT4)**: reduces VRAM, enables larger batch → lower latency per token
- **Multi-GPU**: tensor parallelism to split model across GPUs
- **Streaming**: return tokens as generated to reduce perceived latency
