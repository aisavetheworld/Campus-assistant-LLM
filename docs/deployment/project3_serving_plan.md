# Project 3: End-to-End Serving System — Architecture & Plan

## Goal

Deploy a production-grade HTTP serving system that exposes the full campus AI assistant pipeline as a REST API. The system combines the Project 2 RAG pipeline with the Project 1 DPO-aligned model (Qwen2.5-7B-Instruct + LoRA adapter), mediated by a FastAPI front-end and a vLLM inference backend. Responses are grounded in retrieved university documents and validated by a post-hoc safety layer.

---

## Architecture Diagram

```
                        ┌─────────────────────────────────────────────────────┐
                        │                   FastAPI App                       │
  HTTP POST /chat  ───► │  app/main.py                                        │
  {"query": "...",      │  - request parsing & validation                     │
   "top_k": 5}         │  - orchestrates pipeline stages                     │
                        │  - assembles final response                         │
                        └───────────────────┬─────────────────────────────────┘
                                            │
                        ┌───────────────────▼─────────────────────────────────┐
                        │                 RAG Retrieval                       │
                        │  scripts/rag/rag_answer.py                         │
                        │                                                     │
                        │  ┌─────────────────────────────────────────────┐   │
                        │  │  Query Expansion                            │   │
                        │  │  scripts/rag/query_expansion.py             │   │
                        │  └──────────────────┬──────────────────────────┘   │
                        │                     │                               │
                        │  ┌──────────────────▼──────────────────────────┐   │
                        │  │  Hybrid Retrieval (Dense + BM25)            │   │
                        │  │  scripts/rag/retrieve_hybrid.py             │   │
                        │  │  - FAISS IndexFlatIP  (α=0.8 dense weight)  │   │
                        │  │  - BM25Okapi          (α=0.2 sparse weight) │   │
                        │  └──────────────────┬──────────────────────────┘   │
                        │                     │                               │
                        │  ┌──────────────────▼──────────────────────────┐   │
                        │  │  Grounded Prompt Construction               │   │
                        │  │  rag_answer.build_grounded_prompt()         │   │
                        │  └──────────────────┬──────────────────────────┘   │
                        └─────────────────────┼───────────────────────────────┘
                                             │
                        ┌────────────────────▼────────────────────────────────┐
                        │              vLLM Inference Server                  │
                        │  (separate process, OpenAI-compatible API)          │
                        │  Base model : Qwen/Qwen2.5-7B-Instruct             │
                        │  Adapter    : outputs/deploy/dpo_7b_merged          │
                        │  (DPO LoRA pre-merged via merge_lora.py)            │
                        └────────────────────┬────────────────────────────────┘
                                             │
                        ┌────────────────────▼────────────────────────────────┐
                        │           Post-hoc Safety Validation                │
                        │  scripts/rag/answer_validators.py                   │
                        │  - content policy checks                            │
                        │  - legal-sensitivity flags (F-1/immigration)        │
                        └────────────────────┬────────────────────────────────┘
                                             │
  HTTP Response    ◄───────────────────────────
  {answer, sources, retrieval_metadata,
   safety_metadata, latency}
```

---

## Request Flow

1. **Client sends** `POST /chat` with JSON body `{"query": "...", "top_k": 5}` to the FastAPI app.
2. **FastAPI** (`app/main.py`) validates the request schema and records the request start timestamp.
3. **Query expansion** (`scripts/rag/query_expansion.py`) rewrites or augments the raw query to improve recall.
4. **Hybrid retrieval** (`scripts/rag/retrieve_hybrid.py`) runs dense search (FAISS, MiniLM-L6-v2 embeddings, α=0.8) and sparse search (BM25Okapi, α=0.2) in parallel, merges ranked results, deduplicates by source, and returns the top-k chunks.
5. **Grounded prompt construction** (`rag_answer.build_grounded_prompt`) formats retrieved chunks and the original query into a prompt that instructs the model to answer only from the provided context.
6. **vLLM inference server** receives the grounded prompt via the OpenAI-compatible `/v1/chat/completions` endpoint and generates a response using the merged DPO model.
7. **Post-hoc safety validation** (`scripts/rag/answer_validators.py`) checks the generated answer for policy violations, harmful content, and legal-sensitivity triggers (particularly F-1/immigration advice).
8. **FastAPI** assembles the final response payload and returns it to the client, including answer text, cited sources, retrieval metadata, safety metadata, and per-stage latency.

---

## Component List

| Component | File Path | Role |
|---|---|---|
| FastAPI app | `app/main.py` | HTTP entry point; request/response lifecycle; pipeline orchestration |
| RAG answer driver | `scripts/rag/rag_answer.py` | Top-level retrieval + prompt-building logic |
| Query expansion | `scripts/rag/query_expansion.py` | Rewrites query to improve retrieval recall |
| Hybrid retriever | `scripts/rag/retrieve_hybrid.py` | Dense (FAISS) + sparse (BM25) hybrid search; source dedup |
| Answer validators | `scripts/rag/answer_validators.py` | Post-hoc safety and legal-sensitivity validation |
| LoRA merge script | `scripts/merge_lora.py` (planned) | One-time offline merge of DPO LoRA adapter into base weights |
| vLLM server | separate process | OpenAI-compatible text generation; GPU inference |
| Merged model weights | `outputs/deploy/dpo_7b_merged` | Pre-merged Qwen2.5-7B + DPO LoRA, served by vLLM |
| FAISS index | `outputs/faiss_index/` | Dense vector store for 4098 campus document chunks |
| BM25 index | `outputs/bm25_index/` | Sparse term-frequency index for same corpus |

---

## Component Roles

### vLLM

vLLM is an open-source, high-throughput LLM inference server with PagedAttention-based KV-cache management. It exposes an OpenAI-compatible REST API (`/v1/chat/completions`), allowing the FastAPI app to call it with standard HTTP without a custom inference harness. Key reasons for choosing vLLM:

- Continuous batching handles concurrent requests without per-request model loading overhead.
- PagedAttention significantly reduces GPU memory fragmentation compared to naive KV-cache implementations.
- OpenAI-compatible interface allows straightforward integration and future swap-out.
- Supports LoRA-merged weights natively once the adapter is pre-merged offline.

The vLLM server is started as a separate process (`vllm serve outputs/deploy/dpo_7b_merged`) prior to FastAPI startup. FastAPI communicates with it via HTTP.

### FastAPI

FastAPI serves as the HTTP gateway and pipeline orchestrator. It handles:

- Input validation via Pydantic schemas.
- Routing (`POST /chat`, `GET /health`).
- Sequential invocation of the RAG retrieval stage, vLLM generation call, and safety validation.
- Per-stage latency measurement and structured response assembly.
- Error handling and HTTP status codes for downstream failures (e.g., vLLM unavailable, retrieval timeout).

FastAPI was chosen for its native async support (important for non-blocking I/O when calling vLLM over HTTP), automatic OpenAPI docs, and Pydantic-based validation.

### RAG Retrieval

The retrieval layer (Project 2, frozen config) grounds model generation in verified campus documents, reducing hallucination. Configuration:

- Corpus: 673 sources, 4098 chunks
- Chunk size: 512 words, 50-word overlap
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Dense index: FAISS `IndexFlatIP` (inner-product / cosine similarity)
- Sparse index: BM25Okapi
- Hybrid alpha: 0.8 (dense) / 0.2 (sparse)
- Query expansion: enabled
- Source deduplication: enabled
- Reranker: disabled by default

Retrieval latency is expected to add 100–500ms per request, depending on query expansion time and index size.

### Post-hoc Safety Validation

`scripts/rag/answer_validators.py` applies rule-based and/or model-based checks to the generated answer before it is returned to the client. Its role is to:

- Detect responses that violate content policy (harmful, offensive, or out-of-scope answers).
- Flag answers touching legally sensitive topics, particularly F-1 visa and immigration advice, where the system should disclaim that it is not a substitute for official legal counsel.
- Optionally suppress or modify the answer if validation fails.

Safety metadata (flag status, triggered rules) is included in the response payload so clients can render appropriate disclaimers.

---

## Planned Benchmark Metrics (Phase 4)

Benchmarking will be conducted after the serving system is functionally complete. Target metrics:

### Throughput & Latency

| Metric | Description |
|---|---|
| QPS (queries per second) | Sustained throughput at each concurrency level |
| P50 latency | Median end-to-end response time |
| P95 latency | 95th-percentile end-to-end response time |
| P99 latency | 99th-percentile end-to-end response time |
| Error rate | Fraction of requests returning 4xx/5xx |

### Per-Stage Latency Breakdown

| Stage | Metric |
|---|---|
| RAG retrieval | Time from query received to chunks returned |
| vLLM generation | Time from prompt sent to first token / full response |
| Total end-to-end | Wall-clock time from HTTP request to HTTP response |

### Resource Utilization

| Metric | Description |
|---|---|
| GPU memory peak (GB) | Peak VRAM usage during inference |
| GPU utilization (%) | Average GPU compute utilization under load |

### Concurrency Levels

Tests will be run at: **1, 4, 8, 16** concurrent clients (and optionally **32** if GPU memory permits).

---

## Future Optimization Plan (Phase 5 — Planned)

These optimizations are not in scope for Phase 3 or 4. They are documented here for future reference.

### Model Compression

- **AWQ INT4 / GPTQ INT4 quantization:** Reduce the 7B model's VRAM footprint from ~14GB (fp16) to ~4–7GB, enabling deployment on smaller GPUs (e.g., T4 16GB) and increasing effective batch size. Requires offline quantization before serving.

### Caching

- **Redis FAQ cache:** For high-frequency, low-variance queries (e.g., "when is tuition due?"), cache the full response keyed on a normalized query hash. Target cache hit rate: >20% for FAQ traffic. Requires a Redis sidecar and a cache-lookup step before RAG retrieval.

### Reliability & Rate Control

- **Rate limiting (HTTP 429):** Reject requests above a per-IP or global QPS threshold with `429 Too Many Requests` to protect GPU resources during traffic spikes.
- **Request timeout and fallback:** If vLLM does not respond within a configurable timeout (e.g., 15s), return a graceful fallback message rather than hanging the client.
- **Smaller model fallback:** Under sustained high load, route overflow requests to a smaller model (e.g., Qwen2.5-1.5B-Instruct) to maintain availability at reduced quality.

---

## Known Risks

| Risk | Severity | Mitigation |
|---|---|---|
| GPU OOM for 7B model | High | Requires ≥14GB VRAM (fp16). Use A100/L4 for serving; or apply INT4 quantization (Phase 5). |
| `max_model_len` misconfiguration | Medium | Qwen2.5 has a 128k context window; vLLM defaults may not allocate sufficient KV-cache pages. Must set `--max-model-len` explicitly (e.g., 4096 or 8192) to avoid OOM. |
| LoRA adapter merge required | Medium | vLLM does not support unmerged LoRA adapters in all configurations. The DPO adapter (`outputs/dpo_7b`) must be merged offline into `outputs/deploy/dpo_7b_merged` before starting the vLLM server. |
| Retrieval latency | Low-Medium | Hybrid retrieval with query expansion adds 100–500ms per request. Under high concurrency this compounds. Index caching and async retrieval can mitigate. |
| Grounded prompt exceeds context window | Medium | If top-k=5 chunks are verbose, the grounded prompt may approach or exceed the configured `max_model_len`. Chunk length must be bounded and prompt templates must be audited against the context limit. |
| vLLM cold start time | Low | First request after server startup may be slow due to CUDA kernel compilation and weight loading. Health-check polling should confirm readiness before accepting traffic. |
| F-1/immigration legal sensitivity | High | Any answer touching F-1 visa rules, immigration status, or OPT/CPT eligibility carries legal risk. The safety validator must flag these topics and append a disclaimer. The system must never present generated text as official legal advice. |
