# Project 3 — v0 Status (Phase 0/1/2 Skeleton)

**Date:** 2026-05-28
**Status:** Skeleton complete — not yet end-to-end tested (requires GPU)

---

## 1. Files Created

| File | Purpose |
|------|---------|
| `docs/deployment/project3_serving_plan.md` | Architecture and planning document for the full serving stack |
| `docs/deployment/vllm_runbook.md` | vLLM serving runbook with startup commands, env vars, and troubleshooting |
| `docs/deployment/project3_v0_status.md` | This file — v0 status and how-to-run guide |
| `app/__init__.py` | Empty package marker for the `app` Python package |
| `app/main.py` | FastAPI application with `GET /health` and `POST /chat`; includes RAG retrieval, grounded prompt construction, vLLM call, safety validation, and per-stage latency tracking |
| `scripts/deploy/merge_lora_adapter.py` | One-time script to merge a LoRA adapter into the base model and save the merged weights |
| `scripts/deploy/test_chat_api.py` | Smoke test script that sends 5 representative queries to the running FastAPI server |

---

## 2. How to Run vLLM

```bash
# Step 1: Merge the LoRA adapter (one-time setup)
cd /path/to/repo
python scripts/deploy/merge_lora_adapter.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --adapter_path outputs/dpo_7b \
  --output_dir outputs/deploy/dpo_7b_merged \
  --torch_dtype bfloat16

# Step 2: Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model outputs/deploy/dpo_7b_merged \
  --served-model-name campus-assistant \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90
```

vLLM exposes an OpenAI-compatible API on **port 8000**. FastAPI calls it internally.

---

## 3. How to Run FastAPI

```bash
# Install dependencies (if not already)
pip install -r requirements.txt

# Start FastAPI server (from repo root) — development mode with auto-reload
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

# Production mode (no reload):
uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1
```

FastAPI listens on **port 8080** and calls vLLM on port 8000 internally. Both servers must be running for `/chat` to succeed.

---

## 4. How to Test /chat

```bash
# Health check
curl http://localhost:8080/health

# Quick curl test
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I waive UC SHIP?", "top_k": 5}'

# Full smoke test — 5 representative queries
python scripts/deploy/test_chat_api.py

# Smoke test with custom FastAPI URL
python scripts/deploy/test_chat_api.py --url http://localhost:8080
```

---

## 5. What Is Implemented

- **Phase 0:** Architecture planning and documentation (`project3_serving_plan.md`, `vllm_runbook.md`) ✓
- **Phase 1 skeleton:** LoRA merge script (`scripts/deploy/merge_lora_adapter.py`) ✓
- **Phase 2 skeleton:** FastAPI app with `GET /health` and `POST /chat` (`app/main.py`) ✓
- RAG retrieval integration — all existing `scripts/rag/` utilities wired up ✓
- Grounded prompt construction (context chunks injected into system prompt) ✓
- vLLM call via OpenAI-compatible API ✓
- Post-hoc safety validation ✓
- Per-stage latency tracking (retrieval / generation / total) ✓
- Smoke test script with 5 representative query categories ✓

---

## 6. What Is Still TODO

- [ ] **Install vLLM:** `pip install vllm` — not in `requirements.txt` yet (heavy install, GPU-only)
- [ ] Actually run and test end-to-end (requires GPU with ~14 GB VRAM)
- [ ] Phase 3: Functional validation with the 5 query categories
- [ ] Phase 4: Benchmarking and load testing
- [ ] Phase 5: Quantization (AWQ INT4), Redis cache, rate limiting, timeout/fallback
- [ ] Phase 6: Final deployment documentation

---

## 7. Known Blockers

| Blocker | Notes |
|---------|-------|
| **vLLM requires CUDA GPU** | Apple Silicon (MPS) is NOT supported by vLLM. Must run on a Linux machine with an NVIDIA GPU. |
| **LoRA merge requires downloading base model** | Qwen2.5-7B-Instruct is ~14 GB from HuggingFace. Ensure `HF_HOME` points to a disk with enough space. |
| **FastAPI dependencies** | `fastapi`, `httpx`, `uvicorn` added to `requirements.txt`. Run `pip install -r requirements.txt` if not already installed. |
| **vLLM not in requirements.txt** | Must install separately: `pip install vllm`. Do not add to `requirements.txt` until GPU environment is confirmed. |
