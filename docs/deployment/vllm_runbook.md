# vLLM Serving Runbook

Campus AI Assistant — DPO-tuned Qwen2.5-7B-Instruct

---

## 1. Prerequisites

**Python environment**

Use the project's virtual environment:

```bash
source .venv/bin/activate
pip install vllm
```

**GPU requirement**

The merged BF16 7B model requires approximately **14 GB VRAM**. An NVIDIA A100, A10G, L4, or equivalent card is recommended.

**Step 1 — Merge the LoRA adapter first**

The DPO LoRA adapter at `outputs/dpo_7b` must be merged into the base model before vLLM can serve it. vLLM does not support raw LoRA adapters at inference time.

```bash
python scripts/deploy/merge_lora_adapter.py
```

This writes the merged model (weights + tokenizer) to `outputs/deploy/dpo_7b_merged`.

---

## 2. Serving the Merged Model

Run the following command from the **repo root**:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model outputs/deploy/dpo_7b_merged \
  --served-model-name campus-assistant \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90
```

> **Note:** The `--dtype bfloat16` and `--max-model-len 4096` flags are critical for Qwen2.5 compatibility. Do not omit them.

The server exposes an OpenAI-compatible API at `http://localhost:8000/v1`. The FastAPI service connects to this endpoint.

---

## 3. Verify the Server is Running

**Health check**

```bash
curl http://localhost:8000/health
```

Expected response: `{"status":"ok"}` (or HTTP 200 with an empty body, depending on vLLM version).

**Test generation**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "campus-assistant",
    "messages": [{"role": "user", "content": "What is CPT?"}],
    "max_tokens": 200
  }'
```

A successful response returns a JSON object with `choices[0].message.content` containing the model's answer.

---

## 4. Python Client Test

Use this minimal script to test the vLLM server directly (bypassing the FastAPI layer):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="campus-assistant",
    messages=[{"role": "user", "content": "What is CPT?"}],
    max_tokens=200
)

print(response.choices[0].message.content)
```

Install the dependency if needed: `pip install openai`.

---

## 5. Common Failures and Fixes

### OOM (Out of Memory)

- **Symptom:** `CUDA out of memory` error, or the server crashes on startup
- **Cause:** The 7B BF16 model requires ~14 GB VRAM; the GPU does not have enough free memory
- **Fix:**
  - Reduce `--gpu-memory-utilization` to `0.85`
  - Or reduce `--max-model-len` to `2048`
  - Or consider AWQ INT4 quantization (Phase 5)

---

### max_model_len too large

- **Symptom:** `ValueError: ... exceeds the model's maximum context length`
- **Cause:** Qwen2.5-7B supports up to 128k context, but vLLM may still require an explicit limit to stay within available memory
- **Fix:** Add `--max-model-len 4096` to the serve command (or lower if still OOM)

---

### Tokenizer issue

- **Symptom:** `tokenizer_config.json not found` or encoding errors at inference time
- **Cause:** The tokenizer was not saved alongside the merged model weights
- **Fix:** Re-run `merge_lora_adapter.py` — it saves the tokenizer to `output_dir` automatically

---

### Adapter / merged model issue

- **Symptom:** vLLM fails to load the model, or the model produces garbled / incoherent output
- **Cause:** The LoRA adapter was **not** merged before serving; raw LoRA adapters cannot be loaded directly by vLLM
- **Fix:** Run `python scripts/deploy/merge_lora_adapter.py` first, then restart the server

---

### Port conflict

- **Symptom:** `OSError: [Errno 98] Address already in use`
- **Fix:**
  ```bash
  lsof -i :8000        # find the process occupying port 8000
  kill <PID>           # stop it
  ```
  Alternatively, start vLLM on a different port with `--port 8001` and update `VLLM_BASE_URL` in the FastAPI service accordingly.

---

### Slow first response

This is **expected behaviour**. vLLM compiles CUDA kernels on the first request (warm-up). Subsequent requests are significantly faster.

---

## 6. Environment and Logging

**Monitor GPU usage**

```bash
nvidia-smi -l 1
```

Refreshes every second. Look for GPU memory utilisation near the configured `--gpu-memory-utilization` ceiling once the model is loaded.

**Server logs**

vLLM logs to **stdout** by default. When running interactively, output appears directly in the terminal.

**Run in background**

```bash
mkdir -p logs
nohup python -m vllm.entrypoints.openai.api_server \
  --model outputs/deploy/dpo_7b_merged \
  --served-model-name campus-assistant \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  > logs/vllm.log 2>&1 &
echo "vLLM PID: $!"
```

Tail the log:

```bash
tail -f logs/vllm.log
```

To stop the background server, use the PID printed above:

```bash
kill <PID>
```
