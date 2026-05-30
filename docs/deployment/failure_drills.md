# Failure Drills — Campus Assistant Serving System

## Environment

- GPU: NVIDIA L4 24GB
- Stack: FastAPI (port 8080) → vLLM 0.8.5 (port 8000) → Qwen2.5-7B-Instruct + DPO LoRA
- max_model_len: 4096, gpu_memory_utilization: 0.88

---

## Drill 1: Context Length Exceeded (OOM-like)

**Symptom:** User sends an extremely long query; request fails.

**Reproduction:**
```bash
python3 -c "
import httpx
long_text = 'hello ' * 5000
r = httpx.post('http://localhost:8000/v1/chat/completions',
    json={'model': 'campus-assistant',
          'messages': [{'role': 'user', 'content': long_text}],
          'max_tokens': 10},
    timeout=60)
print(r.status_code, r.text[:200])
"
```

**Observed behavior:**
- vLLM returns `400 BadRequestError`: "This model's maximum context length is 4096 tokens. However, you requested 5040 tokens."
- FastAPI propagates as `502`
- vLLM process stays alive — system does not crash

**Diagnosis:**
```bash
tail -f logs/vllm.log | grep "ValueError\|maximum context"
```
`finish_reason` is absent; `ValueError` appears in vLLM logs. Distinguish from true GPU OOM (process killed, nvidia-smi shows 100% usage).

**Resolution:**
1. Add query length guard in FastAPI before calling vLLM:
   ```python
   if len(query.split()) > 500:
       raise HTTPException(400, "Query too long (max 500 words)")
   ```
2. Or increase `--max-model-len 8192` (no VRAM cost, reduces max concurrency from ~11 to ~5).
3. True GPU OOM: reduce `--gpu-memory-utilization`, enable INT4 quantization, or reduce `--max-model-len`.

---

## Drill 2: vLLM Process Crash

**Symptom:** FastAPI returns `500 Internal Server Error` for all `/chat` requests.

**Reproduction:**
```bash
pkill -f vllm
curl -s -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I waive UC SHIP?", "top_k": 5}'
# → Internal Server Error
```

**Observed behavior:**
- `/health` shows `"vllm_status": "unreachable"`
- `/chat` returns 500 (ConnectError not surfaced as clean 503)

**Diagnosis:**
```bash
curl -s http://localhost:8080/health | python3 -m json.tool
# vllm_status: "unreachable" → vLLM is down
ps aux | grep vllm | grep -v grep
# no process → confirm crash
tail -50 logs/vllm.log
# find root cause (OOM / CUDA error / segfault)
```

**Resolution:**
```bash
cd /Campus-assistant-LLM
nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --served-model-name campus-assistant \
  --enable-lora --max-lora-rank 32 \
  --lora-modules campus-assistant=outputs/dpo_7b \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --max-model-len 4096 \
  --gpu-memory-utilization 0.88 \
  > logs/vllm.log 2>&1 &
```
Long-term: add a process supervisor (systemd / supervisord) for auto-restart.

---

## Drill 3: Latency Spike / Request Jitter

**Symptom:** Response times vary widely (10s–28s) across requests of similar apparent complexity.

**Reproduction:**
```bash
python3 -c "
import httpx, time, statistics
queries = [
    'What is CPT?',
    'How do I waive UC SHIP health insurance? Please provide all steps.',
    'Mailroom hours?',
    'What are all the requirements to apply for OPT including STEM OPT extension?',
    'W grade?',
]
latencies = []
for q in queries:
    t0 = time.time()
    r = httpx.post('http://localhost:8080/chat', json={'query': q, 'top_k': 5}, timeout=60)
    lat = (time.time() - t0) * 1000
    latencies.append(lat)
    print(f'{lat:.0f}ms | {q[:50]}')
print(f'Min={min(latencies):.0f}ms Max={max(latencies):.0f}ms Stdev={statistics.stdev(latencies):.0f}ms')
"
```

**Observed results:**

| Query | Latency | Type |
|-------|---------|------|
| What is CPT? | 10,886ms | short |
| How do I waive UC SHIP... | 18,399ms | long |
| Mailroom hours? | 12,171ms | short |
| What are all the requirements... | 28,152ms | long |
| W grade? | 15,482ms | short |

Min=10.9s  Max=28.2s  Stdev≈6.4s

**Diagnosis:**
- Check `generation_ms` in `/chat` response — jitter in generation side is normal (answer length varies)
- Check `retrieval_ms` — should be stable 17–22ms; spikes here indicate index or BM25 issue
- Check vLLM logs for `Waiting: N reqs` — queue buildup means concurrency limit reached

**Root cause:** Answer length drives generation time, not query length. Longer answers = more tokens = more time. This is expected behavior, not a system fault.

**Resolution:**
- Set `GEN_MAX_TOKENS` lower to cap maximum generation time
- Enable streaming (`stream=True`) to reduce perceived latency
- Continuous batching (already on) and prefix caching (hit rate up to 97%) already mitigate jitter under load

---

## Drill 4: Output Truncation

**Symptom:** Model answer ends mid-sentence; response appears incomplete.

**Reproduction:**
```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "campus-assistant",
    "messages": [{"role": "user", "content": "How do I waive UC SHIP? Give detailed steps."}],
    "max_tokens": 30
  }' | python3 -m json.tool
```

**Observed behavior:**
```
"content": "Waiving UC SHIP ... can be done under certain circumstances, such as if you have comparable coverage through another"
"finish_reason": "length"   ← truncation indicator
```

**Diagnosis:**
```bash
# Check finish_reason in vLLM response
# "stop"   → normal completion
# "length" → truncated by max_tokens limit
grep "GEN_MAX_TOKENS" app/main.py
echo $GEN_MAX_TOKENS
```

**Resolution:**
- Increase `GEN_MAX_TOKENS` (default: 512, recommended: 768–1024 for campus Q&A)
- Add truncation warning in FastAPI response when `finish_reason == "length"`
- Enable streaming so users see partial output immediately rather than waiting for a cut-off response

---

## Summary

| Failure | Signal | First Check | Fix |
|---------|--------|-------------|-----|
| Context too long | 400/502, ValueError in logs | `tail logs/vllm.log` | Query length guard in FastAPI |
| vLLM crash | 500 on all requests, `vllm_status: unreachable` | `ps aux \| grep vllm` | Restart vLLM; add process supervisor |
| Latency jitter | High stdev in response times | `generation_ms` in response | Cap `max_tokens`; enable streaming |
| Output truncation | Answer ends mid-sentence | `finish_reason: length` | Increase `GEN_MAX_TOKENS` |
