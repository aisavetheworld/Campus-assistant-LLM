# Demo Usage — Campus Assistant RAG

Gradio demo of the Project 2 RAG pipeline. Lets you (or a recruiter)
type a question and see the full pipeline result: retrieved chunks,
generated answer, validator results, and pipeline metadata.

## Quick start on Colab (GPU, recommended)

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Campus-assistant-LLM
!git pull

# Install dependencies (once per runtime)
!pip install -q gradio faiss-cpu sentence-transformers rank-bm25 pyyaml peft
!pip install -q -U "torchao>=0.16.0"

# Launch with a public share link
!python scripts/rag/demo_app.py --share
```

Click the public `https://*.gradio.live` URL Gradio prints. The link
stays alive while the Colab cell is running.

The first launch downloads Qwen2.5-7B-Instruct (~15 GB) — give it
1-2 minutes. The DPO LoRA adapter is loaded from `outputs/dpo_7b/`
(committed in repo).

## Variants

```bash
# Use base Qwen2.5-7B without DPO adapter (saves ~200 MB)
!python scripts/rag/demo_app.py --share --base_only

# Disable constraints (raw model output, no retry/fallback)
# (just untick the checkbox in the UI; no flag needed)
```

## Local on Apple Silicon (M1+, recommended for offline demo)

Native MLX backend, 4-bit quantized model (~4.5 GB on disk, ~25 tok/s on M4):

```bash
pip install mlx-lm gradio faiss-cpu sentence-transformers rank-bm25 pyyaml
python scripts/rag/demo_app_mlx.py
```

Open http://localhost:7860. First launch downloads
`mlx-community/Qwen2.5-7B-Instruct-4bit` (~4.5 GB).

This is the fastest and lightest path on a Mac — no CUDA, no PyTorch MPS
quirks, no LoRA adapter loading issues. Single query takes ~10–25 seconds
end-to-end on M4 (8–15 s generation + retrieval / validators ≪ 1 s).

## Local (CPU, retrieval-only — no LLM)

If you want to demo retrieval and inspect the grounded prompt without
spinning up a GPU:

```bash
pip install gradio faiss-cpu sentence-transformers rank-bm25 pyyaml
python scripts/rag/demo_app.py --no_generate
```

Open http://localhost:7860. The UI shows retrieved chunks and the
fully-built grounded prompt, but says "(Generation disabled)" in the
answer panel.

## UI overview

| Panel | Shows |
|-------|-------|
| Query input | Your question |
| Category dropdown | Optional — drives the `safe_escalation` check |
| Constraints toggle | When on: runs 11 validators + 1 retry + fallback. When off: raw model output. |
| Answer | The final answer text |
| Sources | Parsed `Sources:` section |
| Pipeline metadata | Top retrieval score, attempt count, fallback flag, validators-passed count |
| Validator results | All 11 checks with pass/fail + detail |
| Retrieved chunks | Top-5 hybrid retrieval with title / URL / score / first 300 chars |

## Suggested demo flow (for a recruiter / interview)

1. **Happy path** — pick `Am I eligible for CPT if I just started my first semester at UCSD?` (good retrieval, clean answer, all 11 validators pass).
2. **Show the chunks** — open the "Retrieved chunks" accordion. Point at the title / score / URL. "The model only sees these — never the full corpus."
3. **Show validators** — open "Validator results". 11 checks, all green. "These are rule-based, run in <50 ms after the LLM call."
4. **Trip a constraint** — flip the constraint toggle OFF, re-run the same query. The answer may add `"Note:"` or be looser. Re-enable to show the safety net.
5. **Low-confidence fallback** — try `Can I bring my pet lizard to live in the dorms?` — top retrieval score should be low, pipeline returns the fallback message. Show that the system *refuses to guess*.
6. **Was-zero-recall** — try `How much does it cost to live in the on-campus dorms at UCSD?` — explain this used to be a zero-recall query before we added the `cost to live → housing contract rates` query-expansion trigger.

## Architecture (text)

```
query
  │
  ▼
[1] hybrid retrieval (alpha=0.8 dense+BM25, 25 QE triggers, source-dedup)
  │   top-5 chunks  (each: source_id, title, section_title, url, text, score)
  ▼
[2] pre-gen confidence gate (top_score < 0.35 → fallback)
  │
  ▼
[3] Qwen2.5-7B [+DPO LoRA] generate
  │   system prompt = 6 STRICT RULES + rigid format
  │   user prompt   = retrieved context + question
  ▼
[4] 11 post-hoc validators (shared answer_validators.py)
  │     answer_not_empty, cites_source, uses_retrieved_context,
  │     no_hallucinated_deadline, no_hallucinated_fee, no_absolute_promise,
  │     safe_escalation, answer_has_steps, no_extra_notes,
  │     no_forbidden_claims, insufficient_context_behavior
  ▼
  all pass?  ──yes──▶ return answer
  │
  no
  ▼
[5] retry once — append failed-checks' fix_hints to user prompt, regenerate
  │
  ▼
  all pass?  ──yes──▶ return answer
  │
  no
  ▼
[6] return fallback message
```

The LLM (Qwen2.5-7B) is invoked **only in step 3 (and possibly step 5)**.
Steps 1, 2, 4, 6 are deterministic and run on CPU in milliseconds.

## Files referenced

- `scripts/rag/demo_app.py` — this Gradio app
- `scripts/rag/rag_answer.py` — pipeline functions (imported by the app)
- `scripts/rag/answer_validators.py` — the 11 validators
- `configs/rag_generation.yaml` — constraint config
- `configs/rag_query_expansion.json` — 25 QE triggers
- `data/rag/vector_store/` — FAISS index + chunk metadata (4098 chunks)
- `outputs/dpo_7b/` — DPO LoRA adapter (Project 1)
