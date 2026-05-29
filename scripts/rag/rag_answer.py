"""Grounded RAG answer pipeline using hybrid retrieval (alpha=0.8, QE enabled).

Generation constraints (configs/rag_generation.yaml):
  * Evidence-only: answer must come from retrieved official sources.
  * Mandatory Sources: section.
  * Anti-hallucination on dates/fees/policies/guarantees.
  * No absolute promises (definitely / guaranteed / will be approved …).
  * Insufficient-context fallback if top retrieval score < min_top_score.
  * Post-hoc validators (scripts/rag/answer_validators.py) + 1 retry with
    explicit fix hints. If retry still fails → fallback_message.
  * Format: direct answer → next steps → safety note → Sources.
  * No meta-commentary (Note: / Explanation: / As an AI language model / …).

Modes:
  Default             -- retrieve + print grounded prompt preview (no GPU)
  --build_prompt_only -- retrieve + save prompt JSON for batch GPU generation
  --generate          -- retrieve + generate answer with DPO checkpoint (requires GPU)
  --batch_generate    -- load pre-built prompts and generate all answers (Colab/GPU)

Usage (preview single query):
    python scripts/rag/rag_answer.py \
        --query "What is the deadline to waive UC SHIP?"

Usage (batch: process all eval-seed queries):
    python scripts/rag/rag_answer.py \
        --build_prompt_only \
        --eval_seed data/rag/rag_answer_eval_seed.json \
        --output_file outputs/rag_eval/grounded_prompts/batch.json

Usage (batch generate on Colab/GPU — recommended):
    python scripts/rag/rag_answer.py \
        --batch_generate \
        --prompts_file outputs/rag_eval/grounded_prompts/batch.json \
        --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
        --adapter_path outputs/dpo_7b \
        --output_file outputs/rag_eval/generated_answers_dpo.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from retrieve_hybrid import build_bm25, retrieve_hybrid, load_index_and_chunks
from query_expansion import QueryExpander
from answer_validators import (
    load_config as load_constraint_config,
    run_all_checks,
    collect_fix_hints,
    all_passed,
)

INDEX_DIR = Path("data/rag/vector_store")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EXPANSION_CONFIG = Path("configs/rag_query_expansion.json")
CONSTRAINT_CONFIG = Path("configs/rag_generation.yaml")
DEFAULT_ALPHA = 0.8
DEFAULT_TOP_K = 5

GROUNDED_SYSTEM_MESSAGE = (
    "You are an assistant helping students navigate campus administrative tasks at UC San Diego. "
    "You are NOT the university, a professor, the housing office, the insurance office, or a "
    "legal/medical advisor. Do not pretend to be an official office or give legal/medical advice.\n\n"
    "STRICT RULES:\n"
    "1. EVIDENCE-ONLY: Answer using ONLY the retrieved context below. If the context does not "
    "contain the answer, say so explicitly and recommend the relevant UCSD office.\n"
    "2. NO INVENTED FACTS: Do NOT state any deadline, date, week number, fee, dollar amount, "
    "eligibility outcome, approval decision, or guarantee that is not literally present in the "
    "retrieved context.\n"
    "3. NO ABSOLUTE PROMISES: Do NOT use words like 'definitely', 'guaranteed', '100%', "
    "'you are fine/safe/approved', 'will not affect'. Use cautious language ('may', 'typically').\n"
    "4. ALWAYS CITE: End every answer with a 'Sources:' section listing the source title(s) or "
    "URL(s) you used from the retrieved context.\n"
    "5. ESCALATE WHEN UNCERTAIN: For visa, immigration, medical, insurance, or enrollment-risk "
    "questions, recommend contacting the relevant office "
    "(ISEO for visa/CPT/OPT/F-1/J-1; Student Health Services for medical/immunization; "
    "UC SHIP/insurance office for insurance; Registrar/department for enrollment).\n"
    "6. NO META-COMMENTARY: Do NOT write 'Note:', 'Explanation:', 'Rationale:', "
    "'Disclaimer:', 'As an AI language model', or template labels like 'Human:' / 'Assistant:'.\n\n"
    "FORMAT:\n"
    "  <direct answer to the question>\n"
    "  <recommended next steps, if applicable>\n"
    "  <safety note recommending the official office, if applicable>\n"
    "  Sources:\n"
    "  - <Source title> (<URL>)\n"
)

SAFE_ESCALATION = (
    "I could not verify this from the retrieved official sources. "
    "Please check the official UCSD page or contact the relevant office."
)


def build_grounded_prompt(query: str, chunks: list[dict], extra_constraints: str = "") -> str:
    """Build the user-message prompt with retrieved context.

    extra_constraints: optional string appended after the question, used during
    retry to inject "Do not include X" hints from failed validators.
    """
    context_lines = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("title", "Official UCSD Source")
        url = chunk.get("url", "")
        section = chunk.get("section_title", "")
        source_id = chunk.get("source_id", "")
        header_bits = [title]
        if section:
            header_bits.append(section)
        if url:
            header_bits.append(url)
        if source_id:
            header_bits.append(f"id={source_id}")
        header = " — ".join(header_bits)
        context_lines.append(f"[Source {i}: {header}]\n{chunk['text']}")

    context_block = "\n\n".join(context_lines)
    prompt = (
        f"Retrieved context (the ONLY information you may rely on):\n{context_block}\n\n"
        f"Student question:\n{query}\n\n"
        "Answer using ONLY the retrieved context above. "
        "Do not invent deadlines, fees, policies, or guarantees. "
        "End with a 'Sources:' section listing the source titles or URLs."
    )
    if extra_constraints:
        prompt += f"\n\nAdditional constraints for this answer:\n{extra_constraints}"
    return prompt


def format_sources_header(chunks: list[dict]) -> str:
    lines = ["Retrieved sources:"]
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("title", "")
        url = chunk.get("url", "")
        score = chunk.get("score_hybrid", chunk.get("score", 0.0))
        line = f"  [{i}] {title}"
        if url:
            line += f" — {url}"
        line += f" (hybrid={score:.4f})"
        lines.append(line)
    return "\n".join(lines)


def load_retrieval_components(index_dir: Path, embedding_model: str, expansion_config: Path):
    model = SentenceTransformer(embedding_model)
    index, chunks = load_index_and_chunks(index_dir)
    bm25 = build_bm25(chunks)
    expander = QueryExpander(expansion_config)
    return model, index, chunks, bm25, expander


def run_retrieval(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    chunks: list[dict],
    bm25,
    expander: QueryExpander,
    top_k: int = DEFAULT_TOP_K,
    alpha: float = DEFAULT_ALPHA,
) -> list[dict]:
    return retrieve_hybrid(
        query=query,
        index=index,
        chunks=chunks,
        model=model,
        bm25=bm25,
        alpha=alpha,
        top_k=top_k,
        dedup_source=True,
        use_query_expansion=True,
        expander=expander,
    )


def load_generation_model(model_name_or_path: str, adapter_path: str):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
    except ImportError as e:
        raise ImportError(f"Generation requires transformers and peft: {e}")

    # Auto-detect best device + dtype:
    #   CUDA  -> fp16 + device_map="auto"
    #   MPS   -> bf16 + explicit .to("mps")  (Apple Silicon)
    #   CPU   -> fp32
    if torch.cuda.is_available():
        dtype = torch.float16
        device_map = "auto"
        device_label = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dtype = torch.bfloat16
        device_map = None
        device_label = "mps"
    else:
        dtype = torch.float32
        device_map = None
        device_label = "cpu"

    print(f"Loading model: {model_name_or_path} on {device_label} ({dtype})", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if device_map is None:
        model = model.to(device_label)
    if adapter_path and Path(adapter_path).exists():
        print(f"Loading adapter: {adapter_path}", file=sys.stderr)
        model = PeftModel.from_pretrained(model, adapter_path)
        if device_map is None:
            model = model.to(device_label)
    model.eval()
    return tokenizer, model


def generate_one(
    prompt: str,
    system_message: str,
    tokenizer,
    model,
    max_new_tokens: int,
    temperature: float,
) -> str:
    import torch
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_answer(
    prompt: str,
    system_message: str,
    model_name_or_path: str,
    adapter_path: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Single-query generation (loads model on each call — for interactive use only)."""
    tokenizer, model = load_generation_model(model_name_or_path, adapter_path)
    return generate_one(prompt, system_message, tokenizer, model, max_new_tokens, temperature)


def slim_chunks_for_prompt(chunks: list[dict], fields: list[str]) -> list[dict]:
    """Project each retrieved chunk to the fields required by the prompt schema.

    Configured via configs/rag_generation.yaml -> chunk_prompt_fields.
    The original list[dict] is kept verbatim under 'retrieved_chunks' for eval.
    """
    return [{k: c.get(k, "") for k in fields} for c in chunks]


def generate_with_constraints(
    query: str,
    chunks: list[dict],
    category: str,
    safety_expectation: str,
    forbidden_claims: list[str],
    low_confidence: bool,
    constraint_config: dict,
    tokenizer,
    model,
    max_new_tokens: int,
    temperature: float,
) -> dict:
    """Generate an answer with post-hoc validation + 1 retry + fallback.

    Returns dict:
        {
          "answer": final answer string,
          "validation": {check_name: {"passed", "detail", "fix_hint"}},
          "attempts": int (1 or 2),
          "fallback_triggered": bool,
          "fallback_reason": str,
        }
    """
    fallback_message = constraint_config.get("fallback_message", SAFE_ESCALATION).strip()
    max_retries = int(constraint_config.get("max_retries", 1))

    # --- Pre-gen confidence gate ---
    if low_confidence and constraint_config.get("fallback_on_low_confidence", True):
        validation = run_all_checks(
            answer=fallback_message,
            chunks=chunks,
            query=query,
            category=category,
            forbidden_claims=forbidden_claims,
            safety_expectation=safety_expectation,
            low_confidence=True,
            config=constraint_config,
        )
        return {
            "answer": fallback_message,
            "validation": validation,
            "attempts": 0,
            "fallback_triggered": True,
            "fallback_reason": "low_retrieval_confidence",
        }

    # --- Attempt 1 ---
    prompt = build_grounded_prompt(query, chunks)
    answer = generate_one(prompt, GROUNDED_SYSTEM_MESSAGE, tokenizer, model,
                          max_new_tokens, temperature)
    validation = run_all_checks(
        answer=answer, chunks=chunks, query=query, category=category,
        forbidden_claims=forbidden_claims, safety_expectation=safety_expectation,
        low_confidence=False, config=constraint_config,
    )
    attempts = 1
    if all_passed(validation):
        return {"answer": answer, "validation": validation, "attempts": attempts,
                "fallback_triggered": False, "fallback_reason": ""}

    # --- Retry with explicit fix hints ---
    for _ in range(max_retries):
        hints = collect_fix_hints(validation)
        if not hints:
            break
        retry_prompt = build_grounded_prompt(
            query, chunks, extra_constraints="\n".join(f"- {h}" for h in hints)
        )
        answer = generate_one(retry_prompt, GROUNDED_SYSTEM_MESSAGE, tokenizer, model,
                              max_new_tokens, temperature)
        validation = run_all_checks(
            answer=answer, chunks=chunks, query=query, category=category,
            forbidden_claims=forbidden_claims, safety_expectation=safety_expectation,
            low_confidence=False, config=constraint_config,
        )
        attempts += 1
        if all_passed(validation):
            return {"answer": answer, "validation": validation, "attempts": attempts,
                    "fallback_triggered": False, "fallback_reason": ""}

    # --- Retries exhausted → fallback ---
    failed_checks = [k for k, v in validation.items() if not v["passed"]]
    fallback_validation = run_all_checks(
        answer=fallback_message, chunks=chunks, query=query, category=category,
        forbidden_claims=forbidden_claims, safety_expectation=safety_expectation,
        low_confidence=True, config=constraint_config,
    )
    return {
        "answer": fallback_message,
        "validation": fallback_validation,
        "attempts": attempts,
        "fallback_triggered": True,
        "fallback_reason": f"validation_failed_after_retry: {failed_checks}",
    }


def build_prompt_record(
    item_id: str,
    category: str,
    query: str,
    chunks: list[dict],
    constraint_config: dict,
    safety_expectation: str = "",
    forbidden_claims: list[str] | None = None,
) -> dict:
    """Build a self-contained prompt record for offline / GPU-batch generation."""
    top_score = chunks[0].get("score_hybrid", 0.0) if chunks else 0.0
    min_top = float(constraint_config.get("min_top_score", 0.35))
    low_confidence = top_score < min_top

    prompt_fields = constraint_config.get("chunk_prompt_fields",
                                          ["source_id", "title", "section_title", "url", "text"])
    return {
        "id": item_id,
        "category": category,
        "query": query,
        "system_message": GROUNDED_SYSTEM_MESSAGE,
        "user_prompt": build_grounded_prompt(query, chunks),
        "retrieved_chunks": chunks,                              # full, for eval
        "prompt_chunks": slim_chunks_for_prompt(chunks, prompt_fields),
        "top_score": round(float(top_score), 4),
        "low_confidence": low_confidence,
        "safety_expectation": safety_expectation,
        "forbidden_claims": forbidden_claims or [],
        "generated_answer": None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grounded RAG answer pipeline.")
    parser.add_argument("--query", default="")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--index_dir", default=str(INDEX_DIR))
    parser.add_argument("--embedding_model", default=EMBEDDING_MODEL)
    parser.add_argument("--expansion_config", default=str(EXPANSION_CONFIG))
    parser.add_argument("--retrieval_score_threshold", type=float, default=None,
                        help="Override min_top_score from constraint config.")
    parser.add_argument("--constraint_config", default=str(CONSTRAINT_CONFIG),
                        help="Path to YAML with generation constraints.")
    parser.add_argument("--disable_constraints", action="store_true",
                        help="Disable post-hoc validators and retry/fallback (debug only).")
    # Prompt-only / batch mode
    parser.add_argument("--build_prompt_only", action="store_true",
                        help="Build and save grounded prompts without generation.")
    parser.add_argument("--eval_seed", default="",
                        help="Path to eval seed JSON for batch prompt building.")
    parser.add_argument("--output_file", default="",
                        help="Output file for --build_prompt_only mode.")
    # Generation
    parser.add_argument("--generate", action="store_true",
                        help="Run model generation (requires GPU and model checkpoint).")
    parser.add_argument("--batch_generate", action="store_true",
                        help="Load pre-built prompts JSON and generate all answers in one pass.")
    parser.add_argument("--prompts_file", default="outputs/rag_eval/grounded_prompts/batch.json",
                        help="Input prompts JSON for --batch_generate mode.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_path", default="outputs/dpo_7b",
                        help="Path to DPO LoRA adapter (PEFT). Pass '' to use base model.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    constraint_config = load_constraint_config(args.constraint_config)
    if args.retrieval_score_threshold is not None:
        constraint_config["min_top_score"] = args.retrieval_score_threshold

    # --- Batch generate mode: no retrieval needed, prompts already built ---
    if args.batch_generate:
        prompts_path = Path(args.prompts_file)
        if not prompts_path.exists():
            print(f"ERROR: prompts file not found: {prompts_path}", file=sys.stderr)
            sys.exit(1)
        records = json.loads(prompts_path.read_text())
        print(f"Loaded {len(records)} prompts from {prompts_path}", file=sys.stderr)

        tokenizer, gen_model = load_generation_model(args.model_name_or_path, args.adapter_path)
        results = []
        fallback_n = 0
        retry_n = 0
        for i, rec in enumerate(records, 1):
            print(f"[{i}/{len(records)}] {rec['id']}: {rec['query'][:60]}...", file=sys.stderr)

            if args.disable_constraints:
                answer = generate_one(
                    prompt=rec["user_prompt"],
                    system_message=rec.get("system_message", GROUNDED_SYSTEM_MESSAGE),
                    tokenizer=tokenizer, model=gen_model,
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                )
                results.append({
                    "id": rec["id"], "category": rec.get("category", ""),
                    "query": rec["query"], "generated_answer": answer,
                    "retrieved_chunks": rec.get("retrieved_chunks", []),
                    "constraints_enabled": False,
                })
                continue

            outcome = generate_with_constraints(
                query=rec["query"],
                chunks=rec.get("retrieved_chunks", []),
                category=rec.get("category", ""),
                safety_expectation=rec.get("safety_expectation", ""),
                forbidden_claims=rec.get("forbidden_claims", []),
                low_confidence=bool(rec.get("low_confidence", False)),
                constraint_config=constraint_config,
                tokenizer=tokenizer, model=gen_model,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            )
            if outcome["fallback_triggered"]:
                fallback_n += 1
            if outcome["attempts"] > 1:
                retry_n += 1
            results.append({
                "id": rec["id"],
                "category": rec.get("category", ""),
                "query": rec["query"],
                "generated_answer": outcome["answer"],
                "retrieved_chunks": rec.get("retrieved_chunks", []),
                "constraints_enabled": True,
                "attempts": outcome["attempts"],
                "fallback_triggered": outcome["fallback_triggered"],
                "fallback_reason": outcome["fallback_reason"],
                "validation": outcome["validation"],
            })

        out_path = Path(args.output_file) if args.output_file else Path("outputs/rag_eval/generated_answers.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(results)} answers → {out_path}", file=sys.stderr)
        if not args.disable_constraints:
            print(f"  Retries: {retry_n}/{len(results)}, Fallbacks: {fallback_n}/{len(results)}",
                  file=sys.stderr)
        return

    index_path = Path(args.index_dir) / "index.faiss"
    if not index_path.exists():
        print(f"ERROR: FAISS index not found at {index_path}. Run build_index.py first.",
              file=sys.stderr)
        sys.exit(1)

    print("Loading model, index, and BM25...", file=sys.stderr)
    model, index, chunks, bm25, expander = load_retrieval_components(
        Path(args.index_dir), args.embedding_model, Path(args.expansion_config)
    )
    print(f"  {index.ntotal} vectors, {len(chunks)} chunks ready.", file=sys.stderr)

    # --- Batch prompt-only mode (--eval_seed) ---
    if args.build_prompt_only and args.eval_seed:
        eval_items = json.loads(Path(args.eval_seed).read_text())
        records = []
        low_conf_n = 0
        for item in eval_items:
            q = item["query"]
            print(f"  Retrieving: {q[:60]}...", file=sys.stderr)
            retrieved = run_retrieval(q, model, index, chunks, bm25, expander, args.top_k, args.alpha)
            rec = build_prompt_record(
                item_id=item["id"],
                category=item.get("category", ""),
                query=q,
                chunks=retrieved,
                constraint_config=constraint_config,
                safety_expectation=item.get("safety_expectation", ""),
                forbidden_claims=item.get("forbidden_claims", []),
            )
            if rec["low_confidence"]:
                low_conf_n += 1
            records.append(rec)

        out_path = Path(args.output_file) if args.output_file else Path("outputs/rag_eval/grounded_prompts/batch.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(records)} grounded prompts → {out_path}", file=sys.stderr)
        print(f"  Low-confidence (top_score < {constraint_config['min_top_score']}): "
              f"{low_conf_n}/{len(records)}", file=sys.stderr)
        return

    # --- Single query mode ---
    if not args.query:
        print("ERROR: --query is required (or use --eval_seed for batch mode).", file=sys.stderr)
        sys.exit(1)

    retrieved = run_retrieval(
        args.query, model, index, chunks, bm25, expander, args.top_k, args.alpha
    )

    if not retrieved:
        print("No chunks retrieved.")
        return

    top_score = retrieved[0].get("score_hybrid", 0.0)
    min_top = float(constraint_config.get("min_top_score", 0.35))
    low_confidence = top_score < min_top

    print(format_sources_header(retrieved))
    print()

    grounded_prompt = build_grounded_prompt(args.query, retrieved)

    # --- Build prompt only (single query) ---
    if args.build_prompt_only:
        record = build_prompt_record(
            "single", "", args.query, retrieved, constraint_config,
        )
        if args.output_file:
            out_path = Path(args.output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
            print(f"Saved grounded prompt → {args.output_file}")
        else:
            print(json.dumps(record, indent=2, ensure_ascii=False))
        return

    # --- Print prompt preview (default) ---
    if not args.generate:
        print("--- Grounded Prompt Preview ---")
        print(f"[System]\n{GROUNDED_SYSTEM_MESSAGE}\n")
        print(f"[User]\n{grounded_prompt}")
        if low_confidence:
            print(f"\n[Warning] Top hybrid score {top_score:.4f} < min_top_score "
                  f"{min_top} — fallback would trigger.")
        return

    # --- Generate with constraints ---
    if low_confidence and constraint_config.get("fallback_on_low_confidence", True):
        print(f"[Low confidence: hybrid score={top_score:.4f} < {min_top}]")
        print(constraint_config.get("fallback_message", SAFE_ESCALATION).strip())
        return

    tokenizer, gen_model = load_generation_model(args.model_name_or_path, args.adapter_path)
    if args.disable_constraints:
        answer = generate_one(grounded_prompt, GROUNDED_SYSTEM_MESSAGE, tokenizer, gen_model,
                              args.max_new_tokens, args.temperature)
        print("--- Answer (constraints disabled) ---")
        print(answer)
        return

    outcome = generate_with_constraints(
        query=args.query, chunks=retrieved, category="",
        safety_expectation="", forbidden_claims=[],
        low_confidence=low_confidence, constraint_config=constraint_config,
        tokenizer=tokenizer, model=gen_model,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
    )
    print(f"--- Answer (attempts={outcome['attempts']}, "
          f"fallback={outcome['fallback_triggered']}) ---")
    print(outcome["answer"])
    failed = [k for k, v in outcome["validation"].items() if not v["passed"]]
    if failed:
        print(f"\n[Validation failed: {failed}]")


if __name__ == "__main__":
    main()
