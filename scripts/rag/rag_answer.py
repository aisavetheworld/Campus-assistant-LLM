"""Grounded RAG answer pipeline using hybrid retrieval (alpha=0.8, QE enabled).

Modes:
  Default             -- retrieve + print grounded prompt preview (no GPU)
  --build_prompt_only -- retrieve + save prompt JSON for batch GPU generation
  --generate          -- retrieve + generate answer with DPO checkpoint (requires GPU)
  --batch_generate    -- load pre-built prompts and generate all answers (Colab/GPU)

Usage (preview single query):
    python scripts/rag/rag_answer.py \
        --query "What is the deadline to waive UC SHIP?"

Usage (save prompt to file):
    python scripts/rag/rag_answer.py \
        --query "What is the deadline to waive UC SHIP?" \
        --build_prompt_only --output_file outputs/rag_eval/grounded_prompts/single.json

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

INDEX_DIR = Path("data/rag/vector_store")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EXPANSION_CONFIG = Path("configs/rag_query_expansion.json")
DEFAULT_ALPHA = 0.8
DEFAULT_TOP_K = 5

GROUNDED_SYSTEM_MESSAGE = (
    "You are an assistant helping students navigate campus administrative tasks at UC San Diego. "
    "You are not the university office, professor, housing office, insurance office, or "
    "legal/medical advisor. Do not pretend to be an official office.\n\n"
    "The following context was retrieved from official UCSD sources. "
    "Base your answer strictly on this context. Do not invent deadlines, fees, policies, "
    "or guarantees that are not stated in the context. "
    "If the context does not address the question, say you cannot verify the specific detail "
    "from the provided sources and direct the student to the relevant official office. "
    "Always cite the source title or URL when stating a fact. "
    "End your answer with a brief 'Sources:' section listing the sources used."
)

SAFE_ESCALATION = (
    "I wasn't able to find reliable information in my sources to answer this question confidently. "
    "Please contact the relevant UCSD office directly for accurate guidance."
)


def build_grounded_prompt(query: str, chunks: list[dict]) -> str:
    context_lines = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("title", "Official UCSD Source")
        url = chunk.get("url", "")
        section = chunk.get("section_title", "")
        source_ref = title
        if section:
            source_ref += f" — {section}"
        if url:
            source_ref += f" ({url})"
        context_lines.append(f"[Source {i}: {source_ref}]\n{chunk['text']}")

    context_block = "\n\n".join(context_lines)
    return (
        f"Retrieved context:\n{context_block}\n\n"
        f"Student question:\n{query}\n\n"
        "Answer based only on the retrieved context above. "
        "Cite source titles or URLs. "
        "End with a 'Sources:' section."
    )


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

    print(f"Loading model: {model_name_or_path}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path and Path(adapter_path).exists():
        print(f"Loading DPO adapter: {adapter_path}", file=sys.stderr)
        model = PeftModel.from_pretrained(model, adapter_path)
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


def build_prompt_record(item_id: str, category: str, query: str, chunks: list[dict]) -> dict:
    return {
        "id": item_id,
        "category": category,
        "query": query,
        "system_message": GROUNDED_SYSTEM_MESSAGE,
        "user_prompt": build_grounded_prompt(query, chunks),
        "retrieved_chunks": chunks,
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
    parser.add_argument("--retrieval_score_threshold", type=float, default=0.35,
                        help="If top-1 hybrid score below this, use safe escalation.")
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
        for i, rec in enumerate(records, 1):
            print(f"[{i}/{len(records)}] {rec['id']}: {rec['query'][:60]}...", file=sys.stderr)
            answer = generate_one(
                prompt=rec["grounded_prompt"],
                system_message=GROUNDED_SYSTEM_MESSAGE,
                tokenizer=tokenizer,
                model=gen_model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            results.append({
                "id": rec["id"],
                "category": rec.get("category", ""),
                "query": rec["query"],
                "generated_answer": answer,
            })

        out_path = Path(args.output_file) if args.output_file else Path("outputs/rag_eval/generated_answers.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(results)} answers → {out_path}", file=sys.stderr)
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
        for item in eval_items:
            q = item["query"]
            print(f"  Retrieving: {q[:60]}...", file=sys.stderr)
            retrieved = run_retrieval(q, model, index, chunks, bm25, expander, args.top_k, args.alpha)
            records.append(build_prompt_record(
                item_id=item["id"],
                category=item.get("category", ""),
                query=q,
                chunks=retrieved,
            ))

        out_path = Path(args.output_file) if args.output_file else Path("outputs/rag_eval/grounded_prompts/batch.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(records)} grounded prompts → {out_path}", file=sys.stderr)
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
    low_confidence = top_score < args.retrieval_score_threshold

    print(format_sources_header(retrieved))
    print()

    grounded_prompt = build_grounded_prompt(args.query, retrieved)

    # --- Build prompt only (single query) ---
    if args.build_prompt_only:
        record = build_prompt_record("single", "", args.query, retrieved)
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
            print(f"\n[Warning] Top hybrid score {top_score:.4f} < threshold "
                  f"{args.retrieval_score_threshold} — safe escalation would trigger.")
        return

    # --- Generate ---
    if low_confidence:
        print(f"[Low confidence: hybrid score={top_score:.4f}]")
        print(SAFE_ESCALATION)
        return

    answer = generate_answer(
        prompt=grounded_prompt,
        system_message=GROUNDED_SYSTEM_MESSAGE,
        model_name_or_path=args.model_name_or_path,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print("--- Answer ---")
    print(answer)


if __name__ == "__main__":
    main()
