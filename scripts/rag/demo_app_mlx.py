"""Gradio demo using MLX backend (Apple Silicon, M1+ macs).

MLX is Apple's native ML framework — 2-3× faster than PyTorch MPS on
M-series chips and lets you use 4-bit quantized weights (~4.5 GB for
Qwen2.5-7B-Instruct-4bit).

Run on a Mac:
    pip install mlx-lm gradio faiss-cpu sentence-transformers rank-bm25 pyyaml
    python scripts/rag/demo_app_mlx.py

Then open http://localhost:7860.

Pipeline: identical to demo_app.py (hybrid retrieval → constrained
generation with retry + fallback). Only the generate function differs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))
from rag_answer import (
    GROUNDED_SYSTEM_MESSAGE,
    SAFE_ESCALATION,
    INDEX_DIR,
    EMBEDDING_MODEL,
    EXPANSION_CONFIG,
    CONSTRAINT_CONFIG,
    DEFAULT_ALPHA,
    DEFAULT_TOP_K,
    load_retrieval_components,
    run_retrieval,
    build_grounded_prompt,
)
from answer_validators import (
    load_config as load_constraint_config,
    run_all_checks,
    collect_fix_hints,
    all_passed,
)


# Globals filled at startup
RETRIEVER = None
MLX_MODEL = None
MLX_TOKENIZER = None
CONSTRAINT_CONF = None


DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"


EXAMPLE_QUERIES = [
    "Am I eligible for CPT if I just started my first semester at UCSD?",
    "How much does it cost to live in the on-campus dorms at UCSD?",
    "I want to drop below full-time enrollment this quarter due to a medical issue. Will this affect my F-1 status?",
    "How do I submit a maintenance request for a broken heater in my dorm room?",
    "What is the deadline to waive UC SHIP and what insurance do I need to qualify?",
    "Can I bring my pet lizard to live in the dorms with me?",
]


def mlx_generate_one(
    user_prompt: str,
    system_message: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """Single generation using MLX. Mirrors generate_one() from rag_answer.py."""
    from mlx_lm import generate
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]
    prompt = MLX_TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    try:
        # mlx-lm >=0.20 signature
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=temperature)
        return generate(
            MLX_MODEL, MLX_TOKENIZER, prompt=prompt,
            max_tokens=max_tokens, sampler=sampler, verbose=False,
        )
    except Exception:
        # Older mlx-lm signature
        return generate(
            MLX_MODEL, MLX_TOKENIZER, prompt=prompt,
            max_tokens=max_tokens, temp=temperature, verbose=False,
        )


def mlx_generate_with_constraints(
    query: str,
    chunks: list[dict],
    category: str,
    safety_expectation: str,
    forbidden_claims: list[str],
    low_confidence: bool,
    constraint_config: dict,
    max_new_tokens: int,
    temperature: float,
) -> dict:
    """Mirror of generate_with_constraints() but uses MLX for generation."""
    fallback_message = constraint_config.get("fallback_message", SAFE_ESCALATION).strip()
    max_retries = int(constraint_config.get("max_retries", 1))

    if low_confidence and constraint_config.get("fallback_on_low_confidence", True):
        validation = run_all_checks(
            answer=fallback_message, chunks=chunks, query=query, category=category,
            forbidden_claims=forbidden_claims, safety_expectation=safety_expectation,
            low_confidence=True, config=constraint_config,
        )
        return {
            "answer": fallback_message, "validation": validation, "attempts": 0,
            "fallback_triggered": True, "fallback_reason": "low_retrieval_confidence",
        }

    prompt = build_grounded_prompt(query, chunks)
    answer = mlx_generate_one(prompt, GROUNDED_SYSTEM_MESSAGE, max_new_tokens, temperature)
    validation = run_all_checks(
        answer=answer, chunks=chunks, query=query, category=category,
        forbidden_claims=forbidden_claims, safety_expectation=safety_expectation,
        low_confidence=False, config=constraint_config,
    )
    attempts = 1
    if all_passed(validation):
        return {"answer": answer, "validation": validation, "attempts": attempts,
                "fallback_triggered": False, "fallback_reason": ""}

    for _ in range(max_retries):
        hints = collect_fix_hints(validation)
        if not hints:
            break
        retry_prompt = build_grounded_prompt(
            query, chunks,
            extra_constraints="\n".join(f"- {h}" for h in hints),
        )
        answer = mlx_generate_one(retry_prompt, GROUNDED_SYSTEM_MESSAGE,
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

    failed_checks = [k for k, v in validation.items() if not v["passed"]]
    fallback_validation = run_all_checks(
        answer=fallback_message, chunks=chunks, query=query, category=category,
        forbidden_claims=forbidden_claims, safety_expectation=safety_expectation,
        low_confidence=True, config=constraint_config,
    )
    return {
        "answer": fallback_message, "validation": fallback_validation,
        "attempts": attempts, "fallback_triggered": True,
        "fallback_reason": f"validation_failed_after_retry: {failed_checks}",
    }


# ---- UI helpers (copied from demo_app.py) ----

def format_retrieved_chunks_md(chunks: list[dict]) -> str:
    if not chunks:
        return "*No chunks retrieved.*"
    lines = []
    for i, c in enumerate(chunks, 1):
        title = c.get("title", "Untitled")
        url = c.get("url", "")
        section = c.get("section_title", "")
        score = c.get("score_hybrid", 0.0)
        text = (c.get("text", "") or "")[:300].replace("\n", " ")
        head = f"**[{i}] `{c.get('source_id','')}` — {title}**"
        if section:
            head += f" · _{section}_"
        head += f"  ·  hybrid = `{score:.3f}`"
        if url:
            head += f"  ·  [link]({url})"
        lines.append(head)
        lines.append(f"> {text}{'…' if len(c.get('text','')) > 300 else ''}")
        lines.append("")
    return "\n".join(lines)


def format_validation_md(validation: dict) -> str:
    if not validation:
        return "*No validation run.*"
    lines = ["| Check | Passed | Detail |", "|---|---|---|"]
    for name, r in validation.items():
        icon = "✓" if r["passed"] else "✗"
        detail = r.get("detail", "")[:80].replace("|", "\\|")
        lines.append(f"| `{name}` | {icon} | {detail} |")
    return "\n".join(lines)


def format_metadata_md(outcome: dict, top_score: float) -> str:
    n_pass = sum(1 for v in outcome["validation"].values() if v["passed"])
    n_total = len(outcome["validation"])
    lines = [
        f"- **Top retrieval score:** `{top_score:.3f}`",
        f"- **Generation attempts:** `{outcome['attempts']}`",
        f"- **Fallback triggered:** `{outcome['fallback_triggered']}`"
        + (f"  ({outcome['fallback_reason']})" if outcome["fallback_triggered"] else ""),
        f"- **Validators passed:** `{n_pass}/{n_total}`",
    ]
    return "\n".join(lines)


def run_pipeline(query: str, category: str, enable_constraints: bool):
    query = (query or "").strip()
    if not query:
        return "Please enter a question.", "", "", "", ""

    model, index, chunks_corpus, bm25, expander = RETRIEVER
    retrieved = run_retrieval(query, model, index, chunks_corpus, bm25, expander,
                              DEFAULT_TOP_K, DEFAULT_ALPHA)
    retrieved_md = format_retrieved_chunks_md(retrieved)

    top_score = retrieved[0].get("score_hybrid", 0.0) if retrieved else 0.0
    min_top = float(CONSTRAINT_CONF.get("min_top_score", 0.35))
    low_confidence = top_score < min_top

    if enable_constraints:
        outcome = mlx_generate_with_constraints(
            query=query, chunks=retrieved, category=category or "",
            safety_expectation="", forbidden_claims=[],
            low_confidence=low_confidence, constraint_config=CONSTRAINT_CONF,
            max_new_tokens=int(CONSTRAINT_CONF.get("max_new_tokens", 512)),
            temperature=float(CONSTRAINT_CONF.get("temperature", 0.2)),
        )
        answer = outcome["answer"]
        validation_md = format_validation_md(outcome["validation"])
        metadata_md = format_metadata_md(outcome, top_score)
    else:
        prompt = build_grounded_prompt(query, retrieved)
        answer = mlx_generate_one(prompt, GROUNDED_SYSTEM_MESSAGE,
                                  max_tokens=512, temperature=0.2)
        validation_md = "*Constraints disabled — no validation run.*"
        metadata_md = f"- **Top retrieval score:** `{top_score:.3f}`\n- Constraints disabled."

    if "Sources:" in answer:
        body, _, sources_section = answer.partition("Sources:")
        sources_md = "**Sources:**\n" + sources_section.strip()
        answer_md = body.strip()
    else:
        sources_md = "*(no Sources section in answer)*"
        answer_md = answer.strip()

    return answer_md, sources_md, retrieved_md, validation_md, metadata_md


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Campus Assistant — MLX Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # Campus Assistant — RAG Demo (MLX backend)
        Grounded answers about UCSD student admin tasks.
        Running locally on Apple Silicon with MLX-quantized Qwen2.5-7B.
        """)

        with gr.Row():
            with gr.Column(scale=3):
                query = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. Am I eligible for CPT if I just started my first semester?",
                    lines=2,
                )
                with gr.Row():
                    category = gr.Dropdown(
                        label="Category (optional, drives safety check)",
                        choices=["", "international_students", "course_enrollment",
                                 "housing", "health_insurance", "student_health",
                                 "financial_aid", "graduate_students"],
                        value="",
                    )
                    constraints_toggle = gr.Checkbox(
                        label="Enable constraints (validators + retry + fallback)",
                        value=True,
                    )
                submit = gr.Button("Get answer", variant="primary")
                gr.Examples(
                    examples=[[q, ""] for q in EXAMPLE_QUERIES],
                    inputs=[query, category],
                    label="Try these",
                )
            with gr.Column(scale=4):
                answer_md = gr.Markdown(label="Answer", value="*(answer will appear here)*")
                sources_md = gr.Markdown(label="Sources", value="")
                with gr.Accordion("Pipeline metadata", open=False):
                    metadata_md = gr.Markdown()
                with gr.Accordion("Validator results (11 checks)", open=False):
                    validation_md = gr.Markdown()
                with gr.Accordion("Retrieved chunks (top-5)", open=False):
                    retrieved_md = gr.Markdown()

        submit.click(
            fn=run_pipeline,
            inputs=[query, category, constraints_toggle],
            outputs=[answer_md, sources_md, retrieved_md, validation_md, metadata_md],
        )
        query.submit(
            fn=run_pipeline,
            inputs=[query, category, constraints_toggle],
            outputs=[answer_md, sources_md, retrieved_md, validation_md, metadata_md],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MLX_MODEL,
                        help=f"MLX model id (default: {DEFAULT_MLX_MODEL})")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    global RETRIEVER, MLX_MODEL, MLX_TOKENIZER, CONSTRAINT_CONF

    print("Loading retrieval components...", file=sys.stderr)
    RETRIEVER = load_retrieval_components(
        Path(INDEX_DIR), EMBEDDING_MODEL, Path(EXPANSION_CONFIG)
    )
    print(f"  {RETRIEVER[1].ntotal} vectors, {len(RETRIEVER[2])} chunks ready.",
          file=sys.stderr)

    CONSTRAINT_CONF = load_constraint_config(CONSTRAINT_CONFIG)

    print(f"Loading MLX model: {args.model}", file=sys.stderr)
    from mlx_lm import load
    MLX_MODEL, MLX_TOKENIZER = load(args.model)
    print("MLX model ready.", file=sys.stderr)

    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
