"""Gradio demo for the Campus Assistant RAG pipeline.

Run on Colab (GPU):
    !pip install -q gradio faiss-cpu sentence-transformers rank-bm25 pyyaml peft
    !pip install -q -U "torchao>=0.16.0"
    !python scripts/rag/demo_app.py --share

Run locally (CPU, retrieval only — generation disabled):
    pip install gradio faiss-cpu sentence-transformers rank-bm25 pyyaml
    python scripts/rag/demo_app.py --no_generate

Flags:
    --share           Public Gradio link (Colab)
    --port            Server port (default 7860)
    --no_generate     Skip LLM loading; only show retrieval + prompt
    --base_only       Load base Qwen2.5-7B without DPO adapter
    --port_browser    Open browser automatically (local only)
"""

from __future__ import annotations

import argparse
import json
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
    load_generation_model,
    generate_with_constraints,
    generate_one,
    run_retrieval,
    build_grounded_prompt,
)
from answer_validators import load_config as load_constraint_config


# Globals filled at startup
RETRIEVER = None
TOKENIZER = None
GEN_MODEL = None
CONSTRAINT_CONF = None
GENERATE_ENABLED = False


EXAMPLE_QUERIES = [
    # Good case: clear F-1 / CPT question
    "Am I eligible for CPT if I just started my first semester at UCSD?",
    # Was zero-recall — now fixed by query expansion
    "How much does it cost to live in the on-campus dorms at UCSD?",
    # Triggers safety escalation
    "I want to drop below full-time enrollment this quarter due to a medical issue. Will this affect my F-1 status?",
    # How-to format
    "How do I submit a maintenance request for a broken heater in my dorm room?",
    # Will likely trip the no_hallucinated_deadline check if model hedges
    "What is the deadline to waive UC SHIP and what insurance do I need to qualify?",
    # Low-confidence example
    "Can I bring my pet lizard to live in the dorms with me?",
]


def format_retrieved_chunks_md(chunks: list[dict]) -> str:
    """Render retrieved chunks as a markdown bullet list."""
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
        head += f"  ·  hybrid score = `{score:.3f}`"
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


def run_pipeline(query: str, category: str, enable_constraints: bool) -> tuple[str, str, str, str, str]:
    """Returns: (answer, sources_md, retrieved_md, validation_md, metadata_md)"""
    query = (query or "").strip()
    if not query:
        return "Please enter a question.", "", "", "", ""

    model, index, chunks_corpus, bm25, expander = RETRIEVER
    retrieved = run_retrieval(query, model, index, chunks_corpus, bm25, expander,
                              DEFAULT_TOP_K, DEFAULT_ALPHA)
    retrieved_md = format_retrieved_chunks_md(retrieved)

    if not GENERATE_ENABLED:
        prompt = build_grounded_prompt(query, retrieved)
        answer_md = "*Generation disabled (running with `--no_generate`).*\n\n" \
                    "**Grounded prompt preview (first 2000 chars):**\n\n```\n" \
                    + prompt[:2000] + "\n```"
        return answer_md, "", retrieved_md, "", ""

    top_score = retrieved[0].get("score_hybrid", 0.0) if retrieved else 0.0
    min_top = float(CONSTRAINT_CONF.get("min_top_score", 0.35))
    low_confidence = top_score < min_top

    if enable_constraints:
        outcome = generate_with_constraints(
            query=query, chunks=retrieved, category=category or "",
            safety_expectation="", forbidden_claims=[],
            low_confidence=low_confidence, constraint_config=CONSTRAINT_CONF,
            tokenizer=TOKENIZER, model=GEN_MODEL,
            max_new_tokens=int(CONSTRAINT_CONF.get("max_new_tokens", 512)),
            temperature=float(CONSTRAINT_CONF.get("temperature", 0.2)),
        )
        answer = outcome["answer"]
        validation_md = format_validation_md(outcome["validation"])
        metadata_md = format_metadata_md(outcome, top_score)
    else:
        prompt = build_grounded_prompt(query, retrieved)
        answer = generate_one(prompt, GROUNDED_SYSTEM_MESSAGE, TOKENIZER, GEN_MODEL,
                              max_new_tokens=512, temperature=0.2)
        validation_md = "*Constraints disabled — no validation run.*"
        metadata_md = f"- **Top retrieval score:** `{top_score:.3f}`\n- Constraints disabled."

    # Split answer body from Sources section for cleaner display
    if "Sources:" in answer:
        body, _, sources_section = answer.partition("Sources:")
        sources_md = "**Sources:**\n" + sources_section.strip()
        answer_md = body.strip()
    else:
        sources_md = "*(no Sources section in answer)*"
        answer_md = answer.strip()

    return answer_md, sources_md, retrieved_md, validation_md, metadata_md


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Campus Assistant — RAG Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # Campus Assistant — RAG Demo
        Grounded answers about UC San Diego student administrative tasks.
        Each answer is constrained to retrieved official UCSD sources;
        the pipeline runs **hybrid retrieval → strict-prompt generation →
        11 post-hoc validators → retry-with-hint → safe fallback**.
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
                        label="Category (optional, helps safety check)",
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
                with gr.Accordion("Pipeline metadata (attempts, fallback, top score)", open=False):
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
    parser.add_argument("--share", action="store_true", help="Public Gradio link (Colab).")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no_generate", action="store_true",
                        help="Skip LLM; only show retrieval + grounded prompt.")
    parser.add_argument("--base_only", action="store_true",
                        help="Load base Qwen2.5-7B without DPO adapter.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_path", default="outputs/dpo_7b")
    args = parser.parse_args()

    global RETRIEVER, TOKENIZER, GEN_MODEL, CONSTRAINT_CONF, GENERATE_ENABLED

    print("Loading retrieval components...", file=sys.stderr)
    RETRIEVER = load_retrieval_components(
        Path(INDEX_DIR), EMBEDDING_MODEL, Path(EXPANSION_CONFIG)
    )
    print(f"  {RETRIEVER[1].ntotal} vectors, {len(RETRIEVER[2])} chunks ready.",
          file=sys.stderr)

    CONSTRAINT_CONF = load_constraint_config(CONSTRAINT_CONFIG)

    if not args.no_generate:
        adapter = "" if args.base_only else args.adapter_path
        print(f"Loading generation model {args.model_name_or_path}"
              + (f" + adapter {adapter}" if adapter else " (base only)"),
              file=sys.stderr)
        TOKENIZER, GEN_MODEL = load_generation_model(args.model_name_or_path, adapter)
        GENERATE_ENABLED = True
    else:
        print("Skipping model load (--no_generate).", file=sys.stderr)
        GENERATE_ENABLED = False

    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
