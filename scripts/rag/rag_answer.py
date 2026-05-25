"""Generate a grounded RAG answer using retrieved chunks and the DPO checkpoint.

Usage:
    python scripts/rag/rag_answer.py \
        --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
        --sft_adapter_path outputs/final_sft_7b \
        --dpo_adapter_path outputs/dpo_7b \
        --index_dir data/rag/vector_store \
        --query "What is the deadline to waive UC SHIP?" \
        --top_k 3 \
        --max_new_tokens 400

Pipeline:
    1. Retrieve top-k chunks from vector index
    2. Build grounded prompt: system + retrieved context block + user question
    3. Generate answer using DPO checkpoint
    4. Print answer with source citations

TODO:
    - Import retrieve logic from retrieve.py
    - Load base model + SFT adapter + DPO adapter (stacked PEFT)
    - Build grounded system prompt that includes retrieved context
    - Add source title/URL citation at end of each answer
    - Handle low-confidence retrieval: if max score below threshold, trigger safe escalation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


GROUNDED_SYSTEM_MESSAGE = (
    "You are an assistant helping international students navigate campus administrative tasks. "
    "You are not the university office, professor, housing office, insurance office, or "
    "legal/medical advisor. Do not pretend to be an official office.\n\n"
    "The following context was retrieved from official UCSD sources. "
    "Base your answer on this context. If the context does not address the question, "
    "say you cannot verify the specific detail and direct the student to the relevant official office. "
    "Always mention the source title or URL when you cite a fact."
)


def build_grounded_prompt(query: str, chunks: list[dict]) -> str:
    """Build a grounded user prompt with retrieved context blocks."""
    context_lines = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("source_title", "Official UCSD Source")
        url = chunk.get("url", "")
        source_ref = f"{title} ({url})" if url else title
        context_lines.append(f"[Source {i}: {source_ref}]\n{chunk['text']}")

    context_block = "\n\n".join(context_lines)
    return f"Retrieved context:\n{context_block}\n\nStudent question:\n{query}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a grounded RAG answer.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sft_adapter_path", default="outputs/final_sft_7b")
    parser.add_argument("--dpo_adapter_path", default="outputs/dpo_7b")
    parser.add_argument("--index_dir", default="data/rag/vector_store")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--retrieval_score_threshold", type=float, default=0.5,
                        help="Below this score, trigger safe escalation instead of grounded answer.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    index_path = Path(args.index_dir) / "index.faiss"
    if not index_path.exists():
        print(f"ERROR: FAISS index not found at {index_path}. Run build_source_index.py first.",
              file=sys.stderr)
        sys.exit(1)

    # TODO: call retrieve.py logic to get top_k chunks
    # TODO: load base model + SFT adapter + DPO adapter
    # TODO: build grounded prompt using build_grounded_prompt()
    # TODO: generate response
    # TODO: print response + source citations

    raise NotImplementedError(
        "rag_answer not yet implemented. "
        "Implement after retrieve.py and build_source_index.py are functional."
    )


if __name__ == "__main__":
    main()
