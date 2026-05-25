"""Retrieve top-k chunks from the FAISS vector index for a user query.

Usage:
    python scripts/rag/retrieve.py \
        --index_dir data/rag/vector_store \
        --query "What is the deadline to waive UC SHIP?" \
        --top_k 3 \
        --embedding_model sentence-transformers/all-MiniLM-L6-v2

Output:
    Prints top-k chunks with scores, source titles, and URLs.

TODO:
    - Load FAISS index from index_dir/index.faiss
    - Load chunk metadata from index_dir/chunks.jsonl
    - Embed the query using the same model used during indexing
    - Run index.search(query_vec, top_k)
    - Return chunks with distance scores and source metadata
    - Optionally filter by category
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve top-k RAG chunks for a query.")
    parser.add_argument("--index_dir", default="data/rag/vector_store")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--category_filter", default="",
                        help="Optional: restrict retrieval to a specific category.")
    return parser.parse_args()


def load_chunks_metadata(index_dir: Path) -> list[dict]:
    path = index_dir / "chunks.jsonl"
    if not path.exists():
        return []
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main() -> None:
    args = parse_args()
    index_dir = Path(args.index_dir)

    index_path = index_dir / "index.faiss"
    if not index_path.exists():
        print(f"ERROR: FAISS index not found at {index_path}. Run build_source_index.py first.",
              file=sys.stderr)
        sys.exit(1)

    # TODO: implement FAISS load + embed query + search
    raise NotImplementedError(
        "retrieve not yet implemented. "
        "Install sentence-transformers and faiss-cpu, then implement load_index + embed + search."
    )


if __name__ == "__main__":
    main()
