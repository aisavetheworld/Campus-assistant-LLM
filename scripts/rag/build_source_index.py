"""Build a FAISS vector index from processed RAG chunks.

Usage:
    python scripts/rag/build_source_index.py \
        --chunks_dir data/rag/processed_chunks \
        --index_dir data/rag/vector_store \
        --embedding_model sentence-transformers/all-MiniLM-L6-v2

TODO:
    - Load all JSONL files from chunks_dir
    - Embed each chunk using the specified model
    - Build a FAISS flat index (IndexFlatL2 or IndexFlatIP)
    - Save index to index_dir/index.faiss
    - Save chunk metadata (source_id, chunk_id, text, source_title, url) to index_dir/chunks.jsonl
    - Print summary: total chunks indexed, embedding model, index size
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS vector index from RAG chunks.")
    parser.add_argument("--chunks_dir", default="data/rag/processed_chunks")
    parser.add_argument("--index_dir", default="data/rag/vector_store")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def load_chunks(chunks_dir: Path) -> list[dict]:
    chunks = []
    for path in sorted(chunks_dir.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    return chunks


def main() -> None:
    args = parse_args()
    chunks_dir = Path(args.chunks_dir)
    index_dir = Path(args.index_dir)

    if not chunks_dir.exists():
        print(f"ERROR: chunks_dir not found: {chunks_dir}", file=sys.stderr)
        sys.exit(1)

    index_dir.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks(chunks_dir)

    if not chunks:
        print("No chunks found. Run chunk_sources.py first.")
        sys.exit(0)

    print(f"Loaded {len(chunks)} chunks from {chunks_dir}")

    # TODO: implement embedding + FAISS index build
    raise NotImplementedError(
        "build_source_index not yet implemented. "
        "Install sentence-transformers and faiss-cpu, then implement embedding loop and index.add()."
    )


if __name__ == "__main__":
    main()
