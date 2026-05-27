"""Build a FAISS vector index from processed RAG chunks.

Usage:
    python scripts/rag/build_source_index.py \
        --chunks_dir data/rag/processed_chunks \
        --index_dir data/rag/vector_store \
        --embedding_model sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


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
        with open(path, encoding="utf-8") as f:
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

    print("Loading chunks...")
    chunks = load_chunks(chunks_dir)
    if not chunks:
        print("No chunks found. Run chunk_sources.py first.")
        sys.exit(0)
    print(f"  {len(chunks)} chunks loaded")

    print(f"Loading embedding model: {args.embedding_model}")
    model = SentenceTransformer(args.embedding_model)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks (batch_size={args.batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  Embedding shape: {embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine similarity (embeddings are normalized)
    index.add(embeddings.astype(np.float32))
    print(f"  FAISS index built: {index.ntotal} vectors, dim={dim}")

    index_path = index_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    print(f"  Saved index → {index_path}")

    chunks_out = index_dir / "chunk_metadata.jsonl"
    with open(chunks_out, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"  Saved chunk metadata → {chunks_out}")

    print(f"\nDone. {index.ntotal} vectors indexed.")


if __name__ == "__main__":
    main()
