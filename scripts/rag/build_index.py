"""Build a FAISS vector index from the unified chunks.jsonl.

Usage:
    python scripts/rag/build_index.py \
        --chunks data/rag/processed_chunks/chunks.jsonl \
        --index_dir data/rag/vector_store \
        --embedding_model sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/rag/processed_chunks/chunks.jsonl")
    parser.add_argument("--index_dir", default="data/rag/vector_store")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--report_dir", default="outputs/rag_eval")
    return parser.parse_args()


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def write_report(report: dict, out_dir: Path) -> None:
    lines = [
        "# Index Build Report",
        "",
        "## Configuration",
        "",
        f"| Setting | Value |",
        f"|---|---|",
        f"| Embedding model | {report['embedding_model']} |",
        f"| Embedding dim | {report['embedding_dim']} |",
        f"| Index type | {report['index_type']} |",
        f"| Total chunks | {report['total_chunks']} |",
        f"| Build time | {report['build_time_s']:.1f}s |",
        "",
        "## Source Coverage",
        "",
        "| source_id | chunks |",
        "|---|---|",
    ]
    for sid, count in sorted(report["chunks_per_source"].items()):
        lines.append(f"| {sid} | {count} |")

    lines += [
        "",
        "## Category Coverage",
        "",
        "| category | chunks |",
        "|---|---|",
    ]
    for cat, count in sorted(report["chunks_per_category"].items()):
        lines.append(f"| {cat} | {count} |")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index_build_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    chunks_path = Path(args.chunks)
    index_dir = Path(args.index_dir)

    if not chunks_path.exists():
        print(f"ERROR: {chunks_path} not found. Run chunk_sources.py first.", file=sys.stderr)
        sys.exit(1)

    print("Loading chunks...")
    chunks = load_chunks(chunks_path)
    print(f"  {len(chunks)} chunks loaded")

    print(f"Loading embedding model: {args.embedding_model}")
    model = SentenceTransformer(args.embedding_model)

    texts = [c["text"] for c in chunks]
    t0 = time.time()
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — shape: {embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    print(f"  FAISS IndexFlatIP: {index.ntotal} vectors, dim={dim}")

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    print(f"  Saved → {index_dir}/index.faiss")

    meta_path = index_dir / "chunk_metadata.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"  Saved → {meta_path}")

    # Build report
    from collections import Counter
    report = {
        "embedding_model": args.embedding_model,
        "embedding_dim": dim,
        "index_type": "IndexFlatIP (cosine, normalized)",
        "total_chunks": len(chunks),
        "build_time_s": elapsed,
        "chunks_per_source": dict(Counter(c["source_id"] for c in chunks)),
        "chunks_per_category": dict(Counter(c["category"] for c in chunks)),
    }
    write_report(report, Path(args.report_dir))
    print(f"  Report → {args.report_dir}/index_build_report.md")
    print(f"\nDone. {index.ntotal} vectors indexed.")


if __name__ == "__main__":
    main()
