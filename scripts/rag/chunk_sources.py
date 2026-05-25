"""Chunk raw source text files into overlapping fixed-size chunks.

Usage:
    python scripts/rag/chunk_sources.py \
        --sources_meta data/rag/ucsd_sources.json \
        --raw_dir data/rag/raw_sources \
        --output_dir data/rag/processed_chunks \
        --chunk_size 512 \
        --overlap 50

Output format (one JSONL per source):
    {"source_id": "...", "chunk_id": "..._chunk_000", "text": "...", "source_title": "...", "url": "..."}

TODO:
    - Load source metadata from ucsd_sources.json
    - For each source_id, find the corresponding .txt file in raw_dir
    - Tokenize or split by words/characters
    - Apply sliding window chunking: chunk_size tokens, overlap tokens
    - Write chunks to output_dir/<source_id>.jsonl
    - Print summary: sources processed, total chunks created, skipped sources (no .txt file)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk raw source text into overlapping segments.")
    parser.add_argument("--sources_meta", default="data/rag/ucsd_sources.json")
    parser.add_argument("--raw_dir", default="data/rag/raw_sources")
    parser.add_argument("--output_dir", default="data/rag/processed_chunks")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Approximate chunk size in words (not tokens).")
    parser.add_argument("--overlap", type=int, default=50,
                        help="Overlap between consecutive chunks in words.")
    return parser.parse_args()


def load_sources_meta(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def main() -> None:
    args = parse_args()
    sources_meta_path = Path(args.sources_meta)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    if not sources_meta_path.exists():
        print(f"ERROR: sources metadata not found: {sources_meta_path}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    sources = load_sources_meta(sources_meta_path)

    processed = 0
    skipped = 0
    total_chunks = 0

    for source in sources:
        source_id = source["source_id"]
        raw_file = raw_dir / f"{source_id}.txt"

        if not raw_file.exists():
            print(f"  SKIP {source_id}: no .txt file in {raw_dir}")
            skipped += 1
            continue

        text = raw_file.read_text(encoding="utf-8").strip()
        if not text:
            print(f"  SKIP {source_id}: empty .txt file")
            skipped += 1
            continue

        chunks = chunk_text(text, args.chunk_size, args.overlap)
        out_path = output_dir / f"{source_id}.jsonl"

        with open(out_path, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                record = {
                    "source_id": source_id,
                    "chunk_id": f"{source_id}_chunk_{i:03d}",
                    "text": chunk,
                    "source_title": source.get("title", ""),
                    "url": source.get("url", ""),
                    "category": source.get("category", ""),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"  {source_id}: {len(chunks)} chunks")
        total_chunks += len(chunks)
        processed += 1

    print(f"\nDone. {processed} sources processed, {skipped} skipped, {total_chunks} total chunks.")


if __name__ == "__main__":
    main()
