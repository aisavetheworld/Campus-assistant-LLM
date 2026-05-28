"""Chunk raw source text files into overlapping segments with heading detection.

Usage:
    python scripts/rag/chunk_sources.py \
        --sources_meta data/rag/ucsd_sources.json \
        --raw_dir data/rag/raw_sources \
        --output data/rag/processed_chunks/chunks.jsonl \
        --chunk_size 512 \
        --overlap 50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


HEADING_MAX_LEN = 120
HEADING_SKIP_PREFIXES = ("http", "www", "•", "-", "*")
HEADING_SKIP_ENDS = (".", ",", ";", ":", "?")


def is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > HEADING_MAX_LEN:
        return False
    if any(line.startswith(p) for p in HEADING_SKIP_PREFIXES):
        return False
    if line.endswith(HEADING_SKIP_ENDS):
        return False
    word_count = len(line.split())
    if word_count < 2 or word_count > 15:
        return False
    # Heading patterns: "Step N:", "What is X", "X Requirements", title-case short phrases
    if re.match(r"^(Step\s+\d+|What\s+is|How\s+to|When\s+|Why\s+|If\s+you)", line, re.I):
        return True
    # Title-case: most words capitalized
    words = line.split()
    cap_words = sum(1 for w in words if w and w[0].isupper())
    return cap_words / len(words) >= 0.6


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Return list of (section_title, section_text) pairs."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    sections: list[tuple[str, str]] = []
    current_title = ""
    current_lines: list[str] = []

    for para in paragraphs:
        lines = para.split("\n")
        if len(lines) == 1 and is_heading(para):
            if current_lines:
                sections.append((current_title, " ".join(current_lines)))
                current_lines = []
            current_title = para.strip()
        else:
            current_lines.append(para)

    if current_lines:
        sections.append((current_title, " ".join(current_lines)))

    if not sections:
        sections = [("", text)]

    return sections


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def build_chunks(source: dict, raw_dir: Path, chunk_size: int, overlap: int) -> list[dict]:
    sid = source["source_id"]
    txt_path = raw_dir / f"{sid}.txt"

    if not txt_path.exists():
        return []
    text = txt_path.read_text(encoding="utf-8").strip()
    if len(text.split()) < 20:
        return []

    sections = split_into_sections(text)
    chunks = []
    chunk_idx = 0

    for section_title, section_text in sections:
        for chunk_text_str in chunk_text(section_text, chunk_size, overlap):
            wc = len(chunk_text_str.split())
            if wc < 10:
                continue
            chunks.append({
                "chunk_id": f"{sid}_chunk_{chunk_idx:03d}",
                "source_id": sid,
                "title": source.get("title", ""),
                "section_title": section_title,
                "category": source.get("category", ""),
                "url": source.get("url", ""),
                "text": chunk_text_str,
                "word_count": wc,
            })
            chunk_idx += 1

    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources_meta", default="data/rag/ucsd_sources.json")
    parser.add_argument("--raw_dir", default="data/rag/raw_sources")
    parser.add_argument("--output", default="data/rag/processed_chunks/chunks.jsonl")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sources = json.loads(Path(args.sources_meta).read_text())

    all_chunks = []
    skipped = 0
    for source in sources:
        chunks = build_chunks(source, raw_dir, args.chunk_size, args.overlap)
        if not chunks:
            print(f"  SKIP {source['source_id']}: no content")
            skipped += 1
        else:
            print(f"  {source['source_id']}: {len(chunks)} chunks")
            all_chunks.extend(chunks)

    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(sources) - skipped} sources, {len(all_chunks)} total chunks → {out_path}")


if __name__ == "__main__":
    main()
