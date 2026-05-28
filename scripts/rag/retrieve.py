"""Retrieve top-k chunks from the FAISS vector index for a user query.

Usage:
    python scripts/rag/retrieve.py \
        --query "What is the deadline to waive UC SHIP?" \
        --top_k 3

    python scripts/rag/retrieve.py \
        --query "How do I apply for CPT?" \
        --top_k 5 \
        --use_query_expansion \
        --category_filter international_students
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from query_expansion import QueryExpander

INDEX_DIR = Path("data/rag/vector_store")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EXPANSION_CONFIG = Path("configs/rag_query_expansion.json")


def load_index_and_chunks(index_dir: Path) -> tuple:
    index_path = index_dir / "index.faiss"
    meta_path = index_dir / "chunk_metadata.jsonl"

    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}. Run build_index.py first.")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}. Run build_index.py first.")

    index = faiss.read_index(str(index_path))
    chunks = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return index, chunks


def retrieve(
    query: str,
    top_k: int = 3,
    category_filter: str = "",
    dedup_source: bool = True,
    use_query_expansion: bool = False,
    index_dir: Path = INDEX_DIR,
    model_name: str = EMBEDDING_MODEL,
    expansion_config: Path = EXPANSION_CONFIG,
) -> list[dict]:
    index, chunks = load_index_and_chunks(index_dir)
    model = SentenceTransformer(model_name)

    original_query = query
    expanded_query = query
    expansion_info: dict = {}

    if use_query_expansion:
        expander = QueryExpander(config_path=expansion_config)
        result = expander.expand(query)
        expanded_query = result.expanded_query
        expansion_info = {
            "original_query": result.original_query,
            "expanded_query": result.expanded_query,
            "matched_keys": result.matched_keys,
            "appended_terms": result.appended_terms,
        }

    query_vec = model.encode([expanded_query], normalize_embeddings=True).astype(np.float32)

    search_k = top_k * 10 if (dedup_source or category_filter) else top_k
    search_k = min(search_k, index.ntotal)

    scores, indices = index.search(query_vec, search_k)

    results = []
    seen_sources: set[str] = set()
    rank = 1
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunks[idx].copy()
        if category_filter and chunk.get("category") != category_filter:
            continue
        if dedup_source:
            sid = chunk["source_id"]
            if sid in seen_sources:
                continue
            seen_sources.add(sid)
        chunk["rank"] = rank
        chunk["score"] = float(score)
        if expansion_info:
            chunk["_expansion"] = expansion_info
        results.append(chunk)
        rank += 1
        if len(results) >= top_k:
            break

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve top-k RAG chunks for a query.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--index_dir", default=str(INDEX_DIR))
    parser.add_argument("--embedding_model", default=EMBEDDING_MODEL)
    parser.add_argument("--category_filter", default="")
    parser.add_argument("--use_query_expansion", action="store_true")
    parser.add_argument("--expansion_config", default=str(EXPANSION_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = retrieve(
        query=args.query,
        top_k=args.top_k,
        category_filter=args.category_filter,
        use_query_expansion=args.use_query_expansion,
        index_dir=Path(args.index_dir),
        model_name=args.embedding_model,
        expansion_config=Path(args.expansion_config),
    )

    if results and "_expansion" in results[0]:
        exp = results[0]["_expansion"]
        print(f"Original:  {exp['original_query']}")
        print(f"Expanded:  {exp['expanded_query']}")
        print(f"Matched:   {exp['matched_keys']}\n")
    else:
        print(f"Query: {args.query}\n")

    for chunk in results:
        section = f" § {chunk['section_title']}" if chunk.get("section_title") else ""
        print(f"[{chunk['rank']}] score={chunk['score']:.4f} | {chunk['source_id']}{section}")
        print(f"    title: {chunk.get('title', '')}")
        print(f"    url:   {chunk.get('url', '')}")
        print(f"    {chunk['text'][:200]}...")
        print()


if __name__ == "__main__":
    main()
