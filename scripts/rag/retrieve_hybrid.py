"""Hybrid retrieval combining dense FAISS and BM25 sparse retrieval.

Score normalization:
  - Dense scores (cosine on normalized vectors) are already in [−1, 1]; clipped to [0, 1].
  - BM25 scores are min-max normalized per query to [0, 1].
  - Combined: alpha * dense_norm + (1 − alpha) * bm25_norm

alpha=1.0  → dense only
alpha=0.0  → BM25 only
alpha=0.5  → equal weight

Usage:
    python scripts/rag/retrieve_hybrid.py \
        --query "What is the deadline to waive UC SHIP?" \
        --alpha 0.5 \
        --top_k 5
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from query_expansion import QueryExpander

INDEX_DIR = Path("data/rag/vector_store")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EXPANSION_CONFIG = Path("configs/rag_query_expansion.json")


def load_index_and_chunks(index_dir: Path) -> tuple:
    index = faiss.read_index(str(index_dir / "index.faiss"))
    chunks = []
    with open(index_dir / "chunk_metadata.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return index, chunks


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric (preserve hyphenated terms)."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def build_bm25(chunks: list[dict]) -> BM25Okapi:
    corpus = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(corpus)


def normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. If all scores equal, return zeros."""
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def retrieve_hybrid(
    query: str,
    index: faiss.Index,
    chunks: list[dict],
    model: SentenceTransformer,
    bm25: BM25Okapi,
    alpha: float = 0.8,
    top_k: int = 5,
    dedup_source: bool = True,
    use_query_expansion: bool = False,
    expander: QueryExpander | None = None,
) -> list[dict]:
    """Return top-k chunks using hybrid dense + BM25 scoring."""
    search_query = query
    if use_query_expansion and expander is not None:
        search_query = expander.expand(query).expanded_query

    n = len(chunks)
    candidate_k = min(n, max(top_k * 20, 100))

    # --- Dense scores ---
    query_vec = model.encode(
        [search_query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    dense_scores_raw, dense_indices = index.search(query_vec, candidate_k)
    dense_scores_raw = dense_scores_raw[0]
    dense_indices = dense_indices[0]

    # Build per-chunk dense score array (unranked chunks get score 0)
    dense_arr = np.zeros(n, dtype=np.float32)
    for score, idx in zip(dense_scores_raw, dense_indices):
        if idx >= 0:
            dense_arr[idx] = max(0.0, float(score))

    # --- BM25 scores ---
    tokens = tokenize(search_query)
    bm25_arr = np.array(bm25.get_scores(tokens), dtype=np.float32)

    # --- Normalize and combine ---
    dense_norm = normalize(dense_arr)
    bm25_norm = normalize(bm25_arr)
    combined = alpha * dense_norm + (1.0 - alpha) * bm25_norm

    # Rank by combined score descending, then dedup by source_id
    ranked_indices = np.argsort(combined)[::-1]
    results = []
    seen: set[str] = set()
    rank = 1
    for idx in ranked_indices:
        chunk = chunks[idx].copy()
        sid = chunk["source_id"]
        if dedup_source and sid in seen:
            continue
        seen.add(sid)
        chunk["rank"] = rank
        chunk["score_hybrid"] = round(float(combined[idx]), 4)
        chunk["score_dense"] = round(float(dense_norm[idx]), 4)
        chunk["score_bm25"] = round(float(bm25_norm[idx]), 4)
        results.append(chunk)
        rank += 1
        if len(results) >= top_k:
            break

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--alpha", type=float, default=0.8,
                        help="Dense weight. 1.0=dense only, 0.0=BM25 only.")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--index_dir", default=str(INDEX_DIR))
    parser.add_argument("--embedding_model", default=EMBEDDING_MODEL)
    parser.add_argument("--use_query_expansion", action="store_true")
    parser.add_argument("--expansion_config", default=str(EXPANSION_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_dir = Path(args.index_dir)
    model = SentenceTransformer(args.embedding_model)
    index, chunks = load_index_and_chunks(index_dir)
    bm25 = build_bm25(chunks)

    expander = None
    if args.use_query_expansion:
        expander = QueryExpander(Path(args.expansion_config))

    results = retrieve_hybrid(
        query=args.query,
        index=index, chunks=chunks, model=model, bm25=bm25,
        alpha=args.alpha, top_k=args.top_k,
        use_query_expansion=args.use_query_expansion,
        expander=expander,
    )

    label = f"alpha={args.alpha} ({'dense only' if args.alpha == 1.0 else 'BM25 only' if args.alpha == 0.0 else 'hybrid'})"
    print(f"Query: {args.query}")
    print(f"Mode:  {label}\n")
    for c in results:
        section = f" § {c['section_title']}" if c.get("section_title") else ""
        print(f"[{c['rank']}] hybrid={c['score_hybrid']:.4f} dense={c['score_dense']:.4f} "
              f"bm25={c['score_bm25']:.4f} | {c['source_id']}{section}")
        print(f"    {c['text'][:180]}...")
        print()


if __name__ == "__main__":
    main()
