"""Hybrid retrieval ablation: Dense vs BM25 vs Hybrid at alpha ∈ {0.3, 0.5, 0.7}.

Also runs the best hybrid config with query expansion for comparison.

Controlled variables:
  - Chunking: 512/50 (default)
  - Embedding: sentence-transformers/all-MiniLM-L6-v2
  - Source dedup: enabled
  - top_k: 5
  - Eval: 25 queries from data/rag/rag_eval_seed.json

Usage:
    python scripts/rag/ablation_hybrid.py
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

sys.path.insert(0, str(Path(__file__).parent))
from retrieve_hybrid import build_bm25, retrieve_hybrid, load_index_and_chunks
from query_expansion import QueryExpander

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = Path("data/rag/vector_store")
EXPANSION_CONFIG = Path("configs/rag_query_expansion.json")

ALPHA_CONFIGS = [
    {"label": "dense_only",    "alpha": 1.0, "use_expansion": False},
    {"label": "bm25_only",     "alpha": 0.0, "use_expansion": False},
    {"label": "hybrid_a0.3",   "alpha": 0.3, "use_expansion": False},
    {"label": "hybrid_a0.5",   "alpha": 0.5, "use_expansion": False},
    {"label": "hybrid_a0.7",   "alpha": 0.7, "use_expansion": False},
    {"label": "hybrid_a0.5+qe","alpha": 0.5, "use_expansion": True},
    {"label": "hybrid_a0.7+qe","alpha": 0.7, "use_expansion": True},
]

CATEGORIES = ["international_students", "course_enrollment",
               "health_insurance", "student_health", "housing"]


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    top = set(retrieved[:k])
    return sum(1 for e in expected if e in top) / len(expected) if expected else 0.0


def kw_hit(texts: list[str], keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    combined = " ".join(texts).lower()
    return sum(1 for kw in keywords if kw.lower() in combined) / len(keywords)


def eval_config(
    cfg: dict,
    eval_items: list[dict],
    index: faiss.Index,
    chunks: list[dict],
    model: SentenceTransformer,
    bm25,
    expander: QueryExpander,
    top_k: int = 5,
) -> dict:
    per_query = []
    for item in eval_items:
        results = retrieve_hybrid(
            query=item["query"],
            index=index, chunks=chunks, model=model, bm25=bm25,
            alpha=cfg["alpha"], top_k=top_k,
            dedup_source=True,
            use_query_expansion=cfg["use_expansion"],
            expander=expander if cfg["use_expansion"] else None,
        )
        retrieved_ids = [r["source_id"] for r in results]
        retrieved_texts = [r["text"] for r in results]
        expected = item.get("expected_source_ids", [])
        keywords = item.get("must_retrieve_keywords", [])
        per_query.append({
            "id": item["id"],
            "category": item["category"],
            "query": item["query"],
            "expected": expected,
            "retrieved": retrieved_ids,
            "recall@1": recall_at_k(retrieved_ids, expected, 1),
            "recall@3": recall_at_k(retrieved_ids, expected, 3),
            "recall@5": recall_at_k(retrieved_ids, expected, 5),
            "kw": kw_hit(retrieved_texts, keywords),
        })

    n = len(per_query)

    def avg(key: str) -> float:
        return round(sum(r[key] for r in per_query) / n, 4)

    per_cat = {}
    for cat in CATEGORIES:
        rows = [r for r in per_query if r["category"] == cat]
        if not rows:
            continue
        nc = len(rows)
        per_cat[cat] = {
            "n": nc,
            "recall@5": round(sum(r["recall@5"] for r in rows) / nc, 4),
        }

    return {
        "label": cfg["label"],
        "alpha": cfg["alpha"],
        "use_expansion": cfg["use_expansion"],
        "recall@1": avg("recall@1"),
        "recall@3": avg("recall@3"),
        "recall@5": avg("recall@5"),
        "keyword_hit_rate": avg("kw"),
        "per_category": per_cat,
        "weak_ids": [r["id"] for r in per_query if r["recall@5"] < 1.0],
        "per_query": per_query,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_seed", default="data/rag/rag_eval_seed.json")
    parser.add_argument("--index_dir", default=str(INDEX_DIR))
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--embedding_model", default=EMBEDDING_MODEL)
    parser.add_argument("--report_dir", default="outputs/rag_eval")
    parser.add_argument("--expansion_config", default=str(EXPANSION_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_items = json.loads(Path(args.eval_seed).read_text())
    report_dir = Path(args.report_dir)

    print(f"Loading model and index...")
    model = SentenceTransformer(args.embedding_model)
    index, chunks = load_index_and_chunks(Path(args.index_dir))
    bm25 = build_bm25(chunks)
    expander = QueryExpander(Path(args.expansion_config))
    print(f"  {index.ntotal} vectors, {len(chunks)} chunks, BM25 corpus ready")

    rows = []
    for cfg in ALPHA_CONFIGS:
        t0 = time.time()
        print(f"  [{cfg['label']}]...", end=" ", flush=True)
        row = eval_config(cfg, eval_items, index, chunks, model, bm25, expander, args.top_k)
        elapsed = time.time() - t0
        row["eval_time_s"] = round(elapsed, 2)
        rows.append(row)
        print(
            f"R@1={row['recall@1']:.3f} R@3={row['recall@3']:.3f} "
            f"R@5={row['recall@5']:.3f} KW={row['keyword_hit_rate']:.3f} "
            f"weak={len(row['weak_ids'])} ({elapsed:.1f}s)"
        )

    print("\n" + "=" * 75)
    print(f"{'label':<20} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'KW%':>6} {'weak':>5}")
    print("-" * 75)
    for r in rows:
        print(
            f"{r['label']:<20} {r['recall@1']:>6.3f} {r['recall@3']:>6.3f} "
            f"{r['recall@5']:>6.3f} {r['keyword_hit_rate']:>6.3f} {len(r['weak_ids']):>5}"
        )

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "hybrid_ablation_raw.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    print(f"\nRaw results → {report_dir}/hybrid_ablation_raw.json")


if __name__ == "__main__":
    main()
