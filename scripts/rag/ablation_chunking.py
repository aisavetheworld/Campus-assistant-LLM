"""Chunking parameter ablation: compare chunk_size/overlap configs on Recall@k.

Controlled variables:
  - embedding model: sentence-transformers/all-MiniLM-L6-v2
  - retrieval: FAISS IndexFlatIP (cosine, normalized)
  - source dedup: enabled (each source_id contributes at most one chunk to top-k)
  - top_k: 5

Configs tested: 256/50, 512/50, 512/100, 1024/100

Usage:
    python scripts/rag/ablation_chunking.py
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
from chunk_sources import build_chunks

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CATEGORIES = ["international_students", "course_enrollment", "health_insurance",
               "student_health", "housing"]

CONFIGS = [
    {"chunk_size": 256, "overlap": 50},
    {"chunk_size": 512, "overlap": 50},
    {"chunk_size": 512, "overlap": 100},
    {"chunk_size": 1024, "overlap": 100},
]


def build_index(chunks: list[dict], model: SentenceTransformer) -> faiss.Index:
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    top = set(retrieved[:k])
    return sum(1 for e in expected if e in top) / len(expected) if expected else 0.0


def kw_hit(texts: list[str], keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    combined = " ".join(texts).lower()
    return sum(1 for kw in keywords if kw.lower() in combined) / len(keywords)


def eval_config(
    config: dict,
    sources: list[dict],
    raw_dir: Path,
    eval_items: list[dict],
    model: SentenceTransformer,
    top_k: int = 5,
) -> dict:
    chunk_size, overlap = config["chunk_size"], config["overlap"]
    label = f"{chunk_size}/{overlap}"

    all_chunks: list[dict] = []
    for source in sources:
        all_chunks.extend(build_chunks(source, raw_dir, chunk_size, overlap))

    t0 = time.time()
    index = build_index(all_chunks, model)
    build_time = time.time() - t0

    avg_wc = (sum(c["word_count"] for c in all_chunks) / len(all_chunks)) if all_chunks else 0

    # Batch encode all queries
    queries = [item["query"] for item in eval_items]
    query_vecs = model.encode(
        queries, batch_size=64, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)

    search_k = min(top_k * 10, index.ntotal)
    all_scores, all_indices = index.search(query_vecs, search_k)

    per_query = []
    for item, scores, indices in zip(eval_items, all_scores, all_indices):
        retrieved_ids, retrieved_texts = [], []
        seen: set[str] = set()
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            sid = all_chunks[idx]["source_id"]
            if sid in seen:
                continue
            seen.add(sid)
            retrieved_ids.append(sid)
            retrieved_texts.append(all_chunks[idx]["text"])
            if len(retrieved_ids) >= top_k:
                break

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

    per_cat: dict[str, dict] = {}
    for cat in CATEGORIES:
        cat_rows = [r for r in per_query if r["category"] == cat]
        if not cat_rows:
            continue
        nc = len(cat_rows)
        per_cat[cat] = {
            "n": nc,
            "recall@5": round(sum(r["recall@5"] for r in cat_rows) / nc, 4),
        }

    weak = [r for r in per_query if r["recall@5"] < 1.0]

    return {
        "config": label,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "n_chunks": len(all_chunks),
        "avg_word_count": round(avg_wc, 1),
        "build_time_s": round(build_time, 2),
        "recall@1": avg("recall@1"),
        "recall@3": avg("recall@3"),
        "recall@5": avg("recall@5"),
        "keyword_hit_rate": avg("kw"),
        "per_category": per_cat,
        "weak_cases": weak,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources_meta", default="data/rag/ucsd_sources.json")
    parser.add_argument("--raw_dir", default="data/rag/raw_sources")
    parser.add_argument("--eval_seed", default="data/rag/rag_eval_seed.json")
    parser.add_argument("--report_dir", default="outputs/rag_eval")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--embedding_model", default=EMBEDDING_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = json.loads(Path(args.sources_meta).read_text())
    raw_dir = Path(args.raw_dir)
    eval_items = json.loads(Path(args.eval_seed).read_text())

    print(f"Loading embedding model: {args.embedding_model}")
    model = SentenceTransformer(args.embedding_model)
    print(f"Running ablation on {len(CONFIGS)} configs × {len(eval_items)} queries...\n")

    rows = []
    for config in CONFIGS:
        label = f"{config['chunk_size']}/{config['overlap']}"
        print(f"  Config {label}...", end=" ", flush=True)
        row = eval_config(config, sources, raw_dir, eval_items, model, args.top_k)
        rows.append(row)
        print(
            f"chunks={row['n_chunks']} avg_wc={row['avg_word_count']:.0f} "
            f"R@1={row['recall@1']:.3f} R@3={row['recall@3']:.3f} "
            f"R@5={row['recall@5']:.3f} KW={row['keyword_hit_rate']:.3f}"
        )

    print("\n" + "=" * 65)
    print(f"{'config':<12} {'chunks':>6} {'avg_wc':>7} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'KW%':>6}")
    print("-" * 65)
    for r in rows:
        print(
            f"{r['config']:<12} {r['n_chunks']:>6} {r['avg_word_count']:>7.0f} "
            f"{r['recall@1']:>6.3f} {r['recall@3']:>6.3f} {r['recall@5']:>6.3f} "
            f"{r['keyword_hit_rate']:>6.3f}"
        )

    # Save JSON for report generation
    Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.report_dir) / "chunking_ablation_raw.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    print(f"\nRaw results → {args.report_dir}/chunking_ablation_raw.json")


if __name__ == "__main__":
    main()
