"""Evaluate retrieval quality against rag_eval_seed.json.

Loads embedding model, FAISS index, and chunk metadata once.
Processes all eval queries in a single loop.

Usage (baseline):
    python scripts/rag/eval_retrieval.py

Usage (with query expansion):
    python scripts/rag/eval_retrieval.py \
        --use_query_expansion \
        --report_suffix query_expansion

Usage (hybrid FAISS+BM25 with query expansion — recommended):
    python scripts/rag/eval_retrieval.py \
        --hybrid --alpha 0.7 \
        --use_query_expansion \
        --report_suffix hybrid_qe
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from query_expansion import QueryExpander
from retrieve_hybrid import build_bm25, load_reranker, rerank_chunks, retrieve_hybrid

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


def recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for eid in expected_ids if eid in top_k)
    return hits / len(expected_ids) if expected_ids else 0.0


def keyword_hit_rate(retrieved_texts: list[str], keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    combined = " ".join(retrieved_texts).lower()
    return sum(1 for kw in keywords if kw.lower() in combined) / len(keywords)


def mrr(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    expected_set = set(expected_ids)
    for rank, sid in enumerate(retrieved_ids, 1):
        if sid in expected_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int = 5) -> float:
    expected_set = set(expected_ids)
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, sid in enumerate(retrieved_ids[:k])
        if sid in expected_set
    )
    n_rel = min(len(expected_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_all(
    eval_items: list[dict],
    index: faiss.Index,
    chunks: list[dict],
    model: SentenceTransformer,
    top_k: int,
    expander: QueryExpander | None = None,
    bm25=None,
    alpha: float = 0.7,
    reranker=None,
    reranker_backend: str = "crossencoder",
    candidate_k: int = 50,
) -> list[dict]:
    use_hybrid = bm25 is not None

    if use_hybrid:
        # Sequential per-query (required for BM25)
        results = []
        for item in eval_items:
            exp_info: dict = {}
            original_query = item["query"]
            search_query = original_query
            if expander is not None:
                exp = expander.expand(original_query)
                search_query = exp.expanded_query
                exp_info = {
                    "expanded_query": exp.expanded_query,
                    "matched_keys": exp.matched_keys,
                    "appended_terms": exp.appended_terms,
                }

            # Standard retrieval (top_k=5, faiss_k=100) — same as original behavior
            hits = retrieve_hybrid(search_query, index, chunks, model, bm25,
                                   alpha=alpha, top_k=top_k, dedup_source=True)

            # Candidate pool for cR@20/cR@50 and reranking (separate call, larger k)
            if candidate_k > 0 or reranker is not None:
                pool_size = max(candidate_k, top_k)
                hits_pool = retrieve_hybrid(search_query, index, chunks, model, bm25,
                                            alpha=alpha, top_k=pool_size, dedup_source=True)
                candidate_ids = [h["source_id"] for h in hits_pool]
                if reranker is not None:
                    hits = rerank_chunks(original_query, hits_pool[:candidate_k], reranker,
                                         reranker_backend, top_k, dedup_source=True)
            else:
                candidate_ids = [h["source_id"] for h in hits]

            retrieved_ids = [h["source_id"] for h in hits]
            retrieved_texts = [h["text"] for h in hits]
            expected_ids = item.get("expected_source_ids", [])
            keywords = item.get("must_retrieve_keywords", [])

            score_key = "score_rerank" if reranker is not None else "score_hybrid"
            row = {
                "id": item["id"],
                "category": item["category"],
                "query": original_query,
                "expected_source_ids": expected_ids,
                "retrieved_source_ids": retrieved_ids,
                "recall@1": recall_at_k(retrieved_ids, expected_ids, 1),
                "recall@3": recall_at_k(retrieved_ids, expected_ids, 3),
                "recall@5": recall_at_k(retrieved_ids, expected_ids, 5),
                "mrr": mrr(retrieved_ids, expected_ids),
                "ndcg@5": ndcg_at_k(retrieved_ids, expected_ids, 5),
                "candidate_recall@20": recall_at_k(candidate_ids, expected_ids, 20),
                "candidate_recall@50": recall_at_k(candidate_ids, expected_ids, 50),
                "keyword_hit_rate": keyword_hit_rate(retrieved_texts, keywords),
                "top_scores": [round(h.get(score_key, 0.0), 4) for h in hits],
                "safety_expectation": item.get("safety_expectation", ""),
            }
            if exp_info:
                row["expansion"] = exp_info
            results.append(row)
        return results

    # Dense-only batch path
    query_texts: list[str] = []
    expansion_infos: list[dict] = []
    for item in eval_items:
        if expander is not None:
            exp = expander.expand(item["query"])
            query_texts.append(exp.expanded_query)
            expansion_infos.append({
                "expanded_query": exp.expanded_query,
                "matched_keys": exp.matched_keys,
                "appended_terms": exp.appended_terms,
            })
        else:
            query_texts.append(item["query"])
            expansion_infos.append({})

    query_vecs = model.encode(
        query_texts, batch_size=64, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)

    search_k = min(top_k * 10, index.ntotal)
    all_scores, all_indices = index.search(query_vecs, search_k)

    results = []
    for item, scores, indices, exp_info in zip(
        eval_items, all_scores, all_indices, expansion_infos
    ):
        retrieved_ids, retrieved_texts = [], []
        seen: set[str] = set()
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            sid = chunks[idx]["source_id"]
            if sid in seen:
                continue
            seen.add(sid)
            retrieved_ids.append(sid)
            retrieved_texts.append(chunks[idx]["text"])
            if len(retrieved_ids) >= top_k:
                break

        expected_ids = item.get("expected_source_ids", [])
        keywords = item.get("must_retrieve_keywords", [])
        row = {
            "id": item["id"],
            "category": item["category"],
            "query": item["query"],
            "expected_source_ids": expected_ids,
            "retrieved_source_ids": retrieved_ids,
            "recall@1": recall_at_k(retrieved_ids, expected_ids, 1),
            "recall@3": recall_at_k(retrieved_ids, expected_ids, 3),
            "recall@5": recall_at_k(retrieved_ids, expected_ids, 5),
            "keyword_hit_rate": keyword_hit_rate(retrieved_texts, keywords),
            "top_scores": [round(float(s), 4) for s in scores[:top_k]],
            "safety_expectation": item.get("safety_expectation", ""),
        }
        if exp_info:
            row["expansion"] = exp_info
        results.append(row)
    return results


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    agg = {
        "n_queries": n,
        "recall@1": round(sum(r["recall@1"] for r in results) / n, 4),
        "recall@3": round(sum(r["recall@3"] for r in results) / n, 4),
        "recall@5": round(sum(r["recall@5"] for r in results) / n, 4),
        "mrr": round(sum(r.get("mrr", 0.0) for r in results) / n, 4),
        "ndcg@5": round(sum(r.get("ndcg@5", 0.0) for r in results) / n, 4),
        "candidate_recall@20": round(sum(r.get("candidate_recall@20", 0.0) for r in results) / n, 4),
        "candidate_recall@50": round(sum(r.get("candidate_recall@50", 0.0) for r in results) / n, 4),
        "keyword_hit_rate": round(sum(r["keyword_hit_rate"] for r in results) / n, 4),
        "zero_recall@5": sum(1 for r in results if r["recall@5"] == 0.0),
        "zero_recall@20": sum(1 for r in results if r.get("candidate_recall@20", 1.0) == 0.0),
        "zero_recall@50": sum(1 for r in results if r.get("candidate_recall@50", 1.0) == 0.0),
    }
    categories = sorted(set(r["category"] for r in results))
    agg["per_category"] = {}
    for cat in categories:
        cat_rows = [r for r in results if r["category"] == cat]
        nc = len(cat_rows)
        agg["per_category"][cat] = {
            "n": nc,
            "recall@1": round(sum(r["recall@1"] for r in cat_rows) / nc, 4),
            "recall@3": round(sum(r["recall@3"] for r in cat_rows) / nc, 4),
            "recall@5": round(sum(r["recall@5"] for r in cat_rows) / nc, 4),
            "mrr": round(sum(r.get("mrr", 0.0) for r in cat_rows) / nc, 4),
            "ndcg@5": round(sum(r.get("ndcg@5", 0.0) for r in cat_rows) / nc, 4),
            "keyword_hit_rate": round(sum(r["keyword_hit_rate"] for r in cat_rows) / nc, 4),
        }
    return agg


def write_report(
    results: list[dict],
    agg: dict,
    timing: dict,
    out_dir: Path,
    report_name: str = "retrieval_eval_report",
    use_expansion: bool = False,
    use_hybrid: bool = False,
    alpha: float = 0.7,
) -> None:
    title = "# Retrieval Evaluation Report"
    if use_hybrid:
        title += f" (Hybrid alpha={alpha})"
    if use_expansion:
        title += " + Query Expansion"

    lines = [
        title,
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Queries | {agg['n_queries']} |",
        f"| Retrieval mode | {'hybrid FAISS+BM25 alpha=' + str(alpha) if use_hybrid else 'dense-only'} |",
        f"| Query expansion | {'enabled' if use_expansion else 'disabled'} |",
        f"| Recall@1 | {agg['recall@1']:.3f} |",
        f"| Recall@3 | {agg['recall@3']:.3f} |",
        f"| Recall@5 | {agg['recall@5']:.3f} |",
        f"| MRR | {agg.get('mrr', 0.0):.3f} |",
        f"| nDCG@5 | {agg.get('ndcg@5', 0.0):.3f} |",
        f"| Candidate R@20 | {agg.get('candidate_recall@20', 0.0):.3f} |",
        f"| Candidate R@50 | {agg.get('candidate_recall@50', 0.0):.3f} |",
        f"| Zero-recall @5 | {agg.get('zero_recall@5', 0)} |",
        f"| Zero-recall @20 | {agg.get('zero_recall@20', 0)} |",
        f"| Zero-recall @50 | {agg.get('zero_recall@50', 0)} |",
        f"| Keyword Hit Rate | {agg['keyword_hit_rate']:.3f} |",
        f"| Total eval time | {timing['total_s']:.2f}s |",
        f"| Avg per query | {timing['avg_per_query_ms']:.1f}ms |",
        "",
        "## Per-Category Recall@5",
        "",
        "| category | n | R@1 | R@3 | R@5 | MRR | nDCG@5 | KW% |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cat, m in agg["per_category"].items():
        lines.append(
            f"| {cat} | {m['n']} | {m['recall@1']:.3f} | {m['recall@3']:.3f} "
            f"| {m['recall@5']:.3f} | {m.get('mrr', 0.0):.3f} | {m.get('ndcg@5', 0.0):.3f} "
            f"| {m['keyword_hit_rate']:.3f} |"
        )

    lines += [
        "",
        "## Per-Query Results",
        "",
        "| id | category | R@1 | R@3 | R@5 | KW% | retrieved_ids (top-3) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        retrieved = ", ".join(r["retrieved_source_ids"][:3])
        lines.append(
            f"| {r['id']} | {r['category']} "
            f"| {r['recall@1']:.2f} | {r['recall@3']:.2f} | {r['recall@5']:.2f} "
            f"| {r['keyword_hit_rate']:.2f} | {retrieved} |"
        )

    if use_expansion:
        lines += ["", "## Query Expansion Details", ""]
        for r in results:
            exp = r.get("expansion", {})
            if exp.get("matched_keys"):
                lines += [
                    f"**{r['id']}** matched: {exp['matched_keys']}",
                    f"  appended: {exp['appended_terms']}",
                    "",
                ]

    weak = [r for r in results if r["recall@5"] < 1.0]
    if weak:
        lines += ["", "## Weak Cases (R@5 < 1.0)", ""]
        for r in weak:
            lines += [
                f"### {r['id']} — {r['category']}",
                f"**Query:** {r['query']}",
                f"**Expected:** {', '.join(r['expected_source_ids'])}",
                f"**Retrieved:** {', '.join(r['retrieved_source_ids'])}",
                f"**R@5:** {r['recall@5']:.2f}  KW: {r['keyword_hit_rate']:.2f}",
                "",
            ]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{report_name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_seed", default="data/rag/rag_eval_seed.json")
    parser.add_argument("--index_dir", default=str(INDEX_DIR))
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--embedding_model", default=EMBEDDING_MODEL)
    parser.add_argument("--report_dir", default="outputs/rag_eval")
    parser.add_argument("--report_suffix", default="",
                        help="Appended to report filename: retrieval_eval_report_<suffix>")
    parser.add_argument("--use_query_expansion", action="store_true")
    parser.add_argument("--expansion_config", default=str(EXPANSION_CONFIG))
    parser.add_argument("--hybrid", action="store_true",
                        help="Use hybrid FAISS+BM25 retrieval instead of dense-only")
    parser.add_argument("--alpha", type=float, default=0.8,
                        help="Hybrid alpha: weight for dense score (default: 0.8)")
    # Candidate recall
    parser.add_argument("--candidate_k", type=int, default=50,
                        help="Pool size for candidate recall metrics R@20 / R@50 (default: 50).")
    # Reranker
    parser.add_argument("--rerank", action="store_true",
                        help="Enable two-stage reranking with a cross-encoder.")
    parser.add_argument("--reranker_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                        help="Reranker model name.")
    parser.add_argument("--reranker_backend", default="crossencoder",
                        choices=["crossencoder", "flag"],
                        help="'crossencoder' (sentence-transformers) or 'flag' (FlagEmbedding).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_items = json.loads(Path(args.eval_seed).read_text())
    index_dir = Path(args.index_dir)
    report_dir = Path(args.report_dir)

    expander = None
    if args.use_query_expansion:
        expander = QueryExpander(config_path=Path(args.expansion_config))
        print(f"Query expansion: enabled ({len(expander._entries)} keys)")

    print(f"Loading embedding model: {args.embedding_model}")
    model = SentenceTransformer(args.embedding_model)

    print("Loading FAISS index and chunk metadata...")
    index, chunks = load_index_and_chunks(index_dir)
    print(f"  {index.ntotal} vectors, {len(chunks)} chunks")

    bm25 = None
    if args.hybrid:
        print(f"Building BM25 index (hybrid alpha={args.alpha})...")
        bm25 = build_bm25(chunks)

    reranker = None
    if args.rerank:
        if not args.hybrid:
            print("WARNING: --rerank requires --hybrid; enabling hybrid automatically.")
            bm25 = build_bm25(chunks)
        print(f"Loading reranker: {args.reranker_model} ({args.reranker_backend})")
        reranker = load_reranker(args.reranker_model, args.reranker_backend)

    mode = "hybrid" if (args.hybrid or bm25 is not None) else "dense"
    rerank_label = f" + rerank({args.reranker_model})" if reranker else ""
    print(f"Running eval on {len(eval_items)} queries "
          f"(top_k={args.top_k}, mode={mode}{rerank_label}, candidate_k={args.candidate_k})...")
    t0 = time.time()
    results = evaluate_all(
        eval_items, index, chunks, model, args.top_k,
        expander, bm25=bm25, alpha=args.alpha,
        reranker=reranker, reranker_backend=args.reranker_backend,
        candidate_k=args.candidate_k,
    )
    elapsed = time.time() - t0

    timing = {
        "total_s": round(elapsed, 3),
        "avg_per_query_ms": round(elapsed / len(eval_items) * 1000, 1),
    }

    for r in results:
        exp_marker = ""
        if r.get("expansion", {}).get("matched_keys"):
            exp_marker = f" [+{len(r['expansion']['matched_keys'])} keys]"
        print(f"  {r['id']}: R@1={r['recall@1']:.2f} R@3={r['recall@3']:.2f} "
              f"R@5={r['recall@5']:.2f} MRR={r.get('mrr', 0.0):.2f} "
              f"cR@20={r.get('candidate_recall@20', 0.0):.2f} "
              f"cR@50={r.get('candidate_recall@50', 0.0):.2f} "
              f"KW={r['keyword_hit_rate']:.2f}{exp_marker}")

    agg = aggregate(results)
    print(f"\nAggregate ({agg['n_queries']} queries): "
          f"R@1={agg['recall@1']:.3f} R@3={agg['recall@3']:.3f} "
          f"R@5={agg['recall@5']:.3f} MRR={agg.get('mrr', 0.0):.3f} "
          f"nDCG@5={agg.get('ndcg@5', 0.0):.3f} "
          f"cR@20={agg.get('candidate_recall@20', 0.0):.3f} "
          f"cR@50={agg.get('candidate_recall@50', 0.0):.3f} "
          f"KW={agg['keyword_hit_rate']:.3f}")
    print(f"Zero-recall: @5={agg.get('zero_recall@5',0)} "
          f"@20={agg.get('zero_recall@20',0)} @50={agg.get('zero_recall@50',0)}")
    print(f"Timing: {timing['total_s']:.2f}s total, {timing['avg_per_query_ms']:.1f}ms/query")

    suffix = f"_{args.report_suffix}" if args.report_suffix else ""
    report_name = f"retrieval_eval_report{suffix}"
    write_report(results, agg, timing, report_dir, report_name,
                 args.use_query_expansion, args.hybrid, args.alpha)
    report = {"aggregate": agg, "timing": timing, "results": results}
    (report_dir / f"{report_name}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"\nReport → {report_dir}/{report_name}.md")


if __name__ == "__main__":
    main()
