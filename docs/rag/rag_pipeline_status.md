# RAG Pipeline Status

## Implemented Scripts

| Script | Purpose | Status |
|---|---|---|
| `scripts/rag/prepare_sources.py` | Validate source metadata and raw .txt files | Done |
| `scripts/rag/chunk_sources.py` | Chunk raw text into overlapping segments with section detection | Done |
| `scripts/rag/build_index.py` | Embed chunks with sentence-transformers, build FAISS IndexFlatIP | Done |
| `scripts/rag/retrieve.py` | Query the index, return ranked chunks with score/source/url | Done |
| `scripts/rag/eval_retrieval.py` | Compute Recall@1/3/5 and keyword hit rate against eval seed | Done |
| `scripts/rag/rag_answer.py` | Build grounded prompt; optional generation with DPO checkpoint | Done |

## How to Run — Full Pipeline

Run from the repo root in order:

```bash
# 1. Validate source metadata and .txt files
python scripts/rag/prepare_sources.py
# Output: outputs/rag_eval/source_validation_report.md

# 2. Chunk all raw sources
python scripts/rag/chunk_sources.py
# Output: data/rag/processed_chunks/chunks.jsonl

# 3. Embed and build FAISS index
python scripts/rag/build_index.py
# Output: data/rag/vector_store/index.faiss
#         data/rag/vector_store/chunk_metadata.jsonl
#         outputs/rag_eval/index_build_report.md

# 4. Test a single retrieval query
python scripts/rag/retrieve.py --query "What is CPT?" --top_k 3

# 5. Run retrieval evaluation
python scripts/rag/eval_retrieval.py
# Output: outputs/rag_eval/retrieval_eval_report.md
#         outputs/rag_eval/retrieval_eval_report.json

# 6. Preview grounded prompt (no GPU needed)
python scripts/rag/rag_answer.py --query "How do I apply for CPT?" --top_k 3

# 6b. Generate answer with DPO checkpoint (GPU required)
python scripts/rag/rag_answer.py \
    --query "How do I apply for CPT?" \
    --top_k 3 \
    --generate \
    --dpo_adapter_path outputs/dpo_7b
```

## Data Locations

| Path | Contents |
|---|---|
| `data/rag/ucsd_sources.json` | 48 source metadata entries |
| `data/rag/raw_sources/` | Per-source .txt files (46 with content) |
| `data/rag/processed_chunks/chunks.jsonl` | All chunks after chunking step |
| `data/rag/vector_store/index.faiss` | FAISS index (IndexFlatIP, dim=384) |
| `data/rag/vector_store/chunk_metadata.jsonl` | Chunk metadata aligned to FAISS index |
| `data/rag/rag_eval_seed.json` | 10 eval queries with expected sources and keywords |
| `outputs/rag_eval/` | All evaluation reports |

## Current Limitations

1. **SSO-protected page**: `ucsd_housing_assignment_002` (HDH housing portal) requires login — only 14 words were scraped. Source is effectively empty; skip or manually paste content.

2. **Missing URLs**: `ucsd_iseo_general_001` and `ucsd_registrar_waitlist_001` have no URLs yet. Their .txt files need to be populated manually.

3. **Retrieval vocabulary mismatch**: Queries about "full-time enrollment" and visa status may surface OPT chunks instead of visa_status_005, because the enrollment page text doesn't use matching vocabulary. Tracked in eval as a known gap.

4. **Single-hop retrieval only**: The current pipeline does no query expansion or re-ranking. BM25 hybrid search is a planned improvement (Phase 3).

5. **Embedding model is small**: all-MiniLM-L6-v2 (22M params, dim=384) trades quality for speed. Upgrade path: `sentence-transformers/all-mpnet-base-v2` (dim=768) or a domain-tuned model.

6. **Generation not tested locally**: `rag_answer.py --generate` requires a GPU and the DPO checkpoint at `outputs/dpo_7b`. Run this step in Colab.

## Next Steps

- [ ] Fix/fill `ucsd_housing_assignment_002`, `ucsd_iseo_general_001`, `ucsd_registrar_waitlist_001`
- [ ] Run `eval_retrieval.py` and read the report; identify queries with Recall@5 = 0
- [ ] Add BM25 hybrid re-ranking (Phase 3)
- [ ] Run `rag_answer.py --generate` in Colab against the eval seed
- [ ] Expand eval seed from 10 to 25 queries (see `docs/rag/rag_eval_plan.md`)
- [ ] Wire retrieval into Project 3 serving (FastAPI) — not yet started
