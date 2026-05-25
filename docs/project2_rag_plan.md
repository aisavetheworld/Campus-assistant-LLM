# Project 2: RAG with Official UCSD Sources

## Goal

Build a retrieval-augmented campus assistant using official UCSD sources so the model does not rely on memorized or hallucinated policy details. Project 1 trained model behavior (tone, structure, escalation, email writing). Project 2 provides factual grounding.

## Motivation

The SFT/DPO model from Project 1 correctly escalates, formats responses, and avoids overconfident claims — but it cannot cite live policy details. Deadlines, fees, office locations, and eligibility rules change. Hard-coding these into training data causes the model to produce stale or incorrect answers. RAG solves this by retrieving relevant passages from official documents at inference time.

## Target Source Categories

| Category | Key topics |
|---|---|
| Student Mail / Mailroom | package tracking, mailroom hours, missing packages, address format |
| Housing | maintenance requests, room assignment, housing office contact, move-in/out |
| Course Enrollment | add/drop deadlines, waitlist, prerequisites, registrar, academic advisor |
| UC SHIP / Insurance | waiver process, waiver deadlines, claims, coverage periods, SHIP office |
| Student Health | immunization requirements, SHS appointments, referrals, healthcare providers |
| International Students | ISEO, CPT eligibility, OPT timeline, visa status, SEVIS, reduced course load |

## RAG Pipeline

```
1. Collect official source metadata
        ↓
2. Fetch or manually store source text
        ↓
3. Clean source text (strip boilerplate, fix encoding)
        ↓
4. Chunk documents (overlapping fixed-size or semantic chunks)
        ↓
5. Embed chunks (sentence-transformers or OpenAI embeddings)
        ↓
6. Store in vector index (FAISS or ChromaDB)
        ↓
7. Retrieve top-k chunks for user query
        ↓
8. Build grounded prompt (system + retrieved context + user question)
        ↓
9. Generate answer using outputs/dpo_7b
        ↓
10. Evaluate retrieval quality and answer faithfulness
```

## Constraints

- Do not hard-code changing deadlines or fees into SFT/DPO data; these belong in RAG source documents.
- Always cite or show source titles and/or URLs in RAG answers.
- If retrieval confidence is low, the model should say it cannot verify and suggest the student contact the relevant official office.
- Preserve safety escalation behavior from Project 1 — do not override high-risk guardrails with retrieved text.
- Do not use Reddit, student forums, or unofficial blogs as factual sources.

## Implementation Plan

### Phase 2.1: Source Collection

- Define and populate `data/rag/ucsd_sources.json` with official source metadata
- Manually collect source text for each entry and save to `data/rag/raw_sources/`
- Follow `docs/rag/source_collection_guide.md` for what counts as acceptable

### Phase 2.2: Chunking

- Implement `scripts/rag/chunk_sources.py`
- Strategy: fixed-size chunks (512 tokens), 50-token overlap
- Output: `data/rag/processed_chunks/` as JSONL with `source_id`, `chunk_id`, `text`, `source_title`, `url`

### Phase 2.3: Embedding and Indexing

- Implement `scripts/rag/build_source_index.py`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (initial), upgradeable
- Index: FAISS flat index (initial), stored in `data/rag/vector_store/`

### Phase 2.4: Retrieval

- Implement `scripts/rag/retrieve.py`
- Input: user query string
- Output: top-k chunks with scores, source titles, and URLs

### Phase 2.5: Grounded Answer Generation

- Implement `scripts/rag/rag_answer.py`
- Builds a grounded prompt: system message + retrieved context block + user question
- Calls `outputs/dpo_7b` for generation
- Adds source citation to the response

### Phase 2.6: Evaluation

- Follow `docs/rag/rag_eval_plan.md`
- Evaluate retrieval relevance, answer faithfulness, citation presence, and safe escalation

## Open Questions

- Embedding model: local (all-MiniLM-L6-v2) vs API (OpenAI ada-002)? Start local.
- Vector store: FAISS (simple, no server) vs ChromaDB (persistent, queryable)? Start FAISS.
- Reranking: BM25 hybrid or dense only? Start dense-only.
- How to handle pages that require login (e.g., TritonLink)? Manual copy only.

## Not In Scope for Project 2

- FastAPI endpoint (Project 3)
- vLLM (Project 3)
- Automatic large-scale web scraping
- Modifying SFT/DPO training stack from Project 1
