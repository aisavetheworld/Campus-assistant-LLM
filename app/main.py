"""FastAPI skeleton for the Campus AI Assistant serving pipeline.

Endpoints:
    GET  /health  -- liveness + vLLM reachability check
    POST /chat    -- grounded RAG answer with latency breakdown

Startup:
    RAG components (embedding model, FAISS index, BM25, query expander) are
    loaded once via the lifespan context manager so every request can reuse them.

Configuration (all overrideable via env vars):
    VLLM_BASE_URL      -- base URL of the vLLM OpenAI-compatible server
                          (default: http://localhost:8000)
    VLLM_MODEL_NAME    -- model name passed in chat-completions requests
                          (default: campus-assistant)
    VLLM_MAX_TOKENS    -- max tokens to generate (default: 512)
    VLLM_TEMPERATURE   -- sampling temperature (default: 0.2)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# sys.path fix: allow "from <module> import ..." for scripts/rag/*.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "rag"))

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# RAG imports (graceful degradation if heavy deps aren't installed yet)
# ---------------------------------------------------------------------------
try:
    from rag_answer import (
        INDEX_DIR,
        EMBEDDING_MODEL,
        EXPANSION_CONFIG,
        DEFAULT_ALPHA,
        DEFAULT_TOP_K,
        GROUNDED_SYSTEM_MESSAGE,
        load_retrieval_components,
        run_retrieval,
        build_grounded_prompt,
    )
except ImportError as exc:
    raise ImportError(
        f"Could not import from scripts/rag/rag_answer.py: {exc}\n"
        "Make sure sentence-transformers, faiss-cpu, and other RAG deps are installed."
    ) from exc

try:
    from answer_validators import run_all_checks, all_passed
except ImportError as exc:
    # Non-fatal: safety validation will be skipped with a warning
    logging.warning("answer_validators unavailable — safety checks disabled: %s", exc)
    run_all_checks = None  # type: ignore[assignment]
    all_passed = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
VLLM_MODEL_NAME: str = os.getenv("VLLM_MODEL_NAME", "campus-assistant")
VLLM_MAX_TOKENS: int = int(os.getenv("VLLM_MAX_TOKENS", "512"))
VLLM_TEMPERATURE: float = float(os.getenv("VLLM_TEMPERATURE", "0.2"))

# Short timeout used for the vLLM reachability probe in /health
_HEALTH_PROBE_TIMEOUT: float = 2.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("campus_assistant")

# ---------------------------------------------------------------------------
# Module-level state (populated at startup)
# ---------------------------------------------------------------------------
app_state: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Lifespan: load RAG components once at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load RAG components at startup; clean up on shutdown."""
    logger.info("Loading RAG components…")
    t_start = time.time()

    model, index, chunks, bm25, expander = load_retrieval_components(
        index_dir=INDEX_DIR,
        embedding_model=EMBEDDING_MODEL,
        expansion_config=EXPANSION_CONFIG,
    )

    app_state["embedding_model"] = model
    app_state["faiss_index"] = index
    app_state["chunks"] = chunks
    app_state["bm25"] = bm25
    app_state["query_expander"] = expander
    app_state["rag_loaded"] = True

    elapsed = (time.time() - t_start) * 1000
    logger.info(
        "RAG components loaded in %.1f ms — %d vectors, %d chunks.",
        elapsed,
        index.ntotal,
        len(chunks),
    )

    yield  # application is running

    # Shutdown: nothing heavy to clean up
    logger.info("Shutting down — clearing app state.")
    app_state.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Campus AI Assistant",
    description="Grounded RAG Q&A for UCSD students, backed by vLLM.",
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Student question")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20, description="Number of chunks to retrieve")


class SourceItem(BaseModel):
    source_id: str
    title: str
    url: str
    section_title: str
    score: float
    rank: int


class RetrievalMetadata(BaseModel):
    top_k: int
    chunks_retrieved: int
    query_expanded: bool
    expanded_query: str
    scores: list[float]


class SafetyMetadata(BaseModel):
    all_passed: bool
    checks: dict[str, Any]


class LatencyBreakdown(BaseModel):
    retrieval_ms: float
    prompt_build_ms: float
    generation_ms: float
    validation_ms: float
    total_ms: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    retrieval_metadata: RetrievalMetadata
    safety_metadata: SafetyMetadata
    latency: LatencyBreakdown


# ---------------------------------------------------------------------------
# Helper: call vLLM
# ---------------------------------------------------------------------------


def call_vllm(
    prompt: str,
    system_message: str,
    max_tokens: int = VLLM_MAX_TOKENS,
    temperature: float = VLLM_TEMPERATURE,
) -> str:
    """POST to vLLM's OpenAI-compatible chat completions endpoint.

    Raises:
        HTTPException(503): vLLM server is not reachable.
        HTTPException(502): vLLM returned a non-200 response.
    """
    url = f"{VLLM_BASE_URL}/v1/chat/completions"
    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload)
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=(
                "vLLM server is not running. "
                "Start it first with the vllm_runbook.md instructions."
            ),
        )
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail=f"vLLM request timed out: {exc}",
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"vLLM returned HTTP {response.status_code}: {response.text[:200]}",
        )

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected vLLM response shape: {exc}. Body: {str(data)[:200]}",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    """Liveness probe + vLLM reachability check."""
    rag_loaded: bool = app_state.get("rag_loaded", False)

    # Probe vLLM
    vllm_status: str
    try:
        with httpx.Client(timeout=_HEALTH_PROBE_TIMEOUT) as client:
            resp = client.get(f"{VLLM_BASE_URL}/health")
        vllm_status = "ok" if resp.status_code == 200 else f"http_{resp.status_code}"
    except Exception:
        vllm_status = "unreachable"

    return {
        "status": "ok",
        "rag_loaded": rag_loaded,
        "vllm_url": f"{VLLM_BASE_URL}/v1",
        "vllm_status": vllm_status,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Answer a student question using RAG + vLLM generation."""
    if not app_state.get("rag_loaded"):
        raise HTTPException(status_code=503, detail="RAG components are not loaded yet.")

    query = request.query.strip()
    top_k = request.top_k

    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Retrieval
    # ------------------------------------------------------------------
    t1 = time.time()
    chunks = run_retrieval(
        query=query,
        model=app_state["embedding_model"],
        index=app_state["faiss_index"],
        chunks=app_state["chunks"],
        bm25=app_state["bm25"],
        expander=app_state["query_expander"],
        top_k=top_k,
        alpha=DEFAULT_ALPHA,
    )
    retrieval_ms = (time.time() - t1) * 1000

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found for this query.")

    # ------------------------------------------------------------------
    # 2. Build grounded prompt
    # ------------------------------------------------------------------
    t2 = time.time()
    prompt = build_grounded_prompt(query, chunks)
    prompt_build_ms = (time.time() - t2) * 1000

    # ------------------------------------------------------------------
    # 3. Call vLLM
    # ------------------------------------------------------------------
    t3 = time.time()
    answer = call_vllm(prompt, system_message=GROUNDED_SYSTEM_MESSAGE)
    generation_ms = (time.time() - t3) * 1000

    # ------------------------------------------------------------------
    # 4. Safety validation
    # ------------------------------------------------------------------
    t4 = time.time()
    if run_all_checks is not None:
        validation = run_all_checks(
            answer=answer,
            chunks=chunks,
            query=query,
            category=None,
        )
        safety_all_passed = all(v["passed"] for v in validation.values())
    else:
        validation = {}
        safety_all_passed = True
    validation_ms = (time.time() - t4) * 1000

    total_ms = (time.time() - t0) * 1000

    # ------------------------------------------------------------------
    # 5. Build response
    # ------------------------------------------------------------------
    sources = [
        SourceItem(
            source_id=c.get("source_id", ""),
            title=c.get("title", ""),
            url=c.get("url", ""),
            section_title=c.get("section_title", ""),
            score=round(c.get("score_hybrid", 0.0), 4),
            rank=c.get("rank", i + 1),
        )
        for i, c in enumerate(chunks)
    ]

    # Determine whether query expansion was used (expander adds an expanded_query field)
    expanded_query: str = chunks[0].get("expanded_query", "") if chunks else ""
    query_expanded: bool = bool(expanded_query and expanded_query != query)

    retrieval_metadata = RetrievalMetadata(
        top_k=top_k,
        chunks_retrieved=len(chunks),
        query_expanded=query_expanded,
        expanded_query=expanded_query,
        scores=[round(c.get("score_hybrid", 0.0), 4) for c in chunks],
    )

    safety_metadata = SafetyMetadata(
        all_passed=safety_all_passed,
        checks=validation,
    )

    latency = LatencyBreakdown(
        retrieval_ms=round(retrieval_ms, 2),
        prompt_build_ms=round(prompt_build_ms, 2),
        generation_ms=round(generation_ms, 2),
        validation_ms=round(validation_ms, 2),
        total_ms=round(total_ms, 2),
    )

    logger.info(
        "chat | query=%r | chunks=%d | total_ms=%.1f",
        query[:80],
        len(chunks),
        total_ms,
    )

    return ChatResponse(
        answer=answer,
        sources=sources,
        retrieval_metadata=retrieval_metadata,
        safety_metadata=safety_metadata,
        latency=latency,
    )
