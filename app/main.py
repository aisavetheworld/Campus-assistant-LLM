"""FastAPI serving pipeline for the Campus AI Assistant.

Endpoints:
    GET  /health  -- liveness probe
    POST /chat    -- grounded RAG answer with latency breakdown

Generation backends (set via SERVING_BACKEND env var):
    vllm  (default) -- calls vLLM OpenAI-compatible server at VLLM_BASE_URL
    local           -- loads model in-process with transformers + PEFT

Configuration:
    SERVING_BACKEND  -- "vllm" or "local" (default: vllm)
    VLLM_BASE_URL    -- vLLM server URL (default: http://localhost:8000)
    VLLM_MODEL_NAME  -- model name for vLLM (default: campus-assistant)
    BASE_MODEL       -- HuggingFace model id (default: Qwen/Qwen2.5-7B-Instruct)
    ADAPTER_PATH     -- LoRA adapter dir (default: outputs/dpo_7b)
    GEN_MAX_TOKENS   -- max tokens to generate (default: 512)
    GEN_TEMPERATURE  -- sampling temperature (default: 0.2)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "rag"))

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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
        load_generation_model,
        generate_one,
    )
except ImportError as exc:
    raise ImportError(
        f"Could not import from scripts/rag/rag_answer.py: {exc}\n"
        "Make sure sentence-transformers, faiss-cpu, and RAG deps are installed."
    ) from exc

try:
    from answer_validators import run_all_checks, all_passed
except ImportError as exc:
    logging.warning("answer_validators unavailable — safety checks disabled: %s", exc)
    run_all_checks = None  # type: ignore[assignment]
    all_passed = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SERVING_BACKEND: str = os.getenv("SERVING_BACKEND", "vllm")   # "vllm" | "local"

VLLM_BASE_URL: str   = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
VLLM_MODEL_NAME: str = os.getenv("VLLM_MODEL_NAME", "campus-assistant")

BASE_MODEL: str    = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
ADAPTER_PATH: str  = os.getenv("ADAPTER_PATH", "outputs/dpo_7b")

GEN_MAX_TOKENS: int    = int(os.getenv("GEN_MAX_TOKENS", "512"))
GEN_TEMPERATURE: float = float(os.getenv("GEN_TEMPERATURE", "0.2"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("campus_assistant")

app_state: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading RAG components…")
    t0 = time.time()
    emb_model, index, chunks, bm25, expander = load_retrieval_components(
        index_dir=INDEX_DIR,
        embedding_model=EMBEDDING_MODEL,
        expansion_config=EXPANSION_CONFIG,
    )
    app_state.update({
        "embedding_model": emb_model,
        "faiss_index": index,
        "chunks": chunks,
        "bm25": bm25,
        "query_expander": expander,
        "rag_loaded": True,
    })
    logger.info("RAG loaded in %.1f ms — %d chunks.", (time.time() - t0) * 1000, len(chunks))

    if SERVING_BACKEND == "local":
        logger.info("Local backend: loading %s + %s …", BASE_MODEL, ADAPTER_PATH)
        t1 = time.time()
        tokenizer, gen_model = load_generation_model(BASE_MODEL, ADAPTER_PATH)
        app_state["tokenizer"] = tokenizer
        app_state["gen_model"] = gen_model
        logger.info("Model loaded in %.1f ms.", (time.time() - t1) * 1000)

    app_state["model_loaded"] = True
    yield
    logger.info("Shutting down.")
    app_state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Campus AI Assistant", version="0.1.0", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20)

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
# Generation helpers
# ---------------------------------------------------------------------------

def call_vllm(prompt: str, system_message: str,
              max_tokens: int = GEN_MAX_TOKENS,
              temperature: float = GEN_TEMPERATURE) -> str:
    url = f"{VLLM_BASE_URL}/v1/chat/completions"
    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload)
    except httpx.ConnectError:
        raise HTTPException(503, "vLLM server is not running. Start it first.")
    except httpx.TimeoutException as exc:
        raise HTTPException(504, f"vLLM timed out: {exc}")
    if resp.status_code != 200:
        raise HTTPException(502, f"vLLM HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        return resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise HTTPException(502, f"Unexpected vLLM response: {exc}")


def call_local(prompt: str, system_message: str,
               max_tokens: int = GEN_MAX_TOKENS,
               temperature: float = GEN_TEMPERATURE) -> str:
    if not app_state.get("model_loaded"):
        raise HTTPException(503, "Generation model not loaded.")
    return generate_one(
        prompt=prompt,
        system_message=system_message,
        tokenizer=app_state["tokenizer"],
        model=app_state["gen_model"],
        max_new_tokens=max_tokens,
        temperature=temperature,
    )


def generate(prompt: str, system_message: str) -> str:
    if SERVING_BACKEND == "local":
        return call_local(prompt, system_message)
    return call_vllm(prompt, system_message)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    vllm_status = "n/a"
    if SERVING_BACKEND == "vllm":
        try:
            r = httpx.get(f"{VLLM_BASE_URL}/health", timeout=2.0)
            vllm_status = "ok" if r.status_code == 200 else f"http_{r.status_code}"
        except Exception:
            vllm_status = "unreachable"
    return {
        "status": "ok",
        "backend": SERVING_BACKEND,
        "rag_loaded": app_state.get("rag_loaded", False),
        "model_loaded": app_state.get("model_loaded", False),
        "vllm_status": vllm_status,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not app_state.get("rag_loaded"):
        raise HTTPException(503, "RAG not loaded.")

    query  = request.query.strip()
    top_k  = request.top_k
    t0     = time.time()

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
        raise HTTPException(404, "No relevant documents found.")

    t2 = time.time()
    prompt = build_grounded_prompt(query, chunks)
    prompt_build_ms = (time.time() - t2) * 1000

    t3 = time.time()
    answer = generate(prompt, GROUNDED_SYSTEM_MESSAGE)
    generation_ms = (time.time() - t3) * 1000

    t4 = time.time()
    if run_all_checks is not None:
        validation = run_all_checks(answer=answer, chunks=chunks, query=query, category=None)
        safety_ok  = all(v["passed"] for v in validation.values())
    else:
        validation, safety_ok = {}, True
    validation_ms = (time.time() - t4) * 1000

    total_ms = (time.time() - t0) * 1000

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

    expanded_query = chunks[0].get("expanded_query", "") if chunks else ""

    logger.info("chat | query=%r | backend=%s | total_ms=%.1f",
                query[:80], SERVING_BACKEND, total_ms)

    return ChatResponse(
        answer=answer,
        sources=sources,
        retrieval_metadata=RetrievalMetadata(
            top_k=top_k,
            chunks_retrieved=len(chunks),
            query_expanded=bool(expanded_query and expanded_query != query),
            expanded_query=expanded_query,
            scores=[round(c.get("score_hybrid", 0.0), 4) for c in chunks],
        ),
        safety_metadata=SafetyMetadata(all_passed=safety_ok, checks=validation),
        latency=LatencyBreakdown(
            retrieval_ms=round(retrieval_ms, 2),
            prompt_build_ms=round(prompt_build_ms, 2),
            generation_ms=round(generation_ms, 2),
            validation_ms=round(validation_ms, 2),
            total_ms=round(total_ms, 2),
        ),
    )
