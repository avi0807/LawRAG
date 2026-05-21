import asyncio
import json
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.graph_rag_agent import MultiAgentGraphRAG
from config import cfg
from data.ik_loader import SAMPLE_DOCUMENTS
from ingestion.bm25_store import BM25Store
from ingestion.chunker import RecursiveChunker
from ingestion.embedder import Embedder
from ingestion.knowledge_graph import KnowledgeGraph
from ingestion.vector_store import VectorStore
from observability import log

KG_CACHE = "data/kg_cache.pkl"
BM25_CACHE = "data/bm25_cache.pkl"

agent: Optional[MultiAgentGraphRAG] = None
kg: Optional[KnowledgeGraph] = None
vector_store: Optional[VectorStore] = None
bm25_store: Optional[BM25Store] = None

# In-memory conversation store per session
conversation_store: dict = defaultdict(list)

# Per-session sliding-window rate limiter: session_id -> deque of timestamps
_rate_buckets: dict = defaultdict(lambda: deque(maxlen=cfg.rate_limit_per_minute * 4))


def build_pipeline():
    global agent, kg, vector_store, bm25_store

    chunker = RecursiveChunker()
    embedder = Embedder()
    vector_store = VectorStore()
    kg = KnowledgeGraph()
    bm25_store = BM25Store(cache_path=BM25_CACHE)

    all_chunks = []
    for doc in SAMPLE_DOCUMENTS:
        chunks = chunker.chunk_document(
            doc_id=doc["id"],
            title=doc["title"],
            text=doc["content"],
            metadata={"source": doc["source"], "year": doc.get("year", 2024)},
        )
        all_chunks.extend(chunks)

    if vector_store.count() > 0:
        print(f"Vector store already populated: {vector_store.count()} vectors — skipping embedding")
    else:
        chunks_with_embeddings, embeddings = embedder.embed_chunks(all_chunks)
        vector_store.upsert_chunks(chunks_with_embeddings, embeddings)

    bm25_store.build_or_load(all_chunks)

    if kg.load(KG_CACHE):
        if not kg.entity_chunks:
            for ch in all_chunks:
                kg._index_chunk_for_entities(ch, [])
    else:
        kg.extract_and_add(all_chunks)
        kg.save(KG_CACHE)

    agent = MultiAgentGraphRAG(
        vector_store=vector_store,
        knowledge_graph=kg,
        embedder=embedder,
        bm25_store=bm25_store,
    )
    print(
        f"✓ Pipeline ready — {vector_store.count()} vectors, "
        f"{bm25_store.count()} BM25 chunks, "
        f"{kg.get_stats()['nodes']} graph nodes"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, build_pipeline)
    yield


app = FastAPI(
    title="LawRAG API",
    description="Multi-Agent GraphRAG over Indian Supreme Court judgments (BM25 + Dense + Graph)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────
# Auth & rate limiting
# ──────────────────────────────────────────────────────────

def _require_auth(request: Request) -> None:
    """If LAWRAG_API_TOKEN is set in env, every request must carry a matching
    Bearer token. If unset, auth is disabled (dev mode)."""
    if not cfg.api_token:
        return
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth.split(" ", 1)[1].strip()
    if token != cfg.api_token:
        raise HTTPException(status_code=403, detail="Invalid token")


def _rate_limit(session_id: str, request: Request) -> None:
    """Sliding-window rate limit. Keyed by session_id (falls back to client IP)."""
    key = session_id or (request.client.host if request.client else "anon")
    now = time.time()
    bucket = _rate_buckets[key]
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= cfg.rate_limit_per_minute:
        log.warning("rate_limit.exceeded", extra={"session_id": key, "size": len(bucket)})
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({cfg.rate_limit_per_minute}/min). Try again shortly.",
        )
    bucket.append(now)


class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
    confidence_breakdown: dict = {}
    grounding: list = []
    grounding_stats: dict = {}
    graph_nodes: int
    graph_edges: int
    strategy: str
    retry_count: int


@app.get("/health")
def health():
    return {
        "status": "ready" if agent else "loading",
        "vectors": vector_store.count() if vector_store else 0,
        "bm25_chunks": bm25_store.count() if bm25_store else 0,
        "graph_nodes": kg.get_stats()["nodes"] if kg else 0,
        "graph_edges": kg.get_stats()["edges"] if kg else 0,
        "documents": len(SAMPLE_DOCUMENTS),
    }


def _build_contextual_question(req: QueryRequest) -> str:
    history = conversation_store[req.session_id]
    if not history:
        return req.question
    history_text = "\n".join(
        f"Q: {h['question']}\nA: {h['answer'][:300]}" for h in history[-3:]
    )
    return f"Previous conversation:\n{history_text}\n\nNew question: {req.question}"


def _remember(session_id: str, question: str, answer: str):
    conversation_store[session_id].append({"question": question, "answer": answer})
    if len(conversation_store[session_id]) > 10:
        conversation_store[session_id] = conversation_store[session_id][-10:]


@app.post("/query")
async def query(req: QueryRequest, request: Request):
    _require_auth(request)
    _rate_limit(req.session_id, request)

    if not agent:
        raise HTTPException(status_code=503, detail="Pipeline still loading")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(req.question) > 2000:
        raise HTTPException(status_code=413, detail="Question too long (>2000 chars)")

    contextual = _build_contextual_question(req)

    if req.stream:
        return StreamingResponse(
            _stream_query(req, contextual),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    loop = asyncio.get_event_loop()
    final_state = await loop.run_in_executor(
        None, agent.run, contextual, req.session_id
    )

    _remember(req.session_id, req.question, final_state["answer"])

    return QueryResponse(
        answer=final_state["answer"],
        sources=final_state["citations"],
        confidence=final_state["confidence_score"],
        confidence_breakdown=final_state.get("confidence_breakdown", {}),
        grounding=final_state.get("grounding", []),
        grounding_stats=final_state.get("grounding_stats", {}),
        graph_nodes=kg.get_stats()["nodes"],
        graph_edges=kg.get_stats()["edges"],
        strategy=final_state["search_strategy"],
        retry_count=final_state["retry_count"],
    )


async def _stream_query(req: QueryRequest, contextual: str):
    """
    Real SSE streaming. The retrieval pipeline runs in a thread; tokens from
    Ollama are pushed to the client as they arrive.
    """
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()
    final_holder = {}

    def producer():
        try:
            for event in agent.run_stream(contextual, req.session_id):
                if event["type"] == "final":
                    final_holder["event"] = event
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as e:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "error", "message": str(e)},
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

    loop.run_in_executor(None, producer)

    while True:
        event = await queue.get()
        if event is SENTINEL:
            break

        if event["type"] == "token":
            yield f"data: {json.dumps({'word': event['text'], 'done': False})}\n\n"
        elif event["type"] == "status":
            payload = {"done": False, "type": "status", "stage": event.get("stage")}
            # forward any extra fields the agent attached (entities, source_counts, etc.)
            for k, v in event.items():
                if k not in ("type", "stage"):
                    payload[k] = v
            yield f"data: {json.dumps(payload)}\n\n"
        elif event["type"] == "final":
            payload = {
                "done": True,
                "sources": event["citations"],
                "confidence": event["confidence"],
                "confidence_breakdown": event.get("confidence_breakdown", {}),
                "grounding_stats": event.get("grounding_stats", {}),
                "grounding": event.get("grounding", []),
                "strategy": event.get("strategy", "hybrid"),
                "retry_count": event.get("retry_count", 0),
            }
            yield f"data: {json.dumps(payload)}\n\n"
        elif event["type"] == "error":
            yield f"data: {json.dumps({'done': True, 'error': event['message']})}\n\n"

    # Persist memory after stream completes
    final = final_holder.get("event")
    if final:
        _remember(req.session_id, req.question, final["answer"])


@app.delete("/conversation/{session_id}")
def clear_conversation(session_id: str):
    conversation_store[session_id] = []
    return {"cleared": session_id}


@app.get("/stats")
def stats():
    if not kg or not vector_store:
        raise HTTPException(status_code=503, detail="Pipeline still loading")
    return {
        "documents": len(SAMPLE_DOCUMENTS),
        "vectors": vector_store.count(),
        "bm25_chunks": bm25_store.count() if bm25_store else 0,
        **kg.get_stats(),
    }
