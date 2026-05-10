import os
import sys
import pickle
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json

sys.path.insert(0, "/home/avi/projects/R")

from config import cfg
from data.ik_loader import SAMPLE_DOCUMENTS
from ingestion.chunker import RecursiveChunker
from ingestion.embedder import Embedder
from ingestion.vector_store import VectorStore
from ingestion.knowledge_graph import KnowledgeGraph
from agents.graph_rag_agent import MultiAgentGraphRAG

agent: Optional[MultiAgentGraphRAG] = None
kg: Optional[KnowledgeGraph] = None
vector_store: Optional[VectorStore] = None


def build_pipeline():
    """Build the full RAG pipeline on startup."""
    global agent, kg, vector_store

    chunker = RecursiveChunker()
    embedder = Embedder()
    vector_store = VectorStore()
    kg = KnowledgeGraph()

    all_chunks = []
    for doc in SAMPLE_DOCUMENTS:
        chunks = chunker.chunk_document(
            doc_id=doc["id"],
            title=doc["title"],
            text=doc["content"],
            metadata={"source": doc["source"], "year": doc.get("year", 2024)},
        )
        all_chunks.extend(chunks)


    chunks_with_embeddings, embeddings = embedder.embed_chunks(all_chunks)
    vector_store.upsert_chunks(chunks_with_embeddings, embeddings)

    kg_cache = "/home/avi/projects/R/data/kg_cache.pkl"
    if os.path.exists(kg_cache):
        with open(kg_cache, "rb") as f:
            kg.graph = pickle.load(f)
    else:
        kg.extract_and_add(all_chunks)
        with open(kg_cache, "wb") as f:
            pickle.dump(kg.graph, f)

    agent = MultiAgentGraphRAG(
        vector_store=vector_store,
        knowledge_graph=kg,
        embedder=embedder,
    )
    print(f"✓ Pipeline ready — {vector_store.count()} vectors, "
          f"{kg.get_stats()['nodes']} graph nodes")


@asynccontextmanager
async def lifespan(app: FastAPI):
 
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, build_pipeline)
    yield
 

app = FastAPI(
    title="ResearchMind API",
    description="Multi-Agent GraphRAG system over ArXiv papers",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_methods=["*"],
    allow_headers=["*"],
)




class QueryRequest(BaseModel):
    question: str

class Source(BaseModel):
    title: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
    graph_nodes: int
    graph_edges: int
    strategy: str
    retry_count: int




@app.get("/health")
def health():
    return {
        "status": "ready" if agent else "loading",
        "vectors": vector_store.count() if vector_store else 0,
        "graph_nodes": kg.get_stats()["nodes"] if kg else 0,
        "graph_edges": kg.get_stats()["edges"] if kg else 0,
        "documents": len(SAMPLE_DOCUMENTS),
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not agent:
        raise HTTPException(status_code=503, detail="Pipeline still loading")

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    loop = asyncio.get_event_loop()
    final_state = await loop.run_in_executor(None, agent.run, req.question)

    return QueryResponse(
        answer=final_state["answer"],
        sources=final_state["citations"],
        confidence=final_state["confidence_score"],
        graph_nodes=kg.get_stats()["nodes"],
        graph_edges=kg.get_stats()["edges"],
        strategy=final_state["search_strategy"],
        retry_count=final_state["retry_count"],
    )


@app.get("/stats")
def stats():
    if not kg or not vector_store:
        raise HTTPException(status_code=503, detail="Pipeline still loading")
    return {
        "documents": len(SAMPLE_DOCUMENTS),
        "vectors": vector_store.count(),
        **kg.get_stats(),
    }