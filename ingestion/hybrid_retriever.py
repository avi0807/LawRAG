"""
HybridRetriever: BM25 + dense + graph (PPR + entity-chunk lookup) fused with RRF,
then reranked once by FlashRank.

Returns the top `final_top_k` chunks ready for the generator.
"""
from __future__ import annotations

from typing import List, Dict, Iterable, Optional

from rich.console import Console

from config import cfg
from ingestion.bm25_store import BM25Store
from ingestion.embedder import Embedder
from ingestion.knowledge_graph import KnowledgeGraph
from ingestion.vector_store import VectorStore

console = Console()


def _chunk_key(c: Dict) -> str:
    """Stable id for de-dup / RRF accumulation."""
    return c.get("chunk_id") or c.get("text", "")[:80]


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict]],
    k: int = None,
) -> List[Dict]:
    """Standard RRF. Each list is in best-first order. Returns merged list sorted
    by fused score, preserving the richest payload version of each chunk."""
    if k is None:
        k = cfg.rrf_k

    fused: Dict[str, float] = {}
    keep: Dict[str, Dict] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            key = _chunk_key(item)
            if not key:
                continue
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank + 1)
            # Keep the version with the most payload (longest text wins ties).
            existing = keep.get(key)
            if existing is None or len(item.get("text", "")) > len(existing.get("text", "")):
                keep[key] = item

    merged = []
    for key, score in sorted(fused.items(), key=lambda x: x[1], reverse=True):
        c = dict(keep[key])
        c["rrf_score"] = round(score, 5)
        merged.append(c)
    return merged


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        knowledge_graph: KnowledgeGraph,
        embedder: Embedder,
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.kg = knowledge_graph
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        seed_entities: Optional[List[str]] = None,
        strategy: str = "hybrid",
    ) -> Dict:
        """
        strategy: "vector" | "bm25" | "graph" | "hybrid"
        Returns dict with `chunks` (final top_k) and `graph_context`.
        """
        seed_entities = seed_entities or []
        ranked_lists: List[List[Dict]] = []
        graph_context: Dict = {}
        source_counts = {"bm25": 0, "dense": 0, "graph": 0}

        # ---- BM25 ----
        if strategy in ("bm25", "hybrid"):
            bm25_hits = self.bm25_store.search(query, top_k=cfg.bm25_top_k)
            source_counts["bm25"] = len(bm25_hits)
            if bm25_hits:
                ranked_lists.append(bm25_hits)
                console.print(f"  BM25: {len(bm25_hits)} hits")

        # ---- Dense ----
        if strategy in ("vector", "hybrid"):
            qvec = self.embedder.embed_query(query)
            dense_hits = self.vector_store.search(
                query_vector=qvec,
                query_text=query,
                top_k=cfg.dense_top_k,
                rerank=False,
            )
            source_counts["dense"] = len(dense_hits)
            if dense_hits:
                ranked_lists.append(dense_hits)
                console.print(f"  Dense: {len(dense_hits)} hits")

        # ---- Graph (PPR + entity→chunks) ----
        if strategy in ("graph", "hybrid") and seed_entities:
            graph_chunks, graph_context = self._graph_chunks(seed_entities)
            source_counts["graph"] = len(graph_chunks)
            if graph_chunks:
                ranked_lists.append(graph_chunks)
                console.print(f"  Graph: {len(graph_chunks)} chunks via PPR/entity index")

        if not ranked_lists:
            return {
                "chunks": [],
                "graph_context": graph_context,
                "source_counts": source_counts,
                "fused_pool_size": 0,
            }

        # ---- Fuse ----
        fused = reciprocal_rank_fusion(ranked_lists)
        pool = fused[: cfg.rerank_pool]

        # ---- Single rerank pass ----
        reranked = self.vector_store.rerank(query, pool)
        final = reranked[: cfg.final_top_k]

        return {
            "chunks": final,
            "graph_context": graph_context,
            "source_counts": source_counts,
            "fused_pool_size": len(pool),
        }

    # ---- helpers ----

    def _graph_chunks(self, seed_entities: List[str]):
        """Pull real chunks for seed entities + PPR-expanded neighbors.
        Also returns a small `graph_context` dict for downstream use."""
        chunks: List[Dict] = []
        seen = set()

        def _add(c: Dict):
            key = _chunk_key(c)
            if key and key not in seen:
                seen.add(key)
                chunks.append(c)

        # 1. Direct entity chunks
        all_neighborhoods = []
        for ent in seed_entities:
            for ch in self.kg.chunks_for_entity(ent):
                _add({**ch, "score": 0.9, "via": f"entity:{ent}"})
            nb = self.kg.get_neighborhood(ent)
            if nb["found"]:
                all_neighborhoods.append(nb)

        # 2. Personalized PageRank over the graph
        ranked = self.kg.personalized_pagerank(seed_entities)
        for node, _score in ranked:
            for ch in self.kg.chunks_for_entity(node):
                _add({**ch, "score": 0.7, "via": f"ppr:{node}"})

        graph_context = {"neighborhoods": all_neighborhoods, "ppr": ranked}
        return chunks, graph_context
