"""
BM25 sparse retrieval over the same chunks indexed in Qdrant.

Lives entirely in-memory (rank_bm25 is pure Python). Persisted to a pickle
beside the kg cache so we don't rebuild on every startup.
"""
from __future__ import annotations

import os
import pickle
import re
from typing import List, Dict, Optional

from rank_bm25 import BM25Okapi
from rich.console import Console

from ingestion.chunker import Chunk

console = Console()

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    # Lowercase alphanumeric tokens. Keeps "article21" together if no space,
    # but legal text almost always has the space, so this is fine.
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class BM25Store:
    def __init__(self, cache_path: str = "data/bm25_cache.pkl"):
        self.cache_path = cache_path
        self.bm25: Optional[BM25Okapi] = None
        self.payloads: List[Dict] = []   # parallel array, mirrors qdrant payload shape
        self.tokens: List[List[str]] = []

    # -------- build / load --------

    def build(self, chunks: List[Chunk]) -> None:
        self.payloads = []
        self.tokens = []
        for c in chunks:
            self.payloads.append({
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "doc_title": c.doc_title,
                "text": c.text,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                **c.metadata,
            })
            self.tokens.append(_tokenize(c.text))
        self.bm25 = BM25Okapi(self.tokens)
        console.print(f"[green]✓ BM25 index built over {len(self.payloads)} chunks[/green]")

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump({"payloads": self.payloads, "tokens": self.tokens}, f)

    def load(self) -> bool:
        if not os.path.exists(self.cache_path):
            return False
        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)
        self.payloads = data["payloads"]
        self.tokens = data["tokens"]
        self.bm25 = BM25Okapi(self.tokens)
        console.print(f"[dim]Loaded BM25 index ({len(self.payloads)} chunks)[/dim]")
        return True

    def build_or_load(self, chunks: List[Chunk]) -> None:
        if self.load():
            # rebuild only if chunk count drifted (cheap check)
            if len(self.payloads) == len(chunks):
                return
            console.print("[yellow]BM25 cache stale (chunk count mismatch) — rebuilding[/yellow]")
        self.build(chunks)
        self.save()

    # -------- search --------

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        if not self.bm25 or not query.strip():
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        # argpartition for speed on larger corpora
        if top_k >= len(scores):
            idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        else:
            import numpy as np
            part = np.argpartition(-scores, top_k)[:top_k]
            idxs = sorted(part.tolist(), key=lambda i: scores[i], reverse=True)

        results = []
        for i in idxs:
            if scores[i] <= 0:
                continue
            results.append({
                "score": float(round(scores[i], 4)),
                **self.payloads[i],
            })
        return results

    def count(self) -> int:
        return len(self.payloads)
