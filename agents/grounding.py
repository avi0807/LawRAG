"""
Calibrated confidence and sentence-level citation grounding.

Both are lightweight and deterministic — no extra LLM calls. The point is
to give downstream consumers a real, defensible quality signal that doesn't
depend on the model's self-report.

Confidence components:
  1. Normalized rerank top-1 score      — how good was the best chunk?
  2. Score gap (top-1 − top-5)          — was the best chunk clearly best?
  3. Entity grounding rate              — do query entities appear in top chunks?

Citation grounding:
  Splits the answer into sentences; for each sentence, finds which retrieved
  chunks have enough lexical overlap to plausibly support it. Returns
  per-sentence chunk indices.

Neither is bulletproof. They're meaningful signals, not certificates.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from config import cfg


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]{2,}")
# Split on . ! ? followed by whitespace, but tolerate citations like "[1]" mid-sentence.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")

_STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "had",
    "her", "was", "one", "our", "out", "day", "get", "has", "him", "his", "how",
    "man", "new", "now", "old", "see", "two", "way", "who", "boy", "did", "its",
    "let", "put", "say", "she", "too", "use", "this", "that", "with", "from",
    "have", "they", "their", "would", "could", "should", "their", "there",
    "which", "what", "when", "where", "while", "shall", "such", "into", "must",
    "been", "were", "than", "then", "also", "upon", "under",
}


def _tokens(text: str) -> set[str]:
    return {
        t.lower()
        for t in _WORD_RE.findall(text or "")
        if t.lower() not in _STOPWORDS
    }


# ──────────────────────────────────────────────────────────
# Confidence
# ──────────────────────────────────────────────────────────

def _normalize_score(s: float) -> float:
    """Linear clamp into [0, 1]. FlashRank cross-encoder scores already sit
    in that range for our setup (we saw 0.001 → 0.999 in the wild), so a
    sigmoid would just compress useful spread into the middle.

    Negative scores indicate "not relevant"; we floor at 0.
    Scores > 1 (extremely rare) clamp at 1.
    """
    if s is None:
        return 0.0
    try:
        return max(0.0, min(1.0, float(s)))
    except (TypeError, ValueError):
        return 0.0


def _entity_grounding_rate(entities: List[str], chunks: List[Dict]) -> float:
    if not entities or not chunks:
        return 1.0  # nothing to ground = neutral
    head_text = " ".join((c.get("text") or "")[:600] for c in chunks[:5]).lower()
    if not head_text:
        return 0.0
    hits = sum(1 for e in entities if e and e.lower() in head_text)
    return hits / max(1, len(entities))


def calibrated_confidence(
    chunks: List[Dict],
    entities: List[str],
    grounded_rate: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Returns (confidence in [0,1], breakdown dict for logging).

    `grounded_rate` is from `grounding_summary(...)["grounded_rate"]` — the
    fraction of answer sentences with at least one supporting chunk. Defaults
    to 1.0 (neutral) so this can still be called before grounding is computed.
    """
    if not chunks:
        return 0.0, {"top": 0.0, "gap": 0.0, "entity": 0.0, "grounded": 0.0}

    def _s(c: Dict) -> float:
        # Prefer rerank score; fall back to whatever score is present.
        v = c.get("rerank_score")
        if v is None:
            v = c.get("score", 0.0)
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    scores = [_s(c) for c in chunks[:5]]
    top = scores[0] if scores else 0.0
    fifth = scores[-1] if len(scores) >= 5 else (scores[-1] if scores else 0.0)

    top_norm = _normalize_score(top)
    gap = max(0.0, _normalize_score(top) - _normalize_score(fifth))
    entity_rate = _entity_grounding_rate(entities, chunks)
    g_rate = max(0.0, min(1.0, float(grounded_rate)))

    score = (
        cfg.confidence_w_top * top_norm
        + cfg.confidence_w_gap * gap
        + cfg.confidence_w_entity * entity_rate
        + cfg.confidence_w_grounded * g_rate
    )
    score = max(0.0, min(1.0, score))

    breakdown = {
        "top": round(top_norm, 3),
        "gap": round(gap, 3),
        "entity": round(entity_rate, 3),
        "grounded": round(g_rate, 3),
        "raw_top": round(top, 3),
    }
    return round(score, 3), breakdown


# ──────────────────────────────────────────────────────────
# Citation grounding
# ──────────────────────────────────────────────────────────

def split_sentences(answer: str) -> List[str]:
    """Lightweight sentence splitter. Good enough for grounding."""
    if not answer:
        return []
    # Drop trailing CONFIDENCE marker if it leaked through
    answer = re.sub(r"CONFIDENCE:\s*[0-9.]+\s*$", "", answer).strip()
    parts = _SENT_SPLIT_RE.split(answer)
    return [p.strip() for p in parts if len(p.strip()) > 12]


def ground_sentences(
    answer: str,
    chunks: List[Dict],
) -> List[Dict]:
    """
    For each sentence in `answer`, return:
        {"sentence": str,
         "supports": [{"chunk_idx": int, "doc_title": str, "overlap": float}, ...]}

    `supports` is sorted by overlap descending, capped at cfg.citation_top_n.
    A sentence with zero supports is a hallucination warning sign.
    """
    sentences = split_sentences(answer)
    if not sentences or not chunks:
        return [{"sentence": s, "supports": []} for s in sentences]

    chunk_tokens = [_tokens(c.get("text", "")) for c in chunks]

    out: List[Dict] = []
    for s in sentences:
        s_tokens = _tokens(s)
        if not s_tokens:
            out.append({"sentence": s, "supports": []})
            continue
        scored = []
        for i, ct in enumerate(chunk_tokens):
            if not ct:
                continue
            overlap = len(s_tokens & ct) / max(1, len(s_tokens))
            if overlap >= cfg.citation_min_overlap:
                scored.append((i, overlap))
        scored.sort(key=lambda x: -x[1])
        supports = [
            {
                "chunk_idx": i,
                "doc_title": chunks[i].get("doc_title", "Unknown"),
                "overlap": round(o, 3),
            }
            for i, o in scored[: cfg.citation_top_n]
        ]
        out.append({"sentence": s, "supports": supports})
    return out


def grounding_summary(grounded: List[Dict]) -> Dict[str, float]:
    """Aggregate stats on how well the answer is grounded."""
    if not grounded:
        return {"sentences": 0, "grounded_rate": 0.0, "avg_supports": 0.0}
    total = len(grounded)
    grounded_n = sum(1 for g in grounded if g["supports"])
    avg_supports = sum(len(g["supports"]) for g in grounded) / total
    return {
        "sentences": total,
        "grounded_rate": round(grounded_n / total, 3),
        "avg_supports": round(avg_supports, 2),
    }
