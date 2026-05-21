"""
Golden-set evaluation harness for LawRAG.

Run:  python -m evaluation.run_eval

Loads evaluation/golden_set.json, runs each query through the agent, scores it
on three dimensions (entity recall, source recall, keyword presence) plus the
existing RAGAS heuristics, and writes a results JSONL + a summary table.

Designed to be run before/after any retrieval or prompt change so you can see
exactly what got better and what got worse — the discipline that converts
"impressive demo" into "engineer who knows production".
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, List

from rich.console import Console
from rich.table import Table

from agents.graph_rag_agent import MultiAgentGraphRAG
from data.ik_loader import SAMPLE_DOCUMENTS
from evaluation.ragas_eval import SimpleRAGASEvaluator
from ingestion.bm25_store import BM25Store
from ingestion.chunker import RecursiveChunker
from ingestion.embedder import Embedder
from ingestion.knowledge_graph import KnowledgeGraph
from ingestion.vector_store import VectorStore

KG_CACHE = "data/kg_cache.pkl"
BM25_CACHE = "data/bm25_cache.pkl"
GOLDEN_SET = "evaluation/golden_set.json"
RESULTS_DIR = "evaluation/results"

console = Console()


# ──────────────────────────────────────────────────────────
# Pipeline bootstrap (mirrors main.py but quieter)
# ──────────────────────────────────────────────────────────

def build_agent() -> MultiAgentGraphRAG:
    chunker = RecursiveChunker()
    embedder = Embedder()
    vs = VectorStore()
    kg = KnowledgeGraph()
    bm25 = BM25Store(cache_path=BM25_CACHE)

    all_chunks = []
    for doc in SAMPLE_DOCUMENTS:
        chunks = chunker.chunk_document(
            doc_id=doc["id"],
            title=doc["title"],
            text=doc["content"],
            metadata={"source": doc["source"], "year": doc.get("year", 2024)},
        )
        all_chunks.extend(chunks)

    if vs.count() == 0:
        chunks_with, embs = embedder.embed_chunks(all_chunks)
        vs.upsert_chunks(chunks_with, embs)

    bm25.build_or_load(all_chunks)

    if not kg.load(KG_CACHE):
        console.print("[yellow]No KG cache — running full extraction. This is slow.[/yellow]")
        kg.extract_and_add(all_chunks)
        kg.save(KG_CACHE)
    elif not kg.entity_chunks:
        for ch in all_chunks:
            kg._index_chunk_for_entities(ch, [])

    return MultiAgentGraphRAG(
        vector_store=vs,
        knowledge_graph=kg,
        embedder=embedder,
        bm25_store=bm25,
    )


# ──────────────────────────────────────────────────────────
# Per-case scoring
# ──────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    return (s or "").lower()


def score_case(case: Dict, final_state: Dict) -> Dict:
    """Compute case-level scores: entity_recall, source_recall, keyword_presence."""
    answer_lower = _normalize(final_state.get("answer", ""))
    sources = final_state.get("citations", []) or []
    sources_blob = " | ".join(s for s in sources)
    sources_lower = _normalize(sources_blob)

    expected_entities = case.get("expected_entities", []) or []
    expected_sources = case.get("expected_sources", []) or []
    expected_keywords = case.get("expected_keywords", []) or []

    # Entity recall: how many expected entities appear in answer + sources
    if expected_entities:
        hits = sum(
            1 for e in expected_entities
            if _normalize(e) in answer_lower or _normalize(e) in sources_lower
        )
        entity_recall = hits / len(expected_entities)
    else:
        entity_recall = 1.0

    # Source recall: substring match against citation list (any chunk title)
    if expected_sources:
        hits = sum(
            1 for s in expected_sources
            if _normalize(s) in sources_lower
        )
        source_recall = hits / len(expected_sources)
    else:
        source_recall = 1.0

    # Keyword presence in the answer
    if expected_keywords:
        hits = sum(1 for k in expected_keywords if _normalize(k) in answer_lower)
        keyword_presence = hits / len(expected_keywords)
    else:
        keyword_presence = 1.0

    # Negative cases (out-of-scope): reward refusal
    is_negative = case.get("is_negative", False)
    if is_negative:
        refused = "does not contain" in answer_lower
        return {
            "entity_recall": 1.0 if refused else 0.0,
            "source_recall": 1.0 if not sources else 0.5,  # ideally no sources
            "keyword_presence": 1.0 if refused else 0.0,
            "refused": refused,
        }

    return {
        "entity_recall": round(entity_recall, 3),
        "source_recall": round(source_recall, 3),
        "keyword_presence": round(keyword_presence, 3),
    }


# ──────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────

def run_eval(golden_path: str = GOLDEN_SET, output_dir: str = RESULTS_DIR) -> Dict:
    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)
    cases = golden["cases"]

    console.print(f"\n[bold]Loaded {len(cases)} cases from {golden_path}[/bold]")

    agent = build_agent()
    ragas = SimpleRAGASEvaluator()

    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(output_dir, f"eval-{stamp}.jsonl")

    rows = []
    with open(out_path, "w", encoding="utf-8") as out:
        for i, case in enumerate(cases, 1):
            console.print(f"\n[cyan]── [{i}/{len(cases)}] {case['id']}[/cyan]")
            t0 = time.time()
            try:
                final_state = agent.run(case["question"], session_id="eval")
                error = None
            except Exception as e:
                console.print(f"[red]✗ {case['id']} crashed: {e}[/red]")
                final_state = {"answer": "", "citations": [], "retrieved_chunks": []}
                error = f"{type(e).__name__}: {e}"
            elapsed_ms = round((time.time() - t0) * 1000, 1)

            case_scores = score_case(case, final_state)

            ragas_scores = ragas.evaluate_sample(
                question=case["question"],
                answer=final_state.get("answer", ""),
                context_chunks=[c.get("text", "") for c in final_state.get("retrieved_chunks", [])[:5]],
            )

            row = {
                "id": case["id"],
                "question": case["question"],
                "answer_chars": len(final_state.get("answer", "")),
                "confidence": final_state.get("confidence_score", 0.0),
                "confidence_breakdown": final_state.get("confidence_breakdown", {}),
                "grounding_stats": final_state.get("grounding_stats", {}),
                "elapsed_ms": elapsed_ms,
                "error": error,
                "case_scores": case_scores,
                "ragas": ragas_scores,
            }
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(rows)

    # Print table
    table = Table(title=f"Eval Summary ({len(rows)} cases)", show_lines=False)
    table.add_column("Metric")
    table.add_column("Mean", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Min", justify="right")
    for metric, stats in summary["per_metric"].items():
        table.add_row(
            metric,
            f"{stats['mean']:.3f}",
            f"{stats['median']:.3f}",
            f"{stats['min']:.3f}",
        )
    console.print(table)

    console.print(
        f"\n[green]Total wall-clock: {summary['total_seconds']:.1f}s · "
        f"Avg latency: {summary['avg_ms']:.0f}ms · "
        f"Failed: {summary['failed']}[/green]"
    )
    console.print(f"[dim]Per-case results: {out_path}[/dim]")

    summary_path = os.path.join(output_dir, f"summary-{stamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    console.print(f"[dim]Summary: {summary_path}[/dim]")

    return summary


def summarize(rows: List[Dict]) -> Dict:
    import statistics as st

    metric_keys_all = [
        ("entity_recall", lambda r: r["case_scores"]["entity_recall"]),
        ("source_recall", lambda r: r["case_scores"]["source_recall"]),
        ("keyword_presence", lambda r: r["case_scores"]["keyword_presence"]),
        ("confidence", lambda r: r["confidence"]),
        ("grounded_rate", lambda r: r["grounding_stats"].get("grounded_rate", 0.0)),
    ]
    # RAGAS heuristics are word-overlap-based; meaningless on negative
    # (refusal) cases. Compute them on positive cases only.
    metric_keys_positive_only = [
        ("ragas_faithfulness", lambda r: r["ragas"]["faithfulness"]),
        ("ragas_relevancy", lambda r: r["ragas"]["answer_relevancy"]),
        ("ragas_precision", lambda r: r["ragas"]["context_precision"]),
        ("ragas_overall", lambda r: r["ragas"]["overall"]),
    ]

    positive_rows = [
        r for r in rows
        if r.get("error") is None and not r["case_scores"].get("refused", False)
    ]
    healthy_rows = [r for r in rows if r.get("error") is None]

    per_metric = {}
    for name, fn in metric_keys_all:
        vals = [fn(r) for r in healthy_rows]
        if not vals:
            per_metric[name] = {"mean": 0.0, "median": 0.0, "min": 0.0}
            continue
        per_metric[name] = {
            "mean": round(st.mean(vals), 3),
            "median": round(st.median(vals), 3),
            "min": round(min(vals), 3),
        }
    for name, fn in metric_keys_positive_only:
        vals = [fn(r) for r in positive_rows]
        if not vals:
            per_metric[name] = {"mean": 0.0, "median": 0.0, "min": 0.0}
            continue
        per_metric[name] = {
            "mean": round(st.mean(vals), 3),
            "median": round(st.median(vals), 3),
            "min": round(min(vals), 3),
        }

    return {
        "n_cases": len(rows),
        "n_positive": len(positive_rows),
        "failed": sum(1 for r in rows if r.get("error")),
        "total_seconds": round(sum(r["elapsed_ms"] for r in rows) / 1000, 1),
        "avg_ms": round(sum(r["elapsed_ms"] for r in rows) / max(1, len(rows)), 0),
        "per_metric": per_metric,
    }


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LawRAG golden-set evaluation.")
    parser.add_argument("--golden", default=GOLDEN_SET, help="Path to golden set JSON")
    parser.add_argument("--out", default=RESULTS_DIR, help="Output directory")
    args = parser.parse_args()
    run_eval(args.golden, args.out)
