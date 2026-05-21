# LawRAG

A multi-agent retrieval-augmented generation system over Indian Supreme Court judgments and the Constitution of India. Hybrid retrieval (BM25 + dense + knowledge graph), LangGraph orchestration with self-critique, calibrated confidence, sentence-level citation grounding, and real token streaming. Runs locally on Ollama.

---

## Locked baseline (12-query golden set, May 2026)

| Metric | Mean | Median | Min |
|---|---:|---:|---:|
| RAGAS overall | **0.926** | 1.000 | 0.333 |
| RAGAS faithfulness | 0.904 | 1.000 | 0.000 |
| RAGAS answer relevancy | 0.875 | 1.000 | 0.000 |
| RAGAS context precision | **1.000** | 1.000 | 1.000 |
| Entity recall | 0.917 | 1.000 | 0.000 |
| Source recall | 0.875 | 1.000 | 0.000 |
| Sentence grounding rate | 0.821 | 1.000 | 0.000 |
| Calibrated confidence | 0.703 | 0.804 | 0.150 |

**Latency:** ~55s per query end-to-end on a local Qwen3-8B-Q4 model with a 6 GB RTX 3050. Streaming first-token latency: ~1–2s.

**Confidence calibration is real:** bad answers (failed retrieval, out-of-scope queries) cluster at confidence ≤ 0.15. Good answers cluster ≥ 0.85. The system can flag its own failures.

Frozen baseline: `evaluation/results/baseline.json` and `evaluation/results/baseline.jsonl`.

Reproduce: `python -m evaluation.run_eval`.

---

## Architecture

```
Documents (28 SC judgments + Constitution)
   │
   ├─▶ Recursive chunker (512 tok, 50 overlap)
   │       │
   │       ├─▶ nomic-embed-text   ──▶ Qdrant (HNSW, cosine)
   │       ├─▶ rank_bm25 index    ──▶ BM25Okapi (in-memory)
   │       └─▶ Qwen3 NER + regex  ──▶ NetworkX KG + entity→chunk index
   │
   └─▶ Query
           │
           ▼
   ┌────────────────────────────────────────────────┐
   │  LangGraph state machine                       │
   │                                                │
   │  Orchestrate ─▶ Retrieve ─▶ Critique ─┐        │
   │       (entity extract,     │  ▲       │        │
   │        strategy)           │  │       │        │
   │                            │  └─ Retrieve more │
   │                            │     (if insufficient)
   │                            ▼                   │
   │              Generate ──▶ Calibrated confidence│
   │                            + citation grounding│
   └────────────────────────────────────────────────┘
                  │
                  ▼
          Answer + sources + grounded sentences
```

**Retrieval pipeline:** BM25 top-20 + Dense top-20 + Graph chunks via PPR/entity-index → Reciprocal Rank Fusion (k=60) → top 30 → FlashRank cross-encoder rerank → top 8 to generator.

**Critic loop:** heuristic fast-path skips the LLM critic when rerank top-1 ≥ 0.40 *and* a query entity appears in the top-3 chunks. LLM critic only runs when retrieval is borderline. Max 2 retry rounds.

**Streaming:** real Ollama token streaming all the way to the browser via SSE, with rich per-stage progress events (entity extraction, hit counts per retriever, critic verdict).

---

## Stack

| Layer | Tool |
|---|---|
| LLM | `qwen3:8b-q4_K_M` via Ollama |
| Embeddings | `nomic-embed-text` (768-dim) |
| Dense retrieval | Qdrant (local file mode) |
| Sparse retrieval | rank-bm25 |
| Knowledge graph | NetworkX `DiGraph` |
| Reranker | FlashRank (TinyBERT) |
| Agent orchestration | LangGraph |
| API | FastAPI + SSE streaming |
| Frontend | Vanilla HTML/CSS/JS, light theme |
| Eval | Custom golden-set harness + RAGAS-style heuristics |

---

## Setup

```bash
git clone <this repo>
cd LawRAG
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Ollama models
ollama pull qwen3:8b-q4_K_M
ollama pull nomic-embed-text

# Optional Indian Kanoon API token if you want to refresh the corpus
# echo "INDIANKANOON_API_TOKEN=..." > .env
```

Start Ollama (`ollama serve`), then:

```bash
# CLI: ingest + 3 sample queries + RAGAS
python main.py

# OR: REST API
uvicorn api:app --reload --port 8000
# then open ui.html in a browser
```

First run rebuilds the knowledge graph (~10–25 minutes on a 6 GB GPU). Subsequent runs load it from `data/kg_cache.pkl` instantly.

---

## Reproducing the baseline

```bash
python -m evaluation.run_eval
```

Outputs `evaluation/results/eval-{timestamp}.jsonl` (per-case) and `summary-{timestamp}.json` (aggregates). Compare against `evaluation/results/baseline.json` to see deltas.

---

## API

| Endpoint | Method | Notes |
|---|---|---|
| `/health` | GET | status, vector count, BM25 count, KG stats |
| `/query` | POST | Body `{question, session_id, stream}`. SSE if `stream=true`. |
| `/conversation/{session_id}` | DELETE | Clear in-memory chat history |
| `/stats` | GET | Aggregate system stats |

**Auth:** set `LAWRAG_API_TOKEN=<secret>` in env to require `Authorization: Bearer <secret>` on every request. Empty token = dev mode (no auth).

**Rate limit:** sliding-window 30 req/min per `session_id`. Returns 429 when exceeded.

**Question length:** capped at 2000 chars. Returns 413 if exceeded.

---

## Observability

Every query writes one structured trace line to `data/logs/traces-YYYY-MM-DD.jsonl` with stage timings, source counts, confidence breakdown, and grounding stats:

```bash
tail -1 data/logs/traces-$(date -u +%Y-%m-%d).jsonl | jq
```

```json
{
  "trace_id": "fed8f8c427d7",
  "query": "...",
  "total_ms": 51589.99,
  "stages": [...],
  "final": {
    "confidence": 0.997,
    "confidence_breakdown": {"top": 0.998, "gap": 0.986, "entity": 1.0, "grounded": 1.0},
    "grounding_stats": {"sentences": 9, "grounded_rate": 1.0, "avg_supports": 2.78}
  }
}
```

---

## Repo layout

```
LawRAG/
├── api.py                     FastAPI + SSE + auth + rate limit
├── main.py                    CLI: ingest + queries + RAGAS
├── ui.html                    Vanilla frontend with progress streaming
├── config.py                  All tunables
├── observability.py           JSON logger + per-query Trace
├── requirements.txt
│
├── agents/
│   ├── graph_rag_agent.py     LangGraph state machine
│   └── grounding.py           Calibrated confidence + sentence grounding
│
├── ingestion/
│   ├── chunker.py             Recursive sentence-aware chunker
│   ├── embedder.py            Ollama embeddings + LRU cache
│   ├── vector_store.py        Qdrant wrapper
│   ├── bm25_store.py          rank-bm25 wrapper with pickle cache
│   ├── knowledge_graph.py     NetworkX KG + PPR + entity-chunk index
│   └── hybrid_retriever.py    BM25 + dense + graph fused via RRF
│
├── data/
│   ├── ik_loader.py           Indian Kanoon API + Constitution PDF loader
│   ├── constitution.pdf
│   ├── ik_cache/              Cached judgments
│   ├── qdrant_storage/        Vectors on disk
│   ├── kg_cache.pkl           Generated
│   ├── bm25_cache.pkl         Generated
│   └── logs/                  Per-query traces (JSONL)
│
└── evaluation/
    ├── golden_set.json        12 cases incl. one negative (out-of-scope)
    ├── run_eval.py            Harness; writes per-case + summary
    ├── ragas_eval.py          Lightweight RAGAS heuristics
    └── results/
        ├── baseline.json      Frozen baseline summary
        └── baseline.jsonl     Frozen per-case results
```

---

## Configuration

`config.py` holds all tunables. Most-edited:

| Knob | Default | Effect |
|---|---|---|
| `dense_top_k` | 20 | Dense recall pool |
| `bm25_top_k` | 20 | BM25 recall pool |
| `rrf_k` | 60 | RRF damping |
| `rerank_pool` | 30 | Candidates passed to FlashRank |
| `final_top_k` | 8 | Chunks given to generator |
| `skip_critic_min_top_score` | 0.40 | Heuristic threshold to skip LLM critic |
| `confidence_w_top` | 0.35 | Confidence: rerank top-1 weight |
| `confidence_w_gap` | 0.15 | Confidence: top–p5 score gap weight |
| `confidence_w_entity` | 0.15 | Confidence: query entity grounding weight |
| `confidence_w_grounded` | 0.35 | Confidence: sentence-grounding weight |
| `citation_min_overlap` | 0.18 | Word-overlap threshold for "supports" link |
| `rate_limit_per_minute` | 30 | Per-session API rate limit |

---

## What's measured, what's signal vs noise

**Real signals:**
- *RAGAS context precision = 1.000* — retrieval consistently returns relevant chunks
- *Calibrated confidence < 0.20 on the two failure cases* — system flags its own bad outputs
- *Grounding rate 1.000 on 9 of 12 cases* — answers trace back to evidence

**Honest caveats:**
- RAGAS here is a lexical-overlap heuristic, not the real LLM-as-judge RAGAS library. Treat the absolute numbers as approximate; deltas between runs are the trustworthy part.
- Golden set is 12 queries. Sufficient for regression testing, insufficient for confident generalization claims. Expand to 50–200 for production claims.
- Latency is dominated by the local 8B model. Cloud APIs would cut end-to-end latency to ~3–5s.

---

## Known limitations

- **No temporal/overruling awareness.** A 1976 holding may have been overruled in 1980; the system doesn't know.
- **Reranker calibration on abstract queries.** FlashRank's TinyBERT scores ~0.001 on queries like "what is X doctrine". Switching to `bge-reranker-large` would likely fix this.
- **Constitution PDF noise.** Article splitting regex produces some garbage titles (`Article 368 — ]3[226A`).
- **No incremental ingestion.** Adding documents requires a full pipeline rebuild.
- **In-memory state.** BM25 index, KG, and conversation store live in one process. Restart loses chat history.

These are explicitly *not bugs*; they're scope choices. See `RAG_STUDY_GUIDE.md` for what each would take to address.

---

## Studying / interview prep

A self-contained guide covering the full theory and skills behind this project (RAG, vector DBs, BM25, RRF, knowledge graphs, LangChain/LangGraph, streaming, evaluation): **`RAG_STUDY_GUIDE.md`**.
