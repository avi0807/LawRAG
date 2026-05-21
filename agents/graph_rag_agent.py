from collections import OrderedDict
from typing import Annotated, Any, Dict, Iterator, List, Optional, TypedDict
import json
import operator
import re

from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from rich.console import Console

from config import cfg
from ingestion.embedder import Embedder
from ingestion.hybrid_retriever import HybridRetriever
from ingestion.knowledge_graph import KnowledgeGraph
from ingestion.vector_store import VectorStore
from agents.grounding import (
    calibrated_confidence,
    ground_sentences,
    grounding_summary,
)
from observability import Trace, log

console = Console()


GENERATOR_SYSTEM = (
    "You are an Indian legal research assistant. Answer using ONLY the provided "
    "evidence. NEVER use outside knowledge. If the evidence does not contain the "
    "answer, say exactly: 'The knowledge base does not contain information about "
    "this topic.' Always cite sources using [N] notation. End your response with: "
    "CONFIDENCE: 0.X"
)


class AgentState(TypedDict):
    query: str
    key_entities: List[str]
    search_strategy: str
    retry_count: int
    retrieved_chunks: Annotated[List[Dict], operator.add]
    graph_context: Dict[str, Any]
    evidence_sufficient: bool
    critic_feedback: str
    answer: str
    citations: List[str]
    confidence_score: float
    confidence_breakdown: Dict[str, float]
    grounding: List[Dict[str, Any]]
    grounding_stats: Dict[str, float]
    last_retrieval: Dict[str, Any]
    trace: Optional[Any]   # observability.Trace; opaque to LangGraph


class _LRU(OrderedDict):
    def __init__(self, maxsize: int):
        super().__init__()
        self.maxsize = maxsize

    def get_or(self, key, default=None):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return default

    def put(self, key, value):
        self[key] = value
        self.move_to_end(key)
        if len(self) > self.maxsize:
            self.popitem(last=False)


class MultiAgentGraphRAG:

    def __init__(
        self,
        vector_store: VectorStore,
        knowledge_graph: KnowledgeGraph,
        embedder: Embedder,
        bm25_store=None,
    ):
        self.vector_store = vector_store
        self.kg = knowledge_graph
        self.embedder = embedder
        self.llm = OpenAI(base_url=cfg.ollama_base_url, api_key=cfg.ollama_api_key)

        # Hybrid retrieval is required. Fall back gracefully if BM25 missing.
        if bm25_store is None:
            from ingestion.bm25_store import BM25Store
            bm25_store = BM25Store()
        self.bm25_store = bm25_store

        self.retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_store=bm25_store,
            knowledge_graph=knowledge_graph,
            embedder=embedder,
        )
        self.graph = self._build_graph()
        self._answer_cache = _LRU(cfg.answer_cache_size)

    # ------------------------------------------------------------------ LLM helpers

    def _llm(self, system: str, user: str, max_tokens: int = 1200) -> str:
        """Non-streaming, no-think completion."""
        response = self.llm.chat.completions.create(
            model=cfg.ollama_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0,
            extra_body={"think": False},
        )
        return response.choices[0].message.content.strip()

    def _llm_stream(self, system: str, user: str, max_tokens: int = 1500) -> Iterator[str]:
        """Yields token deltas from Ollama via the OpenAI-compatible streaming API."""
        stream = self.llm.chat.completions.create(
            model=cfg.ollama_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0,
            stream=True,
            extra_body={"think": False},
        )
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content
            except (AttributeError, IndexError):
                delta = None
            if delta:
                yield delta

    # ------------------------------------------------------------------ Nodes

    def orchestrate(self, state: AgentState) -> AgentState:
        console.print(f"\n[bold cyan]🎯 Orchestrator[/bold cyan] analyzing: '{state['query']}'")

        result_json = self._llm(
            system=(
                "You are a query analysis agent for Indian legal research. "
                "Extract key legal entities and decide retrieval strategy. "
                "Return ONLY valid JSON, no explanation."
            ),
            user=f"""Query: {state['query']}

Return JSON:
{{
  "key_entities": ["entity1", "entity2"],
  "strategy": "hybrid",
  "reasoning": "why this strategy"
}}

Strategy options:
- "vector": factual questions, definitions, "what is X"
- "bm25": queries with case names, article numbers, statute citations
- "graph": relationship questions, "who cited", "which cases", "how does X relate to Y"
- "hybrid": complex multi-hop questions combining both (default — usually best)

For questions containing "which cases", "who cited", "related to", "what cases",
"which judgments" — always use "graph" or "hybrid" and extract the legal entity
being asked about as a key entity.""",
            max_tokens=300,
        )

        data = self._safe_json(result_json) or {}
        entities = [e for e in data.get("key_entities", []) if isinstance(e, str)]
        strategy = data.get("strategy", "hybrid")
        if strategy not in ("vector", "bm25", "graph", "hybrid"):
            strategy = "hybrid"

        console.print(f"  Entities: {entities}")
        console.print(f"  Strategy: {strategy}")

        return {
            **state,
            "key_entities": entities,
            "search_strategy": strategy,
            "retry_count": 0,
            "retrieved_chunks": [],
            "graph_context": {},
            "evidence_sufficient": False,
            "critic_feedback": "",
            "answer": "",
            "citations": [],
            "confidence_score": 0.0,
            "last_retrieval": {},
        }

    def retrieve(self, state: AgentState) -> AgentState:
        console.print(f"\n[bold cyan]🔍 Retriever[/bold cyan] running {state['search_strategy']} search")

        result = self.retriever.retrieve(
            query=state["query"],
            seed_entities=state["key_entities"],
            strategy=state["search_strategy"],
        )

        new_chunks = result["chunks"]
        console.print(f"  Final top-k after rerank: {len(new_chunks)}")

        return {
            **state,
            "retrieved_chunks": state["retrieved_chunks"] + new_chunks,
            "graph_context": result.get("graph_context", {}),
            "last_retrieval": {
                "source_counts": result.get("source_counts", {}),
                "fused_pool_size": result.get("fused_pool_size", 0),
                "final_count": len(new_chunks),
            },
        }

    def critique(self, state: AgentState) -> AgentState:
        console.print(f"\n[bold cyan]🔬 Critic[/bold cyan] evaluating evidence")

        if state["retry_count"] >= cfg.max_retries:
            console.print("  Max retries reached — proceeding")
            return {**state, "evidence_sufficient": True}

        chunks = state["retrieved_chunks"]
        if not chunks:
            return {
                **state,
                "evidence_sufficient": False,
                "critic_feedback": "no chunks retrieved",
                "retry_count": state["retry_count"] + 1,
            }

        # Heuristic fast-path: if rerank top-1 is strong, skip the LLM critic.
        top = chunks[0]
        top_score = float(top.get("rerank_score", top.get("score", 0.0)) or 0.0)

        # Entity grounding check — does at least one top chunk mention any seed entity?
        entity_match = False
        if state["key_entities"]:
            head = " ".join(c.get("text", "")[:300] for c in chunks[:3]).lower()
            entity_match = any(e.lower() in head for e in state["key_entities"] if e)
        else:
            entity_match = True  # no seeds → don't penalize

        if top_score >= cfg.skip_critic_min_top_score and entity_match:
            console.print(f"  [green]✓ Heuristic pass (top score={top_score:.3f}) — skipping LLM critic[/green]")
            return {**state, "evidence_sufficient": True}

        # Otherwise ask the LLM, but cheap (200 tok, no think).
        evidence_text = "\n\n".join(
            f"[{i+1}] (score: {c.get('rerank_score', c.get('score', 0)):.2f}) {c['text'][:400]}"
            for i, c in enumerate(chunks[:6])
        )

        verdict = self._llm(
            system="You are a strict evidence quality evaluator. Be concise.",
            user=f"""Query: {state['query']}

Retrieved evidence:
{evidence_text}

Is this evidence sufficient to answer the query accurately?
Reply with JSON:
{{"sufficient": true/false, "reason": "brief", "missing": "what's missing if not"}}""",
            max_tokens=180,
        )
        data = self._safe_json(verdict) or {}
        sufficient = bool(data.get("sufficient", True))
        missing = data.get("missing", "") or ""

        if sufficient:
            console.print("  [green]✓ Evidence sufficient[/green]")
        else:
            console.print(f"  [yellow]⚠ Insufficient: {missing}[/yellow]")

        return {
            **state,
            "evidence_sufficient": sufficient,
            "critic_feedback": missing,
            "retry_count": state["retry_count"] + 1,
        }

    def retrieve_more(self, state: AgentState) -> AgentState:
        console.print(f"\n[bold yellow]🔄 Retry retrieval[/bold yellow] (attempt {state['retry_count']})")

        expanded = state["query"]
        if state["critic_feedback"]:
            expanded = f"{state['query']} {state['critic_feedback']}"

        result = self.retriever.retrieve(
            query=expanded,
            seed_entities=state["key_entities"],
            strategy="hybrid",
        )
        console.print(f"  Retrieved {len(result['chunks'])} additional chunks")

        return {
            **state,
            "retrieved_chunks": state["retrieved_chunks"] + result["chunks"],
        }

    def generate(self, state: AgentState) -> AgentState:
        console.print(f"\n[bold cyan]✍️  Generator[/bold cyan] synthesizing answer")

        system, user, sources = self._build_generator_prompt(state)
        raw_answer = self._llm(system=system, user=user, max_tokens=1500)

        if cfg.debug:
            console.print(f"  [dim]raw response: {len(raw_answer)} chars[/dim]")
        if len(raw_answer) < 80:
            console.print(f"  [yellow]⚠ short raw output ({len(raw_answer)} chars), retrying with thinking...[/yellow]")
            response = self.llm.chat.completions.create(
                model=cfg.ollama_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=2000,
                temperature=0,
                # No extra_body here — let the model think.
            )
            raw_answer = response.choices[0].message.content.strip()
            if "<think>" in raw_answer:
                raw_answer = raw_answer.split("</think>")[-1].strip()
            if cfg.debug:
                console.print(f"  [dim]retry raw response: {len(raw_answer)} chars[/dim]")

        answer_text, _llm_confidence = self._parse_confidence(raw_answer)

        # ── Sentence-level citation grounding ──
        top_chunks = state["retrieved_chunks"][: cfg.final_top_k]
        grounded = ground_sentences(answer_text, top_chunks)
        gstats = grounding_summary(grounded)

        # ── Calibrated confidence (replaces LLM-claimed) ──
        confidence, breakdown = calibrated_confidence(
            top_chunks,
            state["key_entities"],
            grounded_rate=gstats.get("grounded_rate", 0.0),
        )

        console.print(
            f"  [green]✓ Answer generated[/green]  "
            f"[dim]confidence={confidence:.2f} (top={breakdown['top']:.2f}, "
            f"gap={breakdown['gap']:.2f}, ent={breakdown['entity']:.2f}, "
            f"grnd={breakdown['grounded']:.2f})  "
            f"sents={gstats['sentences']}[/dim]"
        )

        # Warn on poor grounding
        if gstats["sentences"] >= 3 and gstats["grounded_rate"] < 0.5:
            console.print(
                f"  [yellow]⚠ Low grounding rate "
                f"({gstats['grounded_rate']:.0%}) — possible hallucination[/yellow]"
            )

        return {
            **state,
            "answer": answer_text,
            "citations": sources,
            "confidence_score": confidence,
            "confidence_breakdown": breakdown,
            "grounding": grounded,
            "grounding_stats": gstats,
        }

    # ------------------------------------------------------------------ Streaming

    def stream_generate(self, state: AgentState) -> Iterator[Dict[str, Any]]:
        """Yields {'type': 'token', 'text': ...} during streaming and a final
        {'type': 'final', ...} payload with calibrated confidence + grounding."""
        system, user, sources = self._build_generator_prompt(state)

        full = []
        for delta in self._llm_stream(system=system, user=user, max_tokens=1500):
            full.append(delta)
            # Strip thinking tokens defensively if model leaks any.
            if "<think>" in delta or "</think>" in delta:
                continue
            yield {"type": "token", "text": delta}

        raw = "".join(full)
        if "<think>" in raw:
            raw = raw.split("</think>")[-1].strip()
        answer_text, _llm_confidence = self._parse_confidence(raw)

        # Empty output safety net — same retry pattern as non-streaming generate
        if len(answer_text) < 40:
            response = self.llm.chat.completions.create(
                model=cfg.ollama_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=2000,
                temperature=0,
            )
            retry_raw = response.choices[0].message.content.strip()
            if "<think>" in retry_raw:
                retry_raw = retry_raw.split("</think>")[-1].strip()
            answer_text, _ = self._parse_confidence(retry_raw)
            # Push the retry text as a single token event so the UI catches up.
            if answer_text:
                yield {"type": "token", "text": "\n" + answer_text}

        top_chunks = state["retrieved_chunks"][: cfg.final_top_k]
        grounded = ground_sentences(answer_text, top_chunks)
        gstats = grounding_summary(grounded)
        confidence, breakdown = calibrated_confidence(
            top_chunks,
            state["key_entities"],
            grounded_rate=gstats.get("grounded_rate", 0.0),
        )

        yield {
            "type": "final",
            "answer": answer_text,
            "citations": sources,
            "confidence": confidence,
            "confidence_breakdown": breakdown,
            "grounding": grounded,
            "grounding_stats": gstats,
        }

    def _build_generator_prompt(self, state: AgentState):
        top_chunks = state["retrieved_chunks"][: cfg.final_top_k]
        sources: List[str] = []
        evidence_blocks: List[str] = []
        for chunk in top_chunks:
            title = chunk.get("doc_title", "Unknown")
            if title not in sources:
                sources.append(title)
            src_idx = sources.index(title) + 1
            evidence_blocks.append(f"[{src_idx}] {chunk['text'][:600]}")

        evidence_text = "\n\n".join(evidence_blocks)
        sources_block = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(sources))

        user = (
            f"Question: {state['query']}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            f"Sources:\n{sources_block}\n\n"
            "Provide a comprehensive, accurate answer with citations."
        )
        return GENERATOR_SYSTEM, user, sources

    @staticmethod
    def _parse_confidence(raw: str):
        """Pull a CONFIDENCE: 0.X tail off the answer if present.
        Robust to the model emitting it at the start, missing it entirely,
        or putting garbage after it."""
        confidence = 0.7
        text = (raw or "").strip()
        if not text:
            return raw, confidence

        if "CONFIDENCE:" in text:
            parts = text.rsplit("CONFIDENCE:", 1)
            body = parts[0].strip()
            tail = parts[1].strip()[:4]
            try:
                confidence = float(tail.split()[0])
            except (ValueError, IndexError):
                pass
            # If stripping the marker leaves the body empty (model put
            # CONFIDENCE at the very start), fall back to the raw text
            # minus the marker line so the user still sees something.
            if body:
                text = body
            else:
                # Drop just the "CONFIDENCE: 0.X" line, keep the rest.
                text = re.sub(r"CONFIDENCE:\s*[0-9.]+\s*", "", raw, count=1).strip()

        return text or raw.strip(), confidence

    @staticmethod
    def _safe_json(raw: str) -> Optional[Dict]:
        if not raw:
            return None
        clean = raw.strip().strip("`")
        if "<think>" in clean:
            clean = clean.split("</think>")[-1].strip()
        if clean.startswith("json"):
            clean = clean[4:].strip()
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start == -1 or end <= 0:
            return None
        try:
            return json.loads(clean[start:end])
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------ Routing

    def route_after_critic(self, state: AgentState) -> str:
        return "generate" if state["evidence_sufficient"] else "retrieve_more"

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("orchestrate", self.orchestrate)
        builder.add_node("retrieve", self.retrieve)
        builder.add_node("critique", self.critique)
        builder.add_node("retrieve_more", self.retrieve_more)
        builder.add_node("generate", self.generate)

        builder.add_edge(START, "orchestrate")
        builder.add_edge("orchestrate", "retrieve")
        builder.add_edge("retrieve", "critique")
        builder.add_conditional_edges(
            "critique",
            self.route_after_critic,
            {"generate": "generate", "retrieve_more": "retrieve_more"},
        )
        builder.add_edge("retrieve_more", "critique")
        builder.add_edge("generate", END)
        return builder.compile()

    # ------------------------------------------------------------------ Public API

    def run(self, query: str, session_id: str = "default") -> AgentState:
        console.print(f"\n{'='*60}\n[bold]Query:[/bold] {query}\n{'='*60}")

        cached = self._answer_cache.get_or(query.strip())
        if cached:
            console.print("[dim]↺ answer cache hit[/dim]")
            return cached

        trace = Trace(query=query, session_id=session_id)
        log.info("query.start", extra={"trace_id": trace.trace_id, "query": query[:200]})

        initial: AgentState = {
            "query": query,
            "key_entities": [],
            "search_strategy": "hybrid",
            "retry_count": 0,
            "retrieved_chunks": [],
            "graph_context": {},
            "evidence_sufficient": False,
            "critic_feedback": "",
            "answer": "",
            "citations": [],
            "confidence_score": 0.0,
            "confidence_breakdown": {},
            "grounding": [],
            "grounding_stats": {},
            "last_retrieval": {},
            "trace": trace,
        }
        try:
            final_state = self.graph.invoke(initial)
        except Exception as e:
            log.exception("query.failed", extra={"trace_id": trace.trace_id})
            trace.final = {"error": type(e).__name__, "message": str(e)[:300]}
            trace.flush()
            raise

        trace.final = {
            "confidence": final_state.get("confidence_score"),
            "confidence_breakdown": final_state.get("confidence_breakdown", {}),
            "grounding_stats": final_state.get("grounding_stats", {}),
            "strategy": final_state.get("search_strategy"),
            "retry_count": final_state.get("retry_count"),
            "n_citations": len(final_state.get("citations", [])),
            "answer_chars": len(final_state.get("answer", "")),
        }
        trace.flush()
        log.info("query.done", extra={
            "trace_id": trace.trace_id,
            "total_ms": trace.total_ms(),
            **trace.final,
        })

        self._answer_cache.put(query.strip(), final_state)
        return final_state

    def run_stream(self, query: str, session_id: str = "default") -> Iterator[Dict[str, Any]]:
        """Run orchestrate → retrieve → critique (with retries) synchronously,
        then stream tokens during generation. Yields rich status events with
        payloads so a UI can show real progress (entities, hit counts, verdict).
        """
        console.print(f"\n{'='*60}\n[bold]Query (stream):[/bold] {query}\n{'='*60}")
        trace = Trace(query=query, session_id=session_id)
        log.info("query.start", extra={"trace_id": trace.trace_id, "query": query[:200], "stream": True})

        state: AgentState = {
            "query": query,
            "key_entities": [],
            "search_strategy": "hybrid",
            "retry_count": 0,
            "retrieved_chunks": [],
            "graph_context": {},
            "evidence_sufficient": False,
            "critic_feedback": "",
            "answer": "",
            "citations": [],
            "confidence_score": 0.0,
            "confidence_breakdown": {},
            "grounding": [],
            "grounding_stats": {},
            "last_retrieval": {},
            "trace": trace,
        }

        try:
            # ---- orchestrate ----
            yield {"type": "status", "stage": "orchestrate"}
            with trace.stage("orchestrate"):
                state = self.orchestrate(state)
            yield {
                "type": "status",
                "stage": "orchestrate_done",
                "entities": state["key_entities"],
                "strategy": state["search_strategy"],
            }

            # ---- retrieve ----
            yield {"type": "status", "stage": "retrieve", "strategy": state["search_strategy"]}
            with trace.stage("retrieve") as rec:
                state = self.retrieve(state)
                rec.payload.update(state["last_retrieval"])
            yield {
                "type": "status",
                "stage": "retrieve_done",
                "source_counts": state["last_retrieval"].get("source_counts", {}),
                "fused_pool_size": state["last_retrieval"].get("fused_pool_size", 0),
                "final_count": state["last_retrieval"].get("final_count", 0),
            }

            # ---- critic loop ----
            while True:
                yield {"type": "status", "stage": "critique"}
                prev_retry = state["retry_count"]
                with trace.stage("critique"):
                    state = self.critique(state)
                llm_ran = state["retry_count"] > prev_retry
                yield {
                    "type": "status",
                    "stage": "critique_done",
                    "sufficient": state["evidence_sufficient"],
                    "feedback": state["critic_feedback"],
                    "skipped_llm": (state["evidence_sufficient"] and not llm_ran),
                }
                if state["evidence_sufficient"] or state["retry_count"] >= cfg.max_retries:
                    break
                yield {"type": "status", "stage": "retrieve_more"}
                with trace.stage("retrieve_more"):
                    state = self.retrieve_more(state)
                yield {
                    "type": "status",
                    "stage": "retrieve_more_done",
                    "added": cfg.final_top_k,
                }

            # ---- generate ----
            yield {"type": "status", "stage": "generate"}
            with trace.stage("generate") as gen_rec:
                for event in self.stream_generate(state):
                    if event["type"] == "final":
                        event["strategy"] = state["search_strategy"]
                        event["retry_count"] = state["retry_count"]
                        gen_rec.payload.update({
                            "confidence": event.get("confidence"),
                            "grounding_stats": event.get("grounding_stats", {}),
                            "answer_chars": len(event.get("answer", "")),
                        })
                        trace.final = {
                            "confidence": event.get("confidence"),
                            "confidence_breakdown": event.get("confidence_breakdown", {}),
                            "grounding_stats": event.get("grounding_stats", {}),
                            "strategy": state["search_strategy"],
                            "retry_count": state["retry_count"],
                        }
                    yield event
        except Exception as e:
            log.exception("query.failed", extra={"trace_id": trace.trace_id})
            trace.final = {"error": type(e).__name__, "message": str(e)[:300]}
            yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
        finally:
            trace.flush()
            log.info("query.done", extra={
                "trace_id": trace.trace_id,
                "total_ms": trace.total_ms(),
                **{k: v for k, v in trace.final.items() if k != "confidence_breakdown"},
            })
