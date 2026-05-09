from typing import TypedDict, List, Dict, Any, Annotated
import operator
import json
from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from rich.console import Console

from config import cfg
from ingestion.embedder import Embedder
from ingestion.vector_store import VectorStore
from ingestion.knowledge_graph import KnowledgeGraph

console = Console()


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


class MultiAgentGraphRAG:

    def __init__(self, vector_store: VectorStore, knowledge_graph: KnowledgeGraph, embedder: Embedder):
        self.vector_store = vector_store
        self.kg = knowledge_graph
        self.embedder = embedder
        self.llm = OpenAI(base_url=cfg.ollama_base_url, api_key=cfg.ollama_api_key)
        self.graph = self._build_graph()

    def _llm(self, system: str, user: str, max_tokens: int = 1500) -> str:
        response = self.llm.chat.completions.create(
            model=cfg.ollama_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    def orchestrate(self, state: AgentState) -> AgentState:
        console.print(f"\n[bold cyan]🎯 Orchestrator[/bold cyan] analyzing: '{state['query']}'")

        result_json = self._llm(
            system="You are a query analysis agent. Extract key entities and decide retrieval strategy. Return ONLY valid JSON, no explanation.",
            user=f"""Query: {state['query']}

Return JSON:
{{
  "key_entities": ["entity1", "entity2"],
  "strategy": "hybrid",
  "reasoning": "why this strategy"
}}

Strategy options:
- "vector": factual questions, definitions, "what is X"
- "graph": relationship questions, "who created X", "how does X relate to Y"
- "hybrid": complex multi-hop questions combining both""",
            max_tokens=400,
        )

        try:
            clean = result_json.strip().strip("```json").strip("```")
            data = json.loads(clean)
        except json.JSONDecodeError:
            data = {"key_entities": [], "strategy": "hybrid"}

        console.print(f"  Entities: {data.get('key_entities', [])}")
        console.print(f"  Strategy: {data.get('strategy', 'hybrid')}")

        return {
            **state,
            "key_entities": data.get("key_entities", []),
            "search_strategy": data.get("strategy", "hybrid"),
            "retry_count": 0,
            "retrieved_chunks": [],
            "graph_context": {},
            "evidence_sufficient": False,
            "critic_feedback": "",
            "answer": "",
            "citations": [],
            "confidence_score": 0.0,
        }

    def retrieve(self, state: AgentState) -> AgentState:
        console.print(f"\n[bold cyan]🔍 Retriever[/bold cyan] running {state['search_strategy']} search")

        strategy = state["search_strategy"]
        new_chunks = []
        graph_context = {}

        if strategy in ("vector", "hybrid"):
            query_vector = self.embedder.embed_query(state["query"])
            results = self.vector_store.search(
                query_vector=query_vector,
                query_text=state["query"],
                top_k=cfg.top_k,
            )
            new_chunks.extend(results)
            console.print(f"  Vector search: {len(results)} chunks retrieved")

        if strategy in ("graph", "hybrid"):
            all_neighborhoods = []
            for entity in state["key_entities"]:
                neighborhood = self.kg.get_neighborhood(entity, hops=cfg.max_graph_hops)
                if neighborhood["found"]:
                    all_neighborhoods.append(neighborhood)
                    console.print(f"  Graph: found '{entity}' "
                                  f"({len(neighborhood['nodes'])} nodes, "
                                  f"{len(neighborhood['edges'])} edges)")

            graph_context = {"neighborhoods": all_neighborhoods}

            for nbr in all_neighborhoods:
                for edge in nbr["edges"]:
                    fact_text = f"{edge['from']} {edge['predicate'].replace('_', ' ').lower()} {edge['to']}"
                    new_chunks.append({
                        "text": fact_text,
                        "score": 0.9,
                        "doc_title": "Knowledge Graph",
                        "doc_id": "graph",
                        "chunk_id": f"graph_{edge['from']}_{edge['to']}",
                    })

        seen_ids = set()
        deduped = []
        for chunk in new_chunks:
            cid = chunk.get("chunk_id", chunk.get("text", "")[:50])
            if cid not in seen_ids:
                seen_ids.add(cid)
                deduped.append(chunk)

        deduped.sort(key=lambda x: x.get("score", 0), reverse=True)

        return {
            **state,
            "retrieved_chunks": state["retrieved_chunks"] + deduped,
            "graph_context": graph_context,
        }

    def critique(self, state: AgentState) -> AgentState:
        console.print(f"\n[bold cyan]🔬 Critic[/bold cyan] evaluating evidence")

        if state["retry_count"] >= 2:
            console.print("  Max retries reached — proceeding")
            return {**state, "evidence_sufficient": True}

        evidence_text = "\n\n".join([
            f"[{i+1}] (score: {c.get('score', 0):.2f}) {c['text']}"
            for i, c in enumerate(state["retrieved_chunks"][:8])
        ])

        verdict = self._llm(
            system="You are a strict evidence quality evaluator. Be concise.",
            user=f"""Query: {state['query']}

Retrieved evidence:
{evidence_text}

Is this evidence sufficient to answer the query accurately?
Reply with JSON:
{{"sufficient": true/false, "reason": "brief explanation", "missing": "what's missing if not sufficient"}}""",
            max_tokens=200,
        )

        try:
            clean = verdict.strip().strip("```json").strip("```")
            data = json.loads(clean)
            sufficient = data.get("sufficient", True)
            missing = data.get("missing", "")
        except json.JSONDecodeError:
            sufficient = True
            missing = ""

        if sufficient:
            console.print(f"  [green]✓ Evidence sufficient[/green]")
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

        expanded_query = state["query"]
        if state["critic_feedback"]:
            expanded_query = f"{state['query']} {state['critic_feedback']}"

        query_vector = self.embedder.embed_query(expanded_query)
        new_results = self.vector_store.search(
            query_vector=query_vector,
            query_text=expanded_query,
            top_k=cfg.top_k,
        )
        console.print(f"  Retrieved {len(new_results)} additional chunks")

        return {
            **state,
            "retrieved_chunks": state["retrieved_chunks"] + new_results,
        }

    def generate(self, state: AgentState) -> AgentState:
        console.print(f"\n[bold cyan]✍️  Generator[/bold cyan] synthesizing answer")

        top_chunks = state["retrieved_chunks"][:cfg.top_k * 2]
        sources = []
        evidence_blocks = []

        for i, chunk in enumerate(top_chunks):
            title = chunk.get("doc_title", "Unknown")
            if title not in sources:
                sources.append(title)
            src_idx = sources.index(title) + 1
            evidence_blocks.append(f"[{src_idx}] {chunk['text']}")

        evidence_text = "\n\n".join(evidence_blocks)

        raw_answer = self._llm(
            system="You are a research assistant. Answer questions using ONLY the provided evidence. Always cite sources using [N] notation. If evidence is insufficient, say so. End your response with: CONFIDENCE: 0.X",
            user=f"""Question: {state['query']}

Evidence:
{evidence_text}

Sources:
{chr(10).join(f'[{i+1}] {s}' for i, s in enumerate(sources))}

Provide a comprehensive, accurate answer with citations.""",
            max_tokens=1500,
        )

        confidence = 0.7
        answer_text = raw_answer
        if "CONFIDENCE:" in raw_answer:
            parts = raw_answer.rsplit("CONFIDENCE:", 1)
            answer_text = parts[0].strip()
            try:
                confidence = float(parts[1].strip()[:3])
            except (ValueError, IndexError):
                pass

        console.print(f"  [green]✓ Answer generated (confidence: {confidence:.1f})[/green]")

        return {
            **state,
            "answer": answer_text,
            "citations": sources,
            "confidence_score": confidence,
        }

    def route_after_critic(self, state: AgentState) -> str:
        if state["evidence_sufficient"]:
            return "generate"
        return "retrieve_more"

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
            {
                "generate": "generate",
                "retrieve_more": "retrieve_more",
            }
        )
        builder.add_edge("retrieve_more", "critique")
        builder.add_edge("generate", END)

        return builder.compile()

    def run(self, query: str) -> AgentState:
        console.print(f"\n{'='*60}")
        console.print(f"[bold]Query:[/bold] {query}")
        console.print(f"{'='*60}")

        initial_state: AgentState = {
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
        }

        return self.graph.invoke(initial_state)