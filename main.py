import os
import sys
import pickle
from rich.console import Console
from rich.panel import Panel

from config import cfg
from data.ik_loader import SAMPLE_DOCUMENTS
from ingestion.chunker import RecursiveChunker
from ingestion.embedder import Embedder
from ingestion.vector_store import VectorStore
from ingestion.knowledge_graph import KnowledgeGraph
from agents.graph_rag_agent import MultiAgentGraphRAG
from evaluation.ragas_eval import SimpleRAGASEvaluator

console = Console()


def run_ingestion_pipeline():
    console.print(Panel("[bold]Phase 1: Document Ingestion[/bold]", style="blue"))

    chunker = RecursiveChunker()
    embedder = Embedder()
    vector_store = VectorStore()
    kg = KnowledgeGraph()

    all_chunks = []

    for doc in SAMPLE_DOCUMENTS:
        console.print(f"\n[bold]Processing:[/bold] {doc['title']}")
        chunks = chunker.chunk_document(
            doc_id=doc["id"],
            title=doc["title"],
            text=doc["content"],
            metadata={"source": doc["source"], "year": doc.get("year", 2024)},
        )
        console.print(f"  Produced {len(chunks)} chunks")
        all_chunks.extend(chunks)
    if vector_store.count()>0:
        console.print(f"[dim]Vector store already populated: {vector_store.count()} vectors — skipping embedding[/dim]")

    else:
        console.print(f"\n[bold]Embedding {len(all_chunks)} total chunks...[/bold]")
        chunks_with_embeddings, embeddings = embedder.embed_chunks(all_chunks)

        vector_store.upsert_chunks(chunks_with_embeddings, embeddings)
        console.print(f"[green]✓ Vector store: {vector_store.count()} vectors stored[/green]")

        
    kg_cache = "data/kg_cache.pkl"
    if os.path.exists(kg_cache):
        console.print("[dim]Loading knowledge graph from cache...[/dim]")

        with open(kg_cache, "rb") as f:
            kg.graph = pickle.load(f)
    else:
        kg.extract_and_add(all_chunks)
        with open(kg_cache, "wb") as f:
            pickle.dump(kg.graph, f)
            
    stats = kg.get_stats()
    console.print(f"[green]✓ Knowledge graph: {stats['nodes']} nodes, {stats['edges']} edges[/green]")

    # DEBUG — add after "Loading knowledge graph from cache"

    debug_embedder = Embedder()
    test_vec = debug_embedder.embed_query("Maneka Gandhi personal liberty Article 21 Supreme Court")
    results = vector_store.search(query_vector=test_vec, query_text="Maneka Gandhi", top_k=10)
    console.print("\n[bold]DEBUG — Top 10 results for Maneka Gandhi query:[/bold]")
    for r in results:
        console.print(f"  {r.get('doc_title','?')[:60]} | score: {r.get('score',0):.3f}")

    return vector_store, kg, embedder

    


def run_query_pipeline(agent, evaluator, queries):
    console.print(Panel("[bold]Phase 2: Query Pipeline[/bold]", style="green"))

    results = []
    for query in queries:
        final_state = agent.run(query)

        context_texts = [
            c["text"] for c in final_state["retrieved_chunks"][:5]
        ]

        console.print(f"\n[bold green]Answer:[/bold green]")
        console.print(final_state["answer"])
        console.print(f"\n[dim]Sources: {', '.join(final_state['citations'])}[/dim]")
        console.print(f"[dim]Confidence: {final_state['confidence_score']:.2f}[/dim]")

        results.append({
            "question": query,
            "answer": final_state["answer"],
            "context_chunks": context_texts,
            "state": final_state,
        })

    return results


def main():
    console.print(Panel(
        "[bold]Multi-Agent GraphRAG System[/bold]\n"
        "Vector search + Knowledge Graph + Multi-agent orchestration + RAGAS evaluation",
        style="bold blue"
    ))

    vector_store, kg, embedder = run_ingestion_pipeline()

    agent = MultiAgentGraphRAG(
        vector_store=vector_store,
        knowledge_graph=kg,
        embedder=embedder,
    )
    evaluator = SimpleRAGASEvaluator()

    test_queries = [
         "What did the Supreme Court say about Article 21 in Maneka Gandhi case?",
        "What are the fundamental rights guaranteed under the Indian Constitution?",
        "What is the basic structure doctrine and which case established it?",
    ]

    results = run_query_pipeline(agent, evaluator, test_queries)

    console.print(Panel("[bold]Phase 3: RAGAS Evaluation[/bold]", style="yellow"))

    eval_cases = [
        {
            "question": r["question"],
            "answer": r["answer"],
            "context_chunks": r["context_chunks"],
        }
        for r in results
    ]

    avg_scores = evaluator.evaluate_batch(eval_cases)

    console.print(Panel(
        f"[bold]System Summary[/bold]\n\n"
        f"Documents ingested:   {len(SAMPLE_DOCUMENTS)}\n"
        f"Graph nodes:          {kg.get_stats()['nodes']}\n"
        f"Graph edges:          {kg.get_stats()['edges']}\n"
        f"Vectors stored:       {vector_store.count()}\n\n"
        f"RAGAS Faithfulness:   {avg_scores['faithfulness']:.3f}\n"
        f"RAGAS Relevancy:      {avg_scores['answer_relevancy']:.3f}\n"
        f"RAGAS Precision:      {avg_scores['context_precision']:.3f}\n"
        f"Overall Score:        {avg_scores['overall']:.3f}",
        style="bold green"
    ))


if __name__ == "__main__":
    main()