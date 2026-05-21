import os
from rich.console import Console
from rich.panel import Panel

from agents.graph_rag_agent import MultiAgentGraphRAG
from data.ik_loader import SAMPLE_DOCUMENTS
from evaluation.ragas_eval import SimpleRAGASEvaluator
from ingestion.bm25_store import BM25Store
from ingestion.chunker import RecursiveChunker
from ingestion.embedder import Embedder
from ingestion.knowledge_graph import KnowledgeGraph
from ingestion.vector_store import VectorStore

console = Console()

KG_CACHE = "data/kg_cache.pkl"
BM25_CACHE = "data/bm25_cache.pkl"


def run_ingestion_pipeline():
    console.print(Panel("[bold]Phase 1: Document Ingestion[/bold]", style="blue"))

    chunker = RecursiveChunker()
    embedder = Embedder()
    vector_store = VectorStore()
    kg = KnowledgeGraph()
    bm25 = BM25Store(cache_path=BM25_CACHE)

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

    # Dense
    if vector_store.count() > 0:
        console.print(f"[dim]Vector store already populated: {vector_store.count()} vectors — skipping embedding[/dim]")
    else:
        console.print(f"\n[bold]Embedding {len(all_chunks)} total chunks...[/bold]")
        chunks_with_embeddings, embeddings = embedder.embed_chunks(all_chunks)
        vector_store.upsert_chunks(chunks_with_embeddings, embeddings)
        console.print(f"[green]✓ Vector store: {vector_store.count()} vectors[/green]")

    # BM25
    bm25.build_or_load(all_chunks)

    # Knowledge graph
    if kg.load(KG_CACHE):
        console.print("[dim]Loaded knowledge graph from cache[/dim]")
        # If cache is from old format with no entity_chunks, rebuild the index in memory
        if not kg.entity_chunks:
            console.print("[yellow]KG cache missing entity→chunk index — rebuilding lightweight index...[/yellow]")
            for ch in all_chunks:
                kg._index_chunk_for_entities(ch, [])  # adds article + doc_title buckets
    else:
        kg.extract_and_add(all_chunks)
        kg.save(KG_CACHE)

    stats = kg.get_stats()
    console.print(
        f"[green]✓ KG: {stats['nodes']} nodes, {stats['edges']} edges, "
        f"{len(kg.entity_chunks)} entity buckets[/green]"
    )

    return vector_store, bm25, kg, embedder


def run_query_pipeline(agent, queries):
    console.print(Panel("[bold]Phase 2: Query Pipeline[/bold]", style="green"))
    results = []
    for query in queries:
        final_state = agent.run(query)
        context_texts = [c["text"] for c in final_state["retrieved_chunks"][:5]]

        console.print(f"\n[bold green]Answer:[/bold green]\n{final_state['answer']}")
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
        "[bold]Multi-Agent GraphRAG (BM25 + Dense + Graph + RRF)[/bold]\n"
        "Hybrid retrieval, single rerank, streaming generation, PPR-expanded graph context",
        style="bold blue",
    ))

    vector_store, bm25, kg, embedder = run_ingestion_pipeline()

    agent = MultiAgentGraphRAG(
        vector_store=vector_store,
        knowledge_graph=kg,
        embedder=embedder,
        bm25_store=bm25,
    )
    evaluator = SimpleRAGASEvaluator()

    test_queries = [
        "What did the Supreme Court say about Article 21 in Maneka Gandhi case?",
        "What are the fundamental rights guaranteed under the Indian Constitution?",
        "What is the basic structure doctrine and which case established it?",
    ]

    results = run_query_pipeline(agent, test_queries)

    console.print(Panel("[bold]Phase 3: RAGAS Evaluation[/bold]", style="yellow"))

    eval_cases = [
        {"question": r["question"], "answer": r["answer"], "context_chunks": r["context_chunks"]}
        for r in results
    ]
    avg_scores = evaluator.evaluate_batch(eval_cases)

    console.print(Panel(
        f"[bold]System Summary[/bold]\n\n"
        f"Documents ingested:   {len(SAMPLE_DOCUMENTS)}\n"
        f"Graph nodes:          {kg.get_stats()['nodes']}\n"
        f"Graph edges:          {kg.get_stats()['edges']}\n"
        f"Vectors stored:       {vector_store.count()}\n"
        f"BM25 chunks:          {bm25.count()}\n\n"
        f"RAGAS Faithfulness:   {avg_scores['faithfulness']:.3f}\n"
        f"RAGAS Relevancy:      {avg_scores['answer_relevancy']:.3f}\n"
        f"RAGAS Precision:      {avg_scores['context_precision']:.3f}\n"
        f"Overall Score:        {avg_scores['overall']:.3f}",
        style="bold green",
    ))


if __name__ == "__main__":
    main()
