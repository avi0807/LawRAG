import arxiv
import os
import time
from typing import List, Dict
from rich.console import Console
import json

console = Console()

def fetch_arxiv_papers(
    query: str = "retrieval augmented generation large language models",
    max_results: int = 15,
    cache_dir: str = "data/arxiv_cache"
) -> List[Dict]:
    """
    Fetch papers from ArXiv and return in the same format as SAMPLE_DOCUMENTS
    so the rest of the pipeline works without any changes.
    """
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = os.path.join(cache_dir, "papers.json")

    if os.path.exists(cache_file): 
        console.print(f"[dim]Loading {max_results} papers from cache...[/dim]")
        with open(cache_file, "r") as f:
            return json.load(f)

    console.print(f"[bold]Fetching {max_results} papers from ArXiv...[/bold]")
    console.print(f"[dim]Query: {query}[/dim]")

    client = arxiv.Client(num_retries=5,delay_seconds=5)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    documents = []
    for i, paper in enumerate(client.results(search)):
        doc = {
            "id": f"arxiv_{paper.entry_id.split('/')[-1].replace('.', '_')}",
            "title": paper.title,
            "content": f"{paper.title}\n\n{paper.summary}",
            "source": "arxiv",
            "year": paper.published.year,
            "url": paper.entry_id,
            "authors": [a.name for a in paper.authors[:5]],
            "categories": paper.categories,
        }
        documents.append(doc)
        console.print(f"  [{i+1}/{max_results}] {paper.title[:70]}...")
        time.sleep(1)



    with open(cache_file, "w") as f:
        json.dump(documents, f, indent=2)

    console.print(f"[green]✓ Fetched and cached {len(documents)} papers[/green]")
    return documents

SAMPLE_DOCUMENTS = fetch_arxiv_papers()