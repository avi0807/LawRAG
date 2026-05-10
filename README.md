# LawRAG — Multi-Agent GraphRAG System for Indian Legal Research

A production-grade Retrieval-Augmented Generation pipeline that combines vector search, knowledge graph reasoning, and multi-agent orchestration to answer questions over landmark Indian Supreme Court judgments and the Constitution of India.

---

## Architecture

```text
Indian Kanoon Judgments -> Chunker -> Embedder -> Qdrant (vector DB)
                        -> Qwen3:8b (NER) -> NetworkX (knowledge graph)

Query -> Orchestrator Agent -> Retriever Agent (vector + graph)
      -> Critic Agent -> Generator Agent -> Answer + Citations
      -> RAGAS Evaluator -> Faithfulness / Relevancy / Precision
```

## Stack

| Layer | Tool | Role |
|---|---|---|
| LLM | Qwen3:8b via Ollama | Entity extraction, orchestration, generation |
| Embeddings | nomic-embed-text via Ollama | Text to vector conversion |
| Vector database | Qdrant | Semantic similarity search with reranking |
| Knowledge graph | NetworkX | Entity relationship traversal |
| Agent orchestration | LangGraph | Stateful multi-agent pipeline with retry loops |
| Reranker | FlashRank | Post-retrieval chunk reordering |
| Evaluation | RAGAS-style | Faithfulness, relevancy, context precision |
| Backend | FastAPI | REST API with async pipeline execution |
| Frontend | Vanilla HTML/CSS/JS | Chat interface, no framework dependencies |

---

## How It Works

### Ingestion Pipeline

1. Fetches landmark constitutional judgments from Indian Kanoon
2. Extracts Constitution of India articles from PDF documents
3. Splits judgments into overlapping token-aware chunks using a recursive sentence splitter
4. Embeds all chunks using nomic-embed-text and stores vectors in Qdrant
5. Extracts named entities and relationships from each chunk using Qwen3:8b
6. Builds a directed knowledge graph (NetworkX DiGraph) from extracted legal triples

### Query Pipeline

1. Orchestrator analyzes the question, extracts legal entities, and decides retrieval strategy (vector, graph, or hybrid)
2. Retriever runs vector similarity search against Qdrant and traverses the knowledge graph neighborhood of extracted entities
3. FlashRank reranks retrieved chunks by relevance to the legal query
4. Critic evaluates whether the evidence is sufficient, looping back to retrieve more if not (max 2 retries)
5. Generator synthesizes a grounded answer with citations and confidence score
6. RAGAS evaluator scores the answer on faithfulness, relevancy, and context precision

---

## Dataset

### Landmark Cases Included

- Maneka Gandhi v Union of India
- Kesavananda Bharati v State of Kerala
- A.K. Gopalan v State of Madras
- Minerva Mills v Union of India
- S.R. Bommai v Union of India
- Vishaka v State of Rajasthan
- MC Mehta v Union of India
- Indra Sawhney v Union of India
- ADM Jabalpur v Shivkant Shukla
- Olga Tellis v Bombay Municipal Corporation
- Bachan Singh v State of Punjab
- Shah Bano Begum Case
- Hussainara Khatoon v State of Bihar

### Constitutional Coverage

- Fundamental Rights
- Directive Principles
- Constitutional Amendments
- Judicial Review
- Basic Structure Doctrine
- Preventive Detention
- Environmental Jurisprudence

---

## Setup

### Requirements

- Python 3.10+
- Ollama installed and running
- 16GB RAM recommended
- 6GB VRAM or CPU inference

### Install Ollama and pull models

```bash
curl -fsSL https://ollama.com/install.sh | sh

ollama pull qwen3:8b-q4_K_M
ollama pull nomic-embed-text
```

### Clone and install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/LawRAG.git

cd LawRAG

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

```env
INDIANKANOON_API_TOKEN=your_token_here
```

### Run

```bash
# Terminal 1 — start Ollama
ollama serve

# Terminal 2 — run CLI pipeline
python main.py

# Terminal 2 — or run FastAPI backend
uvicorn api:app --reload --port 8000
```

Then open `ui.html` in your browser to use the legal research interface.

---

## Example Queries

```text
What did the Supreme Court say about Article 21 in Maneka Gandhi?

What is the basic structure doctrine and which case established it?

Which judgments expanded the interpretation of personal liberty?

How does Kesavananda Bharati relate to Minerva Mills?

Which cases discuss preventive detention under Article 22?

What constitutional principles emerged from Vishaka v State of Rajasthan?
```

---

## Project Structure

```text
LawRAG/
├── api.py
├── ui.html
├── main.py
├── config.py
├── requirements.txt
├── data/
│   ├── ik_loader.py
│   └── constitution.pdf
├── ingestion/
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   └── knowledge_graph.py
├── agents/
│   └── graph_rag_agent.py
└── evaluation/
    └── ragas_eval.py
```

---

## Configuration

All settings are defined in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `ollama_model` | `qwen3:8b-q4_K_M` | LLM for all agent tasks |
| `embedding_model` | `nomic-embed-text` | Embedding model |
| `embedding_dim` | `768` | Vector dimensions |
| `chunk_size` | `512` | Max tokens per chunk |
| `chunk_overlap` | `50` | Overlap tokens between chunks |
| `top_k` | `12` | Chunks retrieved per query |
| `max_graph_hops` | `2` | Knowledge graph traversal depth |

---

## Evaluation Metrics

| Metric | Target | Description |
|---|---|---|
| Faithfulness | > 0.8 | Answer claims are grounded in retrieved evidence |
| Answer relevancy | > 0.8 | Answer addresses the legal question |
| Context precision | > 0.7 | Retrieved chunks are relevant to the query |

---

## Knowledge Graph Capabilities

### Entity Types

- CASE
- ARTICLE
- JUDGE
- COURT
- PERSON
- ORGANIZATION
- ACT
- CONCEPT
- AMENDMENT

### Relationship Types

- CITED_IN
- OVERRULED_BY
- UPHELD_BY
- INTERPRETED_BY
- GUARANTEES
- RESTRICTS
- AMENDS
- ESTABLISHES
- DERIVED_FROM

---

## Future Improvements

- Neo4j integration for persistent graph storage
- Citation network visualization
- Streaming FastAPI responses
- Legal benchmark datasets
- Fine-tuned legal embeddings
- Multi-jurisdiction legal support
- Judge-aware reasoning
- Temporal legal reasoning
- PDF upload support
- Conversational memory

