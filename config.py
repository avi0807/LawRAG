from dataclasses import dataclass
import os


@dataclass
class Config:
    # LLM
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen3:8b-q4_K_M"
    ollama_extraction_model: str = "qwen3:8b-q4_K_M"   
    ollama_api_key: str = "ollama"

    # Embeddings
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768

    # Vector store
    qdrant_location: str = "./data/qdrant_storage"
    qdrant_collection: str = "documents"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # ----- Hybrid retrieval -----
    dense_top_k: int = 20            # dense recall pool
    bm25_top_k: int = 20             # BM25 recall pool
    rrf_k: int = 60                  # RRF damping constant
    rerank_pool: int = 30            # candidates passed to FlashRank
    final_top_k: int = 8             # passed to generator
    top_k: int = 8                   # legacy alias used elsewhere

    # ----- Knowledge graph -----
    max_graph_hops: int = 2
    ppr_alpha: float = 0.85
    ppr_top_n: int = 12
    kg_chunk_cap_per_entity: int = 4
    kg_extract_workers: int = 1
    kg_extract_timeout: int = 180

    # ----- Critic heuristic / loop -----
    skip_critic_min_top_score: float = 0.40
    max_retries: int = 2

    # ----- Caching -----
    embed_cache_size: int = 512
    answer_cache_size: int = 64

    # ----- Confidence calibration -----
    # confidence = w_top * top_score + w_gap * top_minus_p5_gap +
    #              w_entity * entity_grounding + w_grounded * sentence_grounded_rate
    # Weights should sum to ~1.0. Grounded rate is the strongest negative
    # signal we have when the LLM hallucinates or returns the fallback string.
    confidence_w_top: float = 0.35
    confidence_w_gap: float = 0.15
    confidence_w_entity: float = 0.15
    confidence_w_grounded: float = 0.35

    # ----- Citation grounding -----
    citation_min_overlap: float = 0.18   # word-overlap threshold for "this sentence is grounded in this chunk"
    citation_top_n: int = 3              # max chunks attributed per sentence

    # ----- API auth & rate limit -----
    api_token: str = os.environ.get("LAWRAG_API_TOKEN", "")  # empty = auth disabled (dev mode)
    rate_limit_per_minute: int = 30      # per session_id

    # ----- Observability -----
    debug: bool = False                  # gate noisy diagnostic prints
    log_dir: str = "data/logs"           # per-query trace JSONL


cfg = Config()
