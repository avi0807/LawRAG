from dataclasses import dataclass

@dataclass
class Config:
    # gemini_api_key:str=""
    # gemini_model:str="gemini-3.0-flash"

    ollama_base_url:str="http://localhost:11434/v1"
    ollama_model:str="qwen3:8b-q4_K_M"
    ollama_api_key:str="ollama"


    embedding_model:str="nomic-embed-text"
    embedding_dim:int=768

    qdrant_location:str="./data/qdrant_storage"
    qdrant_collection:str="documents"

    chunk_size:int=512
    chunk_overlap:int=50

    top_k:int=12

    max_graph_hops:int=2

cfg=Config()
