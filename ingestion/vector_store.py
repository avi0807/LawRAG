from typing import List,Dict
import numpy as  np 
from qdrant_client import QdrantClient
from flashrank import Ranker,RerankRequest
from qdrant_client.models import(
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    )
from rich.console import Console
from config import cfg
from ingestion.chunker import Chunk

console=Console()

class VectorStore:
    def __init__(self):
        self.client=QdrantClient(path=cfg.qdrant_location)
        self.collection_name=cfg.qdrant_collection
        self._create_collection()
        self.ranker=Ranker()
    
    def _create_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=cfg.embedding_dim,
                    distance=Distance.COSINE,
                )
            )
            console.print(f"[green] Created Qdrant collection: {self.collection_name}[/green]")
    def upsert_chunks(self,chunks:List[Chunk],embeddings: np.ndarray):
        points=[]
        for chunk,vector in zip(chunks,embeddings):
            point=PointStruct(
                id=self._chunk_id_to_int(chunk.chunk_id),
                vector=vector.tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "doc_title": chunk.doc_title,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    **chunk.metadata                    
                }
            )
            points.append(point)
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        console.print(f"[green]Upserted {len(points)} chunks into vector store[/green]")
    
    def search(self,query_vector,query_text:str="",top_k=None,filter_by=None,) -> List[Dict]:
        if top_k is None:
            top_k=cfg.top_k
        qdrant_filter=None
        if filter_by:
            conditions=[
                FieldCondition(key=k,match=MatchValue(value=v))
                for k,v in filter_by.items()
            ]
            qdrant_filter=Filter(must=conditions)
        
        response=self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        results=response.points
        

        raw=[
            {"score":round(r.score,4),**r.payload}
            for r in results
        ]

        if query_text:
       
            passages=[
            {"id":i,"text":c.get("text","")}
            for i,c in enumerate(raw)
            ]
            reranked=self.ranker.rerank(RerankRequest(query=query_text,passages=passages))
            order=[r["id"] for r in reranked]
            raw=[raw[i] for i in order]
        return raw

    
    def _chunk_id_to_int(self,chunk_id:str)->int:
        return abs(hash(chunk_id)) % (2**63)
    
    def count(self)->int:
        return self.client.count(collection_name=self.collection_name).count

