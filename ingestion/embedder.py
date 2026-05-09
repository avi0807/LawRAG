import numpy as np 
from openai import OpenAI
from config import cfg

client=OpenAI(base_url=cfg.ollama_base_url,api_key=cfg.ollama_api_key)

class Embedder:
    def embed_texts(self,texts:list[str]):
        response=client.embeddings.create(
            model=cfg.embedding_model,
            input=texts,       
        )
        vectors=[item.embedding for item in response.data]
        return np.array(vectors,dtype=np.float32)
    def embed_chunks(self,chunks):
        texts=[c.text for c in chunks]
        return chunks,self.embed_texts(texts)
    def embed_query(self,query:str)->np.ndarray:
        response=client.embeddings.create(
            model=cfg.embedding_model,
            input=[query], 
        )

        return np.array(response.data[0].embedding,dtype=np.float32)
    

