import json 
from typing import List,Dict,Tuple
import networkx as nx
from openai import OpenAI
from rich.console import Console 
from config import cfg
from ingestion.chunker import Chunk  
console=Console()

class Entity:
    """
    A node in the knowledge graph 
    """
    def __init__(self,name:str,entity_type:str,source_doc:str):
        self.name=name
        self.entity_type=entity_type
        self.source_doc=source_doc

class Relation:
    """
    A directed edge in the knowledge graph 
    """
    def __init__(self,subject:str,predicate:str,obj:str,source_doc:str):
        self.subject=subject
        self.predicate=predicate
        self.obj=obj
        self.source_doc=source_doc

class KnowledgeGraph:
    def __init__(self):
        self.graph=nx.DiGraph()
        self.llm=OpenAI(base_url=cfg.ollama_base_url,api_key=cfg.ollama_api_key)
    
    def extract_and_add(self,chunks:List[Chunk]):
        console.print(f"\n[bold]Building Knowledge Graph from {len(chunks)} chunks...[/bold]")
        for chunk in chunks:
            entities,relations=self._extract_with_llm(chunk)
            self._add_to_graph(entities,relations,chunk.doc_id)
        console.print(
            f"[green] Graph built: {self.graph.number_of_nodes()} nodes, " # pyright: ignore[reportAttributeAccessIssue]
            f"{self.graph.number_of_edges()} edges [/green]"
        )
    def _extract_with_llm(self,chunk:Chunk)->Tuple[List[Entity],List[Relation]]:
        prompt=f"""Extract named entities and relationships from this text.

          Text:
          {chunk.text}

         Return ONLY valid JSON in this exact format (no explanation, no markdown):
        {{
         "entities": [
          {{"name": "EntityName", "type": "PERSON|ORGANIZATION|TECHNOLOGY|CONCEPT|LOCATION"}}
         ],
         "relations": [
           {{"subject": "EntityName", "predicate": "RELATION_TYPE", "object": "EntityName"}}
         ]
        }} 
        Rules:
        - Only extract entities explicitly named in the text
        - Predicate must be UPPERCASE_WITH_UNDERSCORES (e.g. FOUNDED_BY, DEVELOPED_BY, WON, TREATS)
        - Both subject and object must be entities you listed
        - Maximum 15 entities, 20 relations
        - Return empty arrays if nothing clear to extract"""

        try:
            response=self.llm.chat.completions.create(
                model=cfg.ollama_model,
                messages=[{"role":"user","content":prompt}],
                temperature=0,
                extra_body={"think":False}                    
            )
            raw=response.choices[0].message.content.strip()    
            if "<think>" in raw:
                raw = raw.split("</think>")[-1].strip()        
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw=raw.strip()
            if not raw:
                return [],[]
            start = raw.find("{")               
            end = raw.rfind("}") + 1
            if start==-1 or end==0:
                return [],[]
            raw = raw[start:end]
            data = json.loads(raw)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            console.print(f"[yellow]⚠ Extraction failed for {chunk.chunk_id}: {e}[/yellow]")
            return [], []



        entities = [
            Entity(
                name=e.get("name", ""),
                entity_type=e.get("type", "CONCEPT"),
                source_doc=chunk.doc_id,
            )
            for e in data.get("entities", [])
            if isinstance(e,dict) and e.get("name")
        ]

        relations = [
            Relation(
                subject=r.get("subject", ""),
                predicate=r.get("predicate", "RELATED_TO"),
                obj=r.get("object", ""),
                source_doc=chunk.doc_id,
            )
            for r in data.get("relations", [])
            if isinstance(r,dict) and r.get("subject") and r.get("object")
        ]

        return entities, relations
    
    def _add_to_graph(self,entities:List[Entity],relations:List[Relation],doc_id:str):
        #adding nodes
        for entity in entities:
            self.graph.add_node(
                entity.name,
                type=entity.entity_type,
                source=entity.source_doc,
            ) 
        entity_names={e.name for e in entities}
        #adding edges
        for rel in relations:
            if rel.subject in entity_names and rel.obj in entity_names:
                self.graph.add_edge(
                    rel.subject,
                    rel.obj,
                    predicate=rel.predicate,
                    source=rel.source_doc, # type: ignore
                )
    def get_neighborhood(self,entity_name:str,hops:int= None)->Dict: # pyright: ignore[reportArgumentType]
        if hops is None:
            hops=cfg.max_graph_hops
        
        best_node=self._find_node(entity_name)
        if not best_node:
            return{"nodes":[],"edges":[],"found":False}
        
        subgraph=nx.ego_graph(self.graph,best_node,radius=hops)

        nodes=[
            {
                "name":n,
                "type":self.graph.nodes[n].get("type","UNKNOWN"),
                "source":self.graph.nodes[n].get('source',"") # pyright: ignore[reportAttributeAccessIssue]
            }
            for n in subgraph.nodes()
        ]

        edges=[

            {
                "from":u,
                "to":v,
                "predicate":self.graph.edges[u,v].get("predicate","RELATED_TO"),
            }
            for u, v in subgraph.edges()
        ]
        return{
            "center":best_node,
            "nodes":nodes,
            "edges":edges,
            "found":True,
        }
    
    def _find_node(self,name:str) -> str | None:
        name_lower=name.lower()

        for node in self.graph.nodes():
            if node.lower()==name_lower:
                return node
        for node in self.graph.nodes():
            if name_lower in node.lower() or node.lower() in name_lower:
                return node
        return None
    
    def get_all_nodes(self) -> List[str]:
        return list(self.graph.nodes())
    def get_stats(self)->Dict:
        return {
            "nodes":self.graph.number_of_nodes(),
            "edges":self.graph.number_of_edges(),
            "density":round(nx.density(self.graph),4),
        }


