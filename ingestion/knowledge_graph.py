import json
import os
import pickle
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional

import networkx as nx
from openai import OpenAI
from rich.console import Console

from config import cfg
from ingestion.chunker import Chunk

console = Console()


# Patterns for case citations inside judgment text.
# Examples it catches:
#   "Maneka Gandhi v. Union of India"
#   "Kesavananda Bharati vs State of Kerala"
#   "A.K. Gopalan v Madras"
_CASE_CITE_RE = re.compile(
    r"([A-Z][A-Za-z.\s']{2,60}?)\s+v(?:s)?\.?\s+([A-Z][A-Za-z.\s']{2,60}?)(?=[,.\(\n]|\s+\(|\s+AIR|\s+SCC|\s+\d{4})",
)
# Article references: "Article 21", "Article 21A", "Articles 14 and 21"
_ARTICLE_RE = re.compile(r"Article\s+(\d+[A-Z]?)", re.IGNORECASE)


class Entity:
    def __init__(self, name: str, entity_type: str, source_doc: str):
        self.name = name
        self.entity_type = entity_type
        self.source_doc = source_doc


class Relation:
    def __init__(self, subject: str, predicate: str, obj: str, source_doc: str):
        self.subject = subject
        self.predicate = predicate
        self.obj = obj
        self.source_doc = source_doc


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.llm = OpenAI(base_url=cfg.ollama_base_url, api_key=cfg.ollama_api_key)
        # entity name (canonical) -> list of (chunk_id, doc_id, doc_title, text)
        self.entity_chunks: Dict[str, List[Dict]] = defaultdict(list)
        # Protects the graph + entity_chunks during parallel ingestion.
        self._write_lock = threading.Lock()

    # ---------------- Build ----------------

    def extract_and_add(self, chunks: List[Chunk]):
        """Parallel KG construction. LLM calls run in a thread pool; graph
        mutations happen under a lock. Same model, same prompts, same quality —
        just wall-clock cut by the worker count."""
        n = len(chunks)
        workers = max(1, cfg.kg_extract_workers)
        console.print(
            f"\n[bold]Building Knowledge Graph from {n} chunks "
            f"({workers} parallel workers)...[/bold]"
        )

        completed = 0
        failed = 0

        def _process(chunk: Chunk):
            try:
                entities, relations = self._extract_with_llm(chunk)
            except Exception as e:
                return chunk, [], [], e
            return chunk, entities, relations, None

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_process, c) for c in chunks]
            for fut in as_completed(futures):
                chunk, entities, relations, err = fut.result()
                completed += 1

                if err is not None:
                    failed += 1

                # Mutate the graph under a lock — keeps NetworkX happy.
                with self._write_lock:
                    self._add_to_graph(entities, relations, chunk.doc_id)
                    self._add_regex_facts(chunk)
                    self._index_chunk_for_entities(chunk, entities)

                # Live single-line progress.
                status = (
                    f"  [{completed}/{n}] {chunk.doc_title[:48]:<48}  "
                    f"+{len(entities)}E/{len(relations)}R  "
                    f"failed: {failed}"
                )
                if err is not None:
                    status += f"  ⚠ {type(err).__name__}"
                console.print(status, end="\r")

        console.print(
            f"\n[green]✓ Graph built: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges, "
            f"{len(self.entity_chunks)} entity→chunk buckets "
            f"({failed} chunks failed)[/green]"
        )

    def _extract_with_llm(self, chunk: Chunk) -> Tuple[List[Entity], List[Relation]]:
        prompt = f"""Extract legal entities and relationships from this Indian legal text.
Text: {chunk.text}

Return ONLY valid JSON in this exact format (no explanation, no markdown):
{{
  "entities": [
    {{"name": "EntityName", "type": "CASE|ARTICLE|JUDGE|COURT|PERSON|ORGANIZATION|ACT|CONCEPT|AMENDMENT"}}
  ],
  "relations": [
    {{"subject": "EntityName", "predicate": "RELATION_TYPE", "object": "EntityName"}}
  ]
}}

Predicate guide (use these where applicable):
- CITED_IN, OVERRULED_BY, UPHELD_BY, INTERPRETED_BY
- GUARANTEES, RESTRICTS, AMENDS, VIOLATES
- AUTHORED_BY, DECIDED_BY, FILED_BY
- ESTABLISHES, APPLIES_TO, DERIVED_FROM

Rules:
- Only extract entities explicitly named in the text
- Predicate must be UPPERCASE_WITH_UNDERSCORES
- Both subject and object must be in your entities list
- Maximum 15 entities, 20 relations
- Return empty arrays if nothing clear to extract"""

        try:
            response = self.llm.chat.completions.create(
                model=cfg.ollama_extraction_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                extra_body={"think": False},
                timeout=cfg.kg_extract_timeout,
            )
            raw = response.choices[0].message.content.strip()
            if "<think>" in raw:
                raw = raw.split("</think>")[-1].strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            if not raw:
                return [], []
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return [], []
            data = json.loads(raw[start:end])
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            console.print(f"[yellow]⚠ Extraction failed for {chunk.chunk_id}: {e}[/yellow]")
            return [], []
        except Exception as e:
            # Ollama timeouts, network blips, etc. — skip this chunk, keep going
            console.print(f"[yellow]⚠ LLM call failed for {chunk.chunk_id}: {type(e).__name__}[/yellow]")
            return [], []

        entities = [
            Entity(e.get("name", ""), e.get("type", "CONCEPT"), chunk.doc_id)
            for e in data.get("entities", [])
            if isinstance(e, dict) and e.get("name")
        ]
        relations = [
            Relation(
                r.get("subject", ""),
                r.get("predicate", "RELATED_TO"),
                r.get("object", ""),
                chunk.doc_id,
            )
            for r in data.get("relations", [])
            if isinstance(r, dict) and r.get("subject") and r.get("object")
        ]
        return entities, relations

    def _add_to_graph(self, entities: List[Entity], relations: List[Relation], doc_id: str):
        for e in entities:
            if not e.name:
                continue
            self.graph.add_node(e.name, type=e.entity_type, source=e.source_doc)
        names = {e.name for e in entities}
        for r in relations:
            if r.subject in names and r.obj in names:
                self.graph.add_edge(
                    r.subject, r.obj,
                    predicate=r.predicate,
                    source=r.source_doc,
                )

    def _add_regex_facts(self, chunk: Chunk):
        """Cheap, deterministic enrichment: case citations and article mentions."""
        text = chunk.text or ""

        # Article mentions -> connect doc title to article node
        articles_found = set()
        for m in _ARTICLE_RE.finditer(text):
            art = f"Article {m.group(1).upper()}"
            articles_found.add(art)
            self.graph.add_node(art, type="ARTICLE", source=chunk.doc_id)

        # Case citations "X v Y" -> add CASE nodes and CITED_IN edges from this doc
        cited_cases = set()
        for m in _CASE_CITE_RE.finditer(text):
            left = m.group(1).strip().rstrip(".,")
            right = m.group(2).strip().rstrip(".,")
            # Filter out obvious junk
            if len(left) < 3 or len(right) < 3:
                continue
            case_name = f"{left} v {right}"
            if len(case_name) > 100:
                continue
            cited_cases.add(case_name)
            self.graph.add_node(case_name, type="CASE", source=chunk.doc_id)

        # Connect doc-as-case to cited cases (CITES) and to articles (DISCUSSES)
        host = chunk.doc_title or chunk.doc_id
        if host and (cited_cases or articles_found):
            self.graph.add_node(host, type="CASE", source=chunk.doc_id)
            for c in cited_cases:
                if c != host:
                    self.graph.add_edge(host, c, predicate="CITES", source=chunk.doc_id)
            for a in articles_found:
                self.graph.add_edge(host, a, predicate="DISCUSSES", source=chunk.doc_id)

    def _index_chunk_for_entities(self, chunk: Chunk, entities: List[Entity]):
        """Map every entity in this chunk back to the chunk so retrieval can pull
        real chunks (not synthetic 'X RELATED_TO Y' strings) when traversing the graph."""
        seen_names = set()

        for e in entities:
            if e.name and e.name not in seen_names:
                seen_names.add(e.name)
                self._record_chunk(e.name, chunk)

        # Also index articles found by regex
        for m in _ARTICLE_RE.finditer(chunk.text or ""):
            art = f"Article {m.group(1).upper()}"
            if art not in seen_names:
                seen_names.add(art)
                self._record_chunk(art, chunk)

        # And the doc title as a CASE node
        if chunk.doc_title and chunk.doc_title not in seen_names:
            self._record_chunk(chunk.doc_title, chunk)

    def _record_chunk(self, entity_name: str, chunk: Chunk):
        self.entity_chunks[entity_name].append({
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "doc_title": chunk.doc_title,
            "text": chunk.text,
        })

    # ---------------- Persistence ----------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "graph": self.graph,
                "entity_chunks": dict(self.entity_chunks),
            }, f)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        # backward compat: old caches stored just the graph
        if isinstance(data, dict) and "graph" in data:
            self.graph = data["graph"]
            self.entity_chunks = defaultdict(list, data.get("entity_chunks", {}))
        else:
            self.graph = data
            self.entity_chunks = defaultdict(list)
        return True

    # ---------------- Query ----------------

    def get_neighborhood(self, entity_name: str, hops: int = None) -> Dict:
        if hops is None:
            hops = cfg.max_graph_hops

        best = self._find_node(entity_name)
        if not best:
            return {"nodes": [], "edges": [], "found": False}

        sub = nx.ego_graph(self.graph, best, radius=hops)
        nodes = [
            {"name": n,
             "type": self.graph.nodes[n].get("type", "UNKNOWN"),
             "source": self.graph.nodes[n].get("source", "")}
            for n in sub.nodes()
        ]
        edges = [
            {"from": u, "to": v,
             "predicate": self.graph.edges[u, v].get("predicate", "RELATED_TO")}
            for u, v in sub.edges()
        ]
        return {"center": best, "nodes": nodes, "edges": edges, "found": True}

    def personalized_pagerank(self, seed_entities: List[str], top_n: int = None) -> List[Tuple[str, float]]:
        """Run PPR seeded at the resolved seed entities. Returns (node, score) pairs."""
        if top_n is None:
            top_n = cfg.ppr_top_n
        if self.graph.number_of_nodes() == 0:
            return []

        seeds = []
        for s in seed_entities:
            n = self._find_node(s)
            if n:
                seeds.append(n)
        if not seeds:
            return []

        personalization = {n: 0.0 for n in self.graph.nodes()}
        weight = 1.0 / len(seeds)
        for s in seeds:
            personalization[s] = weight

        try:
            # PPR is defined for undirected/directed; nx handles both.
            scores = nx.pagerank(
                self.graph,
                alpha=cfg.ppr_alpha,
                personalization=personalization,
                max_iter=100,
                tol=1e-4,
            )
        except nx.PowerIterationFailedConvergence:
            scores = nx.pagerank(self.graph, alpha=cfg.ppr_alpha,
                                 personalization=personalization, max_iter=300, tol=1e-3)

        # Drop seeds themselves so we surface *new* nodes
        ranked = [(n, sc) for n, sc in scores.items() if n not in seeds]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    def chunks_for_entity(self, entity_name: str, cap: int = None) -> List[Dict]:
        """Return real indexed chunks for an entity (resolves fuzzy)."""
        if cap is None:
            cap = cfg.kg_chunk_cap_per_entity
        node = self._find_node(entity_name)
        if not node:
            return []
        # exact bucket first
        out = list(self.entity_chunks.get(node, []))
        if not out:
            # fall back to fuzzy match across keys
            lower = node.lower()
            for k, v in self.entity_chunks.items():
                if lower in k.lower() or k.lower() in lower:
                    out.extend(v)
                    if len(out) >= cap * 2:
                        break
        # de-dup by chunk_id, preserve order
        seen = set()
        unique = []
        for c in out:
            cid = c.get("chunk_id")
            if cid and cid not in seen:
                seen.add(cid)
                unique.append(c)
            if len(unique) >= cap:
                break
        return unique

    def _find_node(self, name: str) -> Optional[str]:
        if not name:
            return None
        name_lower = name.lower()
        # exact
        for node in self.graph.nodes():
            if node.lower() == name_lower:
                return node
        # contains
        for node in self.graph.nodes():
            if name_lower in node.lower() or node.lower() in name_lower:
                return node
        return None

    def get_all_nodes(self) -> List[str]:
        return list(self.graph.nodes())

    def get_stats(self) -> Dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": round(nx.density(self.graph), 4) if self.graph.number_of_nodes() > 1 else 0.0,
        }
