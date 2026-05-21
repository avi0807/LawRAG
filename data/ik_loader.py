import os
import json
import time
import requests
import re
from typing import List, Dict
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()

console = Console()

INDIANKANOON_BASE = "https://api.indiankanoon.org"

LANDMARK_CASE_IDS = [
    "1766147",   # Maneka Gandhi v Union of India
    "257876",    # Kesavananda Bharati v State of Kerala
    "1857950",   # A.K. Gopalan v State of Madras
    "1939993",   # Minerva Mills v Union of India
    "1765180",   # S.R. Bommai v Union of India
    "1233519",   # Vishaka v State of Rajasthan
    "13107",     # MC Mehta v Union of India
    "709776",    # Indra Sawhney v Union of India
    "195901",    # ADM Jabalpur v Shivkant Shukla
    "186701",    # Olga Tellis v Bombay Municipal Corporation
    "261785",    # State of Madras v Champakam Dorairajan
    "1475152",   # Bachan Singh v State of Punjab
    "1418990",   # Hussainara Khatoon v State of Bihar
    "75151",     # Mohd Ahmed Khan v Shah Bano Begum
    "220807",    # Consumer Education Research Centre v Union of India
]

SEARCH_QUERIES = [
    "fundamental rights article 21",
    "right to equality article 14",
    "freedom of speech article 19",
    "directive principles state policy",
    "judicial review constitutional validity",
    "right to education article 21A",
    "environmental law public interest",
    "criminal procedure bail anticipatory",
]


def clean_text(raw: str) -> str:
    clean = re.sub(r'<[^>]+>', ' ', raw)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


def fetch_document(doc_id: str, session: requests.Session) -> Dict | None:
    try:
        url = f"{INDIANKANOON_BASE}/doc/{doc_id}/"
        res = session.post(url, timeout=15)
        if res.status_code == 200:
            return res.json()
        console.print(f"  [yellow]⚠ Doc {doc_id} returned {res.status_code}[/yellow]")
        return None
    except Exception as e:
        console.print(f"  [yellow]⚠ Doc {doc_id} failed: {e}[/yellow]")
        return None


def search_cases(query: str, session: requests.Session, max_results: int = 5) -> List[str]:
    try:
        url = f"{INDIANKANOON_BASE}/search/"
        res = session.post(url, data={"formInput": query, "pagenum": 0}, timeout=15)
        if res.status_code == 200:
            data = res.json()
            docs = data.get("docs", [])
            return [str(d["tid"]) for d in docs[:max_results] if "tid" in d]
    except Exception as e:
        console.print(f"  [yellow]⚠ Search failed for '{query}': {e}[/yellow]")
    return []


def load_constitution(pdf_path: str = "data/constitution.pdf") -> List[Dict]:
    """
    Extract text from the Constitution PDF and split by Part/Article
    so each chunk is a meaningful legal unit rather than an arbitrary page.
    """
    if not os.path.exists(pdf_path):
        console.print(f"[yellow]⚠ Constitution PDF not found at {pdf_path} — skipping[/yellow]")
        return []

    console.print("[bold]Extracting Constitution of India...[/bold]")

    try:
        from pypdf import PdfReader
    except ImportError:
        console.print("[red]pypdf not installed. Run: pip install pypdf[/red]")
        return []

    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    console.print(f"  Extracted {len(full_text):,} characters from {len(reader.pages)} pages")

    # Split by Article boundaries
    # Matches "Article 1.", "Article 21A.", "Article 370." etc.
    article_pattern = re.compile(r'(?=Article\s+\d+[A-Z]?\.)', re.IGNORECASE)
    raw_articles = article_pattern.split(full_text)

    documents = []
    for i, chunk in enumerate(raw_articles):
        chunk = chunk.strip()
        if len(chunk) < 80:
            continue

        # Extract article number for the title
        match = re.match(r'Article\s+(\d+[A-Z]?)\.?\s*(.*?)(?:\n|\.)', chunk, re.IGNORECASE)
        if match:
            article_num = match.group(1)
            article_title = match.group(2).strip()[:80]
            title = f"Article {article_num} — {article_title}" if article_title else f"Article {article_num}"
        else:
            title = f"Constitution — Section {i}"

        doc = {
            "id": f"constitution_article_{i:03d}",
            "title": title,
            "content": chunk[:3000],   # cap per article
            "source": "constitution_of_india",
            "year": 1950,
            "court": "Parliament of India",
            "citations": [],
            "url": "https://legislative.gov.in",
        }
        documents.append(doc)

    console.print(f"[green]✓ Extracted {len(documents)} Articles from Constitution[/green]")
    return documents


def fetch_indian_kanoon(
    max_cases: int = 30,
    cache_dir: str = "data/ik_cache",
) -> List[Dict]:
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "cases.json")

    if os.path.exists(cache_file):
        console.print("[dim]Loading Indian Kanoon cases from cache...[/dim]")
        with open(cache_file, "r") as f:
            return json.load(f)

    console.print("[bold]Fetching Indian Supreme Court judgments...[/bold]")

    session = requests.Session()
    token=os.environ.get("INDIANKANOON_API_TOKEN","")
    if not token:
        console.print("[red] INDIANKANOON_API_TOKEN not set in .env[/red]")
    session.headers.update({"User-Agent": "NyayRAG/1.0 (legal research project)",
     "Authorization": f"Token {token}", 
     })

    all_ids = list(LANDMARK_CASE_IDS)

    console.print("[dim]Searching for additional cases...[/dim]")
    for query in SEARCH_QUERIES:
        ids = search_cases(query, session, max_results=3)
        all_ids.extend(ids)
        time.sleep(1)

    all_ids = list(dict.fromkeys(all_ids))[:max_cases]
    console.print(f"[dim]Fetching {len(all_ids)} cases...[/dim]")

    documents = []
    for i, doc_id in enumerate(all_ids):
        data = fetch_document(doc_id, session)
        if not data:
            time.sleep(2)
            continue

        title = data.get("title", f"Case {doc_id}")
        raw_text = data.get("doc", data.get("judgment", ""))
        court = data.get("courtName", "Supreme Court of India")
        date = data.get("judgmentDate", "")
        citations = data.get("citedDocs", [])

        text = clean_text(raw_text)
        if len(text) < 100:
            continue

        text = text[800:9000]

        doc = {
            "id": f"ik_{doc_id}",
            "title": title,
            "content": f"CASE: {title}\n\nJUDGMENT: {text[1000:9000]}",
            "source": "indian_kanoon",
            "year": int(date[:4]) if date and len(date) >= 4 else 2000,
            "court": court,
            "citations": [str(c) for c in citations[:10]],
            "url": f"https://indiankanoon.org/doc/{doc_id}/",
        }
        documents.append(doc)

        console.print(f"  [{i+1}/{len(all_ids)}] {title[:70]}... ({len(text)} chars)")
        time.sleep(1)

    console.print(f"[green]✓ Fetched {len(documents)} judgments[/green]")

    with open(cache_file, "w") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    return documents


def load_all_documents() -> List[Dict]:
    """
    Merge Indian Kanoon judgments + Constitution of India into one dataset.
    Constitution is NOT cached separately — it loads fresh from PDF each time
    since PDF reading is fast (a few seconds).
    """
    judgments = fetch_indian_kanoon()
    constitution = load_constitution()

    total = judgments + constitution
    console.print(
        f"[bold green]✓ Total dataset: {len(judgments)} judgments + "
        f"{len(constitution)} constitutional articles = {len(total)} documents[/bold green]"
    )
    return total


# Drop-in replacement
SAMPLE_DOCUMENTS = load_all_documents()