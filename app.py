import os
import json
import subprocess
import zipfile
import urllib.request
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import chromadb
from openai import OpenAI
import anthropic

# ---------------------------------------------------------
# FIX: Cloudflare firewall bypass helper
# ---------------------------------------------------------
def download_chroma_db(url: str, dest: str):
    """Download a file from R2 using browser-like headers to bypass Cloudflare 1010 firewall."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                "Version/18.0 Safari/605.1.15"
            ),
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        },
    )

    try:
        with urllib.request.urlopen(req) as response, open(dest, "wb") as out_file:
            out_file.write(response.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        print("HTTPError:", e.code)
        print("Response body snippet:", body[:500])
        raise


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")
CHROMA_DB_URL = "https://pub-28c95b6f026c497c908d911f7409ec0f.r2.dev/chroma_db.zip"
TOP_K = 10

AVAILABLE_COUNTRIES = [
    "United States", "United Kingdom", "China", "Russia", "Japan",
    "Spain", "Germany", "France", "Australia", "Canada", "India",
    "Taiwan", "Netherlands", "Sweden", "Jamaica", "South Korea",
    "Israel", "Singapore", "Brazil", "Mexico", "Poland", "Italy"
]

app = FastAPI(title="NSS Document Search")

openai_client = None
claude_client = None
collection = None


# ---------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    parsed_query: dict
    filters: dict


# ---------------------------------------------------------
# Startup
# ---------------------------------------------------------
@app.on_event("startup")
async def startup():
    global openai_client, claude_client, collection

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY environment variable")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY environment variable")

    # Download chroma_db if missing
    if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
        print("📥 Database not found. Downloading from R2...")
        zip_path = "/tmp/chroma_db.zip"

        print(f"   Downloading {CHROMA_DB_URL}...")
        download_chroma_db(CHROMA_DB_URL, zip_path)
        print("   Download complete!")

        print("   Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("   Extraction complete!")

        os.remove(zip_path)
        print("✓ Database ready!")

    openai_client = OpenAI()
    claude_client = anthropic.Anthropic()

    db = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = db.get_collection("nss_documents")
    print(f"✓ Loaded {collection.count():,} chunks")


# ---------------------------------------------------------
# Embeddings
# ---------------------------------------------------------
def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return response.data[0].embedding


# ---------------------------------------------------------
# DB Helpers
# ---------------------------------------------------------
def get_latest_year_for_country(country: str) -> int | None:
    results = collection.get(
        where={"country": country},
        include=["metadatas"]
    )
    if results["metadatas"]:
        years = [m.get("year") for m in results["metadatas"] if m.get("year")]
        if years:
            return max(years)
    return None


# ---------------------------------------------------------
# Query Parsing
# ---------------------------------------------------------
def understand_query(query: str) -> dict:
    response = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": query}],
        system=f"""You are a query parser for a National Security Strategy document database.

Extract structured filters...
(omitted here, unchanged)
"""
    )

    try:
        return json.loads(response.content[0].text)
    except json.JSONDecodeError:
        return {"search_query": query}


def build_filters(parsed: dict) -> dict:
    filters = {}

    if parsed.get("country"):
        filters["country"] = parsed["country"]
        if parsed.get("wants_latest"):
            latest_year = get_latest_year_for_country(parsed["country"])
            if latest_year:
                filters["year_min"] = latest_year
                filters["year_max"] = latest_year

    if parsed.get("year"):
        filters["year_min"] = parsed["year"]
        filters["year_max"] = parsed["year"]
    elif parsed.get("year_min") or parsed.get("year_max"):
        if parsed.get("year_min"):
            filters["year_min"] = parsed["year_min"]
        if parsed.get("year_max"):
            filters["year_max"] = parsed["year_max"]

    return filters


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------
def retrieve(query: str, filters: dict = None, n_results: int = TOP_K):
    query_embedding = get_embedding(query)

    where_filter = None
    if filters:
        conditions = []
        if "country" in filters:
            conditions.append({"country": filters["country"]})
        if "year_min" in filters:
            conditions.append({"year": {"$gte": filters["year_min"]}})
        if "year_max" in filters:
            conditions.append({"year": {"$lte": filters["year_max"]}})

        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter
    )

    return results


# ---------------------------------------------------------
# Formatting & Answer Generation
# ---------------------------------------------------------
def format_context(results) -> str:
    context_parts = []
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        context_parts.append(
            f"[Source {i+1}: {meta['country']} {meta['year']}, {meta['doc_name']} p.{meta['page']}]\n{doc}"
        )
    return "\n\n---\n\n".join(context_parts)


def generate_answer(query: str, context: str) -> str:
    system_prompt = """You are an expert analyst..."""

    user_prompt = f"""Based on the following excerpts...
{context}
"""

    response = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": user_prompt}],
        system=system_prompt
    )

    return response.content[0].text


# ---------------------------------------------------------
# API Endpoint
# ---------------------------------------------------------
@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    parsed = understand_query(request.query)
    filters = build_filters(parsed)
    search_query = parsed.get("search_query", request.query)

    results = retrieve(search_query, filters=filters if filters else None)

    if not results['documents'][0]:
        return QueryResponse(
            answer="No relevant documents found.",
            sources=[],
            parsed_query=parsed,
            filters=filters
        )

    context = format_context(results)
    answer = generate_answer(request.query, context)

    sources = [
        {
            "country": meta["country"],
            "year": meta["year"],
            "doc_name": meta["doc_name"],
            "page": meta["page"]
        }
        for meta in results['metadatas'][0]
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        parsed_query=parsed,
        filters=filters
    )


# ---------------------------------------------------------
# Root HTML
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    return """<!DOCTYPE html>
<html> ... (UNCHANGED) ...
</html>"""


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
