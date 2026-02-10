import os
from pathlib import Path
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

# Config
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

openrouter_client = None
collection = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    parsed_query: dict
    filters: dict

def download_chroma_db(url: str, dest: str):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        },
    )
    with urllib.request.urlopen(req) as response, open(dest, "wb") as out_file:
        out_file.write(response.read())

@app.on_event("startup")
async def startup():
    global openrouter_client, collection
    
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("Set OPENROUTER_API_KEY environment variable")
    
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
    
    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY")
    )
    
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = db.get_collection("nss_documents")
    print(f"✓ Loaded {collection.count():,} chunks")

def get_embedding(text: str) -> list[float]:
    response = openrouter_client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=[text]
    )
    return response.data[0].embedding

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

def understand_query(query: str) -> dict:
    response = openrouter_client.chat.completions.create(
        model="anthropic/claude-3.5-sonnet",
        max_tokens=300,
        messages=[
            {"role": "system", "content": f"""You are a query parser for a National Security Strategy document database.

Extract structured filters from the user's question. Return ONLY valid JSON, no other text.

Available countries: {', '.join(AVAILABLE_COUNTRIES)}

Return format:
{{
    "country": "Country Name" or null,
    "year": 2020 or null,
    "year_min": 2000 or null,
    "year_max": 2020 or null,
    "wants_latest": true/false,
    "search_query": "the core question to search for"
}}

Rules:
- "wants_latest": true if user says "latest", "most recent", "current", "newest"
- If user mentions a specific year, set "year" to that
- If user mentions a range like "2000-2020" or "since 2015", use year_min/year_max
- "search_query": rephrase as a good semantic search query (remove country/year references, focus on the TOPIC)
- Normalize country names to match the available list exactly

Examples:
- "What is the latest US NSS about?" → {{"country": "United States", "wants_latest": true, "search_query": "main themes priorities objectives strategy overview"}}
- "How has Japan's defense strategy evolved from 2010 to 2020?" → {{"country": "Japan", "year_min": 2010, "year_max": 2020, "search_query": "defense strategy evolution changes"}}
- "Compare China and Russia on cyber threats" → {{"country": null, "search_query": "China Russia cyber threats cybersecurity comparison"}}"""},
            {"role": "user", "content": query}
        ]
    )
    
    try:
        return json.loads(response.choices[0].message.content)
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

def format_context(results) -> str:
    context_parts = []
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        context_parts.append(
            f"[Source {i+1}: {meta['country']} {meta['year']}, {meta['doc_name']} p.{meta['page']}]\n{doc}"
        )
    return "\n\n---\n\n".join(context_parts)

def generate_answer(query: str, context: str) -> str:
    system_prompt = """You are an expert analyst of National Security Strategy documents. 
You answer questions based on the provided document excerpts.

Guidelines:
- Cite specific countries and years when making claims
- If comparing countries, be specific about similarities and differences  
- If the provided context doesn't contain enough information, say so
- Be concise but thorough
- Use markdown formatting for readability"""

    user_prompt = f"""Based on the following excerpts from National Security Strategy documents, answer this question:

**Question:** {query}

**Document Excerpts:**
{context}

Provide a well-structured answer with specific citations to the source documents."""

    response = openrouter_client.chat.completions.create(
        model="anthropic/claude-3.5-sonnet",
        max_tokens=1500,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    return response.choices[0].message.content

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
            answer="No relevant documents found for your query. Try broadening your search or checking the country/year filters.",
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

@app.get("/api/stats")
async def get_stats():
    countries = {}
    years = set()
    
    total = collection.count()
    batch_size = 5000
    
    for offset in range(0, total, batch_size):
        batch = collection.get(
            include=["metadatas"],
            limit=batch_size,
            offset=offset
        )
        for meta in batch["metadatas"]:
            country = meta.get("country", "Unknown")
            year = meta.get("year")
            
            if country not in countries:
                countries[country] = set()
            if year:
                countries[country].add(year)
                years.add(year)
    
    return {
        "total_chunks": total,
        "countries": len(countries),
        "country_list": sorted(countries.keys()),
        "year_range": [min(years), max(years)] if years else None
    }

@app.get("/about", response_class=HTMLResponse)
async def about_page():
    return (Path(__file__).parent / "templates" / "about.html").read_text()

@app.get("/", response_class=HTMLResponse)
async def root():
    return (Path(__file__).parent / "templates" / "index.html").read_text()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

