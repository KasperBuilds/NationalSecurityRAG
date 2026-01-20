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

openai_client = None
claude_client = None
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
    global openai_client, claude_client, collection
    
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY environment variable")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY environment variable")
    
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

def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
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
    response = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": query}],
        system=f"""You are a query parser for a National Security Strategy document database.

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
- "Compare China and Russia on cyber threats" → {{"country": null, "search_query": "China Russia cyber threats cybersecurity comparison"}}"""
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

    response = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": user_prompt}],
        system=system_prompt
    )
    
    return response.content[0].text

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
    all_meta = collection.get(include=["metadatas"])
    
    countries = {}
    years = set()
    
    for meta in all_meta["metadatas"]:
        country = meta.get("country", "Unknown")
        year = meta.get("year")
        
        if country not in countries:
            countries[country] = set()
        if year:
            countries[country].add(year)
            years.add(year)
    
    return {
        "total_chunks": collection.count(),
        "countries": len(countries),
        "country_list": sorted(countries.keys()),
        "year_range": [min(years), max(years)] if years else None
    }

@app.get("/about", response_class=HTMLResponse)
async def about_page():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>About - National Security Strategy Intelligence</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        :root {
            --bg: #0d1117;
            --bg-secondary: #161b22;
            --surface: #1c2128;
            --border: #30363d;
            --text: #e6edf3;
            --text-muted: #8b949e;
            --text-dim: #6e7681;
            --gold: #d4a853;
            --gold-dark: #b8923f;
            --gold-dim: rgba(212, 168, 83, 0.15);
            --green: #3fb950;
            --red: #f85149;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            line-height: 1.6;
        }
        
        /* Navigation */
        nav {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: rgba(13, 17, 23, 0.9);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border);
            z-index: 100;
            padding: 1rem 2rem;
        }
        
        .nav-content {
            max-width: 1000px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .nav-logo {
            font-weight: 700;
            font-size: 1rem;
            color: var(--text);
            text-decoration: none;
        }
        
        .nav-logo span { color: var(--gold); }
        
        .nav-links {
            display: flex;
            gap: 2rem;
        }
        
        .nav-links a {
            font-size: 0.85rem;
            color: var(--text-muted);
            text-decoration: none;
            transition: color 0.2s;
        }
        
        .nav-links a:hover { color: var(--gold); }
        .nav-links a.active { color: var(--gold); }
        
        .container { max-width: 1000px; margin: 0 auto; padding: 2rem; }
        
        /* About Section */
        #about {
            padding: 6rem 0 4rem;
        }
        
        .about-header {
            text-align: center;
            margin-bottom: 4rem;
        }
        
        .about-header h2 {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        
        .about-header p {
            color: var(--text-muted);
            max-width: 600px;
            margin: 0 auto;
        }
        
        /* Map Section */
        .map-section {
            margin-bottom: 4rem;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border);
        }
        
        .map-header {
            background: var(--surface);
            padding: 1.25rem 1.5rem;
            border-bottom: 1px solid var(--border);
        }
        
        .map-header h3 {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        
        .map-header p {
            font-size: 0.85rem;
            color: var(--text-muted);
        }
        
        #map {
            height: 400px;
            background: #a8d5f5;
        }
        
        .map-legend {
            background: var(--surface);
            padding: 1rem 1.5rem;
            border-top: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 1.5rem;
            flex-wrap: wrap;
        }
        
        .legend-title {
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-muted);
        }
        
        .legend-scale { display: flex; gap: 0.5rem; flex-wrap: wrap; }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.7rem;
            color: var(--text-muted);
        }
        
        .legend-color {
            width: 16px;
            height: 12px;
            border-radius: 2px;
        }
        
        /* Data Source */
        .data-source {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 4rem;
            text-align: center;
        }
        
        .data-source p {
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }
        
        .data-source a {
            color: var(--gold);
            text-decoration: none;
            font-weight: 500;
        }
        
        .data-source a:hover { text-decoration: underline; }
        
        .data-source .attribution {
            font-size: 0.8rem;
            color: var(--text-dim);
            margin-top: 0.75rem;
        }
        
        /* Comparison Section */
        .comparison-section { margin-bottom: 4rem; }
        
        .comparison-section h3 {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2.5rem;
        }
        
        .feature-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
        }
        
        .feature-icon {
            width: 40px;
            height: 40px;
            background: var(--gold-dim);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
            margin-bottom: 1rem;
        }
        
        .feature-card h4 { font-size: 0.95rem; font-weight: 600; margin-bottom: 0.5rem; }
        .feature-card p { font-size: 0.85rem; color: var(--text-muted); line-height: 1.6; }
        
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }
        
        .comparison-table th {
            text-align: left;
            padding: 1rem;
            background: var(--surface);
            border-bottom: 1px solid var(--border);
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }
        
        .comparison-table td { padding: 1rem; border-bottom: 1px solid var(--border); }
        .comparison-table td:first-child { color: var(--text-muted); }
        .check { color: var(--green); }
        .cross { color: var(--red); opacity: 0.8; }
        
        /* Footer */
        footer {
            padding: 2rem 0;
            border-top: 1px solid var(--border);
            text-align: center;
        }
        
        .footer-links { display: flex; justify-content: center; gap: 2rem; margin-bottom: 1rem; }
        .footer-links a {
            font-size: 0.8rem;
            color: var(--text-muted);
            text-decoration: none;
            transition: color 0.2s;
        }
        .footer-links a:hover { color: var(--gold); }
        .footer-credit { font-size: 0.75rem; color: var(--text-dim); }
        
        /* Leaflet */
        .leaflet-container { background: #a8d5f5; }
        .leaflet-control-zoom { border: 1px solid var(--border) !important; }
        .leaflet-control-zoom a {
            background: var(--surface) !important;
            color: var(--text) !important;
            border-bottom: 1px solid var(--border) !important;
        }
        .leaflet-control-zoom a:hover { background: var(--bg-secondary) !important; }
        
        .country-tooltip {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.5rem 0.75rem;
            font-family: 'Inter', sans-serif;
            font-size: 0.8rem;
            color: var(--text);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
        .country-tooltip strong { color: var(--gold); }
        
        @media (max-width: 640px) {
            nav { padding: 1rem; }
            .nav-links { gap: 1rem; }
            .container { padding: 1rem; }
            #map { height: 280px; }
            .legend-scale { gap: 0.25rem; }
        }
    </style>
</head>
<body>
    <nav>
        <div class="nav-content">
            <a href="/" class="nav-logo">NSS<span>Intel</span></a>
            <div class="nav-links">
                <a href="/about" class="active">About</a>
                <a href="https://github.com/KasperBuilds/NationalSecurityRAG" target="_blank">GitHub</a>
            </div>
        </div>
    </nav>

    <div class="container">
        <section id="about">
            <div class="about-header">
                <h2>About This Project</h2>
                <p>An AI-powered search engine for national security strategy documents, built with retrieval-augmented generation.</p>
            </div>
            
            <!-- World Map -->
            <div class="map-section">
                <div class="map-header">
                    <h3>Global Document Coverage</h3>
                    <p>820 National Security Strategy documents spanning 112 countries from 1962 to 2025</p>
                </div>
                <div id="map"></div>
                <div class="map-legend">
                    <span class="legend-title">Documents</span>
                    <div class="legend-scale">
                        <div class="legend-item"><div class="legend-color" style="background: #e8e0f0;"></div>0–1</div>
                        <div class="legend-item"><div class="legend-color" style="background: #d4c4e8;"></div>1–5</div>
                        <div class="legend-item"><div class="legend-color" style="background: #b9a3d4;"></div>5–10</div>
                        <div class="legend-item"><div class="legend-color" style="background: #9d7fc0;"></div>10–15</div>
                        <div class="legend-item"><div class="legend-color" style="background: #7c5aa8;"></div>15–20</div>
                        <div class="legend-item"><div class="legend-color" style="background: #5c3d8a;"></div>20–25</div>
                        <div class="legend-item"><div class="legend-color" style="background: #4a2c72;"></div>25+</div>
                    </div>
                </div>
            </div>
            
            <!-- Why RAG Section -->
            <div class="comparison-section">
                <h3>Why RAG Over Standard LLMs?</h3>
                
                <div class="features-grid">
                    <div class="feature-card">
                        <div class="feature-icon">🎯</div>
                        <h4>Source-Grounded</h4>
                        <p>Every claim links to specific documents and page numbers you can verify.</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">🔍</div>
                        <h4>Semantic Search</h4>
                        <p>Finds conceptually relevant passages, not just keyword matches.</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">🧠</div>
                        <h4>Smart Filtering</h4>
                        <p>Extracts country, year, and intent from natural language queries.</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">📊</div>
                        <h4>Cross-Document</h4>
                        <p>Compare strategies across countries and time periods.</p>
                    </div>
                </div>
                
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Capability</th>
                            <th>This System (RAG)</th>
                            <th>ChatGPT / Claude</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Page-level citations</td>
                            <td><span class="check">✓ Always provided</span></td>
                            <td><span class="cross">✗ Often fabricated</span></td>
                        </tr>
                        <tr>
                            <td>2025 documents</td>
                            <td><span class="check">✓ Included</span></td>
                            <td><span class="cross">✗ Training cutoff</span></td>
                        </tr>
                        <tr>
                            <td>Verifiable claims</td>
                            <td><span class="check">✓ Check the source</span></td>
                            <td><span class="cross">✗ Trust required</span></td>
                        </tr>
                        <tr>
                            <td>Hallucination risk</td>
                            <td><span class="check">✓ Minimal</span></td>
                            <td><span class="cross">✗ Significant</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <!-- Data Source Attribution -->
            <div class="data-source">
                <p>Document data sourced from</p>
                <a href="https://militarydoctrines.com/" target="_blank">militarydoctrines.com</a>
                <p class="attribution">
                    The Military Doctrines project is led by <strong>J Andrés Gannon</strong>, 
                    Assistant Professor of Political Science at <strong>Vanderbilt University</strong>.
                </p>
            </div>
        </section>
        
        <footer>
            <div class="footer-links">
                <a href="https://github.com/KasperBuilds/NationalSecurityRAG" target="_blank">GitHub</a>
                <a href="https://militarydoctrines.com/" target="_blank">Data Source</a>
            </div>
            <p class="footer-credit">Built with ChromaDB · Kasper Hong</p>
        </footer>
    </div>
    
    <script>
        const docCounts = {
            "Albania": 11, "Argentina": 4, "Armenia": 2, "Aruba": 1, "Australia": 13,
            "Austria": 10, "Azerbaijan": 1, "Belarus": 2, "Belgium": 5, "Belize": 2,
            "Bermuda": 1, "Bolivia": 3, "Bosnia": 3, "Brazil": 8, "Brunei": 3,
            "Bulgaria": 8, "Burkina Faso": 1, "Cambodia": 4, "Canada": 5,
            "Central African Republic": 2, "Chile": 4, "China": 11, "Colombia": 6,
            "Cook Islands": 1, "Costa Rica": 2, "Croatia": 7, "Czech Republic": 16,
            "Czechia": 16, "Denmark": 11, "Dominican Republic": 1, "Ecuador": 7, 
            "El Salvador": 2, "Estonia": 10, "Ethiopia": 2, "Finland": 10, "France": 7, 
            "Gambia": 1, "Georgia": 8, "Germany": 9, "Greece": 4, "Guatemala": 7, 
            "Guyana": 1, "Haiti": 1, "Honduras": 1, "Hungary": 7, "Iceland": 1, 
            "India": 17, "Indonesia": 2, "Iraq": 1, "Ireland": 8, "Israel": 1, 
            "Italy": 8, "Jamaica": 3, "Japan": 21, "Kenya": 1, "Kyrgyzstan": 1, 
            "Latvia": 12, "Lebanon": 1, "Liberia": 2, "Lithuania": 13, "Luxembourg": 5, 
            "Malaysia": 2, "Maldives": 1, "Malta": 2, "Mexico": 5, "Moldova": 3, 
            "Mongolia": 3, "Montenegro": 4, "Nepal": 1, "Netherlands": 8, 
            "New Zealand": 11, "Nicaragua": 1, "Niger": 1, "Nigeria": 1, 
            "North Macedonia": 4, "Norway": 12, "Pakistan": 3, "Palau": 1, 
            "Papua New Guinea": 1, "Paraguay": 1, "Peru": 2, "Philippines": 4, 
            "Poland": 9, "Portugal": 4, "Romania": 9, "Russia": 11, "Rwanda": 1, 
            "Samoa": 1, "Serbia": 4, "Sierra Leone": 1, "Singapore": 2, "Slovakia": 7, 
            "Slovenia": 13, "Solomon Islands": 1, "South Africa": 6, "South Korea": 12,
            "Republic of Korea": 12, "Spain": 10, "Sweden": 5, "Switzerland": 5, 
            "Taiwan": 23, "Tanzania": 1, "Thailand": 2, "Timor-Leste": 1, 
            "Trinidad and Tobago": 1, "Turkey": 3, "Uganda": 1, "Ukraine": 13, 
            "United Kingdom": 18, "Uruguay": 4, "United States": 21, 
            "United States of America": 21, "Vanuatu": 1, "Vietnam": 2
        };
        
        function getColor(count) {
            if (count >= 25) return '#4a2c72';
            if (count >= 20) return '#5c3d8a';
            if (count >= 15) return '#7c5aa8';
            if (count >= 10) return '#9d7fc0';
            if (count >= 5) return '#b9a3d4';
            if (count >= 1) return '#d4c4e8';
            return '#e8e0f0';
        }
        
        function getCountryCount(name) {
            return docCounts[name] || 0;
        }
        
        // Initialize map immediately on about page
        function initMap() {
            const map = L.map('map', {
                center: [30, 0],
                zoom: 2,
                minZoom: 1,
                maxZoom: 6
            });
            
            L.tileLayer('https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png', {
                attribution: ''
            }).addTo(map);
            
            fetch('https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson')
                .then(res => res.json())
                .then(data => {
                    L.geoJSON(data, {
                        style: function(feature) {
                            const name = feature.properties.ADMIN || feature.properties.name;
                            const count = getCountryCount(name);
                            return {
                                fillColor: getColor(count),
                                weight: 0.5,
                                opacity: 1,
                                color: '#fff',
                                fillOpacity: 0.85
                            };
                        },
                        onEachFeature: function(feature, layer) {
                            const name = feature.properties.ADMIN || feature.properties.name;
                            const count = getCountryCount(name);
                            if (count > 0) {
                                layer.bindTooltip(
                                    '<strong>' + name + '</strong><br>' + count + ' document' + (count > 1 ? 's' : ''),
                                    { className: 'country-tooltip', sticky: true }
                                );
                            }
                            layer.on({
                                mouseover: function(e) { e.target.setStyle({ weight: 2, color: '#d4a853' }); },
                                mouseout: function(e) { e.target.setStyle({ weight: 0.5, color: '#fff' }); }
                            });
                        }
                    }).addTo(map);
                });
        }
        
        // Initialize map when page loads
        window.addEventListener('DOMContentLoaded', initMap);
    </script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>National Security Strategy Intelligence</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --bg: #0d1117;
            --bg-secondary: #161b22;
            --surface: #1c2128;
            --border: #30363d;
            --text: #e6edf3;
            --text-muted: #8b949e;
            --text-dim: #6e7681;
            --gold: #d4a853;
            --gold-dark: #b8923f;
            --gold-dim: rgba(212, 168, 83, 0.15);
            --green: #3fb950;
            --red: #f85149;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            line-height: 1.6;
        }
        
        /* Navigation */
        nav {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: rgba(13, 17, 23, 0.9);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border);
            z-index: 100;
            padding: 1rem 2rem;
        }
        
        .nav-content {
            max-width: 1000px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .nav-logo {
            font-weight: 700;
            font-size: 1rem;
            color: var(--text);
            text-decoration: none;
        }
        
        .nav-logo span { color: var(--gold); }
        
        .nav-links {
            display: flex;
            gap: 2rem;
        }
        
        .nav-links a {
            font-size: 0.85rem;
            color: var(--text-muted);
            text-decoration: none;
            transition: color 0.2s;
        }
        
        .nav-links a:hover { color: var(--gold); }
        
        .container { max-width: 1000px; margin: 0 auto; padding: 2rem; }
        
        /* Hero Section */
        .hero {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            text-align: center;
            padding: 6rem 0 4rem;
        }
        
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: var(--gold-dim);
            border: 1px solid var(--gold);
            border-radius: 100px;
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--gold);
            margin-bottom: 2rem;
        }
        
        .badge::before {
            content: '';
            width: 6px;
            height: 6px;
            background: var(--gold);
            border-radius: 50%;
        }
        
        .hero h1 {
            font-size: clamp(2.5rem, 6vw, 4rem);
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 1.5rem;
            letter-spacing: -0.02em;
        }
        
        .hero h1 span { color: var(--gold); display: block; }
        
        .hero-subtitle {
            font-size: 1.1rem;
            color: var(--text-muted);
            max-width: 550px;
            margin: 0 auto 3rem;
            line-height: 1.7;
        }
        
        /* Search Box */
        .search-container {
            width: 100%;
            max-width: min(900px, 95%);
            margin: 0 auto 2rem;
        }
        
        .search-box {
            display: flex;
            align-items: center;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.5rem;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        
        .search-box:focus-within {
            border-color: var(--gold);
            box-shadow: 0 0 0 3px var(--gold-dim);
        }
        
        .search-icon { padding: 0 1rem; color: var(--text-muted); }
        .search-icon svg { width: 20px; height: 20px; }
        
        #query-input {
            flex: 1;
            padding: 1rem 0.5rem;
            font-family: inherit;
            font-size: 1rem;
            background: transparent;
            border: none;
            color: var(--text);
            outline: none;
        }
        
        #query-input::placeholder { color: var(--text-dim); }
        
        .search-btn {
            padding: 0.875rem 1.5rem;
            background: var(--gold);
            color: var(--bg);
            border: none;
            border-radius: 8px;
            font-family: inherit;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: background 0.2s;
        }
        
        .search-btn:hover { background: var(--gold-dark); }
        .search-btn svg { width: 16px; height: 16px; }
        
        .examples {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 0.5rem;
        }
        
        .example-btn {
            padding: 0.5rem 1rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 100px;
            color: var(--text-muted);
            font-family: inherit;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .example-btn:hover { border-color: var(--gold); color: var(--gold); }
        
        /* Loading */
        #loading { display: none; padding: 3rem 0; }
        #loading.active { display: block; }
        
        .loading-container { max-width: 500px; margin: 0 auto; text-align: center; }
        
        .loading-spinner {
            width: 48px;
            height: 48px;
            border: 3px solid var(--border);
            border-top-color: var(--gold);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 1.5rem;
        }
        
        @keyframes spin { to { transform: rotate(360deg); } }
        
        .loading-text { 
            color: var(--text-muted); 
            font-size: 0.9rem;
            min-height: 1.5em;
        }
        
        /* Results */
        #results { display: none; padding: 2rem 0; }
        #results.active { display: block; }
        
        .result-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem 1.25rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            margin-bottom: 1.5rem;
            font-size: 0.8rem;
            color: var(--text-muted);
            flex-wrap: wrap;
        }
        
        .result-header code {
            background: var(--bg);
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
        }
        
        .answer-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 1.5rem;
        }
        
        .answer-card h2, .answer-card h3 { font-size: 1.1rem; margin: 1.5rem 0 0.75rem; font-weight: 600; }
        .answer-card h2:first-child, .answer-card h3:first-child { margin-top: 0; }
        .answer-card p { margin-bottom: 1rem; color: var(--text); }
        .answer-card ul, .answer-card ol { margin: 1rem 0; padding-left: 1.5rem; }
        .answer-card li { margin-bottom: 0.5rem; }
        .answer-card strong { color: #fff; font-weight: 600; }
        .answer-card code {
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.85em;
            background: var(--bg);
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
        }
        
        .sources-section {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.25rem;
        }
        
        .sources-title {
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-dim);
            margin-bottom: 0.75rem;
        }
        
        .sources-list { display: flex; flex-wrap: wrap; gap: 0.5rem; }
        
        .source-tag {
            font-size: 0.75rem;
            padding: 0.4rem 0.75rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-muted);
        }
        
        /* About Section */
        #about {
            padding: 6rem 0;
        }
        
        .about-header {
            text-align: center;
            margin-bottom: 4rem;
        }
        
        .about-header h2 {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        
        .about-header p {
            color: var(--text-muted);
            max-width: 600px;
            margin: 0 auto;
        }
        
        /* Map Section */
        .map-section {
            margin-bottom: 4rem;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border);
        }
        
        .map-header {
            background: var(--surface);
            padding: 1.25rem 1.5rem;
            border-bottom: 1px solid var(--border);
        }
        
        .map-header h3 {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        
        .map-header p {
            font-size: 0.85rem;
            color: var(--text-muted);
        }
        
        #map {
            height: 400px;
            background: #a8d5f5;
        }
        
        .map-legend {
            background: var(--surface);
            padding: 1rem 1.5rem;
            border-top: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 1.5rem;
            flex-wrap: wrap;
        }
        
        .legend-title {
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-muted);
        }
        
        .legend-scale { display: flex; gap: 0.5rem; flex-wrap: wrap; }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.7rem;
            color: var(--text-muted);
        }
        
        .legend-color {
            width: 16px;
            height: 12px;
            border-radius: 2px;
        }
        
        /* Data Source */
        .data-source {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 4rem;
            text-align: center;
        }
        
        .data-source p {
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }
        
        .data-source a {
            color: var(--gold);
            text-decoration: none;
            font-weight: 500;
        }
        
        .data-source a:hover { text-decoration: underline; }
        
        .data-source .attribution {
            font-size: 0.8rem;
            color: var(--text-dim);
            margin-top: 0.75rem;
        }
        
        /* Comparison Section */
        .comparison-section { margin-bottom: 4rem; }
        
        .comparison-section h3 {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2.5rem;
        }
        
        .feature-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
        }
        
        .feature-icon {
            width: 40px;
            height: 40px;
            background: var(--gold-dim);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
            margin-bottom: 1rem;
        }
        
        .feature-card h4 { font-size: 0.95rem; font-weight: 600; margin-bottom: 0.5rem; }
        .feature-card p { font-size: 0.85rem; color: var(--text-muted); line-height: 1.6; }
        
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }
        
        .comparison-table th {
            text-align: left;
            padding: 1rem;
            background: var(--surface);
            border-bottom: 1px solid var(--border);
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }
        
        .comparison-table td { padding: 1rem; border-bottom: 1px solid var(--border); }
        .comparison-table td:first-child { color: var(--text-muted); }
        .check { color: var(--green); }
        .cross { color: var(--red); opacity: 0.8; }
        
        /* Footer */
        footer {
            padding: 2rem 0;
            border-top: 1px solid var(--border);
            text-align: center;
        }
        
        .footer-links { display: flex; justify-content: center; gap: 2rem; margin-bottom: 1rem; }
        .footer-links a {
            font-size: 0.8rem;
            color: var(--text-muted);
            text-decoration: none;
            transition: color 0.2s;
        }
        .footer-links a:hover { color: var(--gold); }
        .footer-credit { font-size: 0.75rem; color: var(--text-dim); }
        
        /* Leaflet */
        .leaflet-container { background: #a8d5f5; }
        .leaflet-control-zoom { border: 1px solid var(--border) !important; }
        .leaflet-control-zoom a {
            background: var(--surface) !important;
            color: var(--text) !important;
            border-bottom: 1px solid var(--border) !important;
        }
        .leaflet-control-zoom a:hover { background: var(--bg-secondary) !important; }
        
        .country-tooltip {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.5rem 0.75rem;
            font-family: 'Inter', sans-serif;
            font-size: 0.8rem;
            color: var(--text);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
        .country-tooltip strong { color: var(--gold); }
        
        @media (max-width: 640px) {
            nav { padding: 1rem; }
            .nav-links { gap: 1rem; }
            .container { padding: 1rem; }
            .hero { padding: 5rem 0 3rem; min-height: auto; }
            .hero h1 { font-size: 2rem; }
            .search-btn span { display: none; }
            #map { height: 280px; }
            .legend-scale { gap: 0.25rem; }
        }
    </style>
</head>
<body>
    <nav>
        <div class="nav-content">
            <a href="/" class="nav-logo">NSS<span>Intel</span></a>
            <div class="nav-links">
                <a href="/about">About</a>
                <a href="https://github.com/KasperBuilds/NationalSecurityRAG" target="_blank">GitHub</a>
            </div>
        </div>
    </nav>

    <div class="container">
        <!-- Main Search Section -->
        <section class="hero" id="home">
            <div class="badge">RAG-Powered Analysis</div>
            <h1>
                National Security
                <span>Strategy Intelligence</span>
            </h1>
            <p class="hero-subtitle">
                Query national security documents with retrieval-augmented generation. 
                Sourced answers, not hallucinations.
            </p>
            
            <div class="search-container">
                <div class="search-box">
                    <div class="search-icon">
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
                        </svg>
                    </div>
                    <input type="text" id="query-input" placeholder="Ask about national security strategies..." autocomplete="off">
                    <button class="search-btn" onclick="submitQuery()">
                        <span>Search</span>
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/>
                        </svg>
                    </button>
                </div>
            </div>
            
            <div class="examples">
                <button class="example-btn" onclick="setQuery('What is the latest US NSS about?')">Latest US Strategy</button>
                <button class="example-btn" onclick="setQuery('How does Japan view China as a threat?')">Japan on China</button>
                <button class="example-btn" onclick="setQuery('Compare cyber security approaches')">Cyber Comparison</button>
                <button class="example-btn" onclick="setQuery('NATO in European strategies')">NATO Evolution</button>
            </div>
        </section>
        
        <div id="loading">
            <div class="loading-container">
                <div class="loading-spinner"></div>
                <div class="loading-text" id="loading-text"></div>
            </div>
        </div>
        
        <div id="results">
            <div class="result-header">
                <span>Query:</span>
                <code id="parsed-query"></code>
                <span id="filters-display"></span>
            </div>
            <div class="answer-card" id="answer"></div>
            <div class="sources-section">
                <div class="sources-title">Sources Referenced</div>
                <div class="sources-list" id="sources"></div>
            </div>
        </div>
        
        <footer>
            <div class="footer-links">
                <a href="https://github.com/KasperBuilds/NationalSecurityRAG" target="_blank">GitHub</a>
                <a href="https://militarydoctrines.com/" target="_blank">Data Source</a>
            </div>
            <p class="footer-credit">Built with ChromaDB · Kasper Hong</p>
        </footer>
    </div>
    
    <script>
        // Search functionality
        const input = document.getElementById('query-input');
        const loading = document.getElementById('loading');
        const loadingText = document.getElementById('loading-text');
        const results = document.getElementById('results');
        const answerDiv = document.getElementById('answer');
        const sourcesDiv = document.getElementById('sources');
        const parsedQueryEl = document.getElementById('parsed-query');
        const filtersDisplay = document.getElementById('filters-display');
        
        const actionVerbs = [
            'Analyzing',
            'Searching',
            'Processing',
            'Retrieving',
            'Examining',
            'Evaluating',
            'Synthesizing',
            'Compiling',
            'Reviewing',
            'Organizing'
        ];
        
        let verbInterval = null;
        let verbIndex = 0;
        
        function startVerbCycle() {
            verbIndex = 0;
            loadingText.textContent = actionVerbs[verbIndex] + '...';
            
            verbInterval = setInterval(() => {
                verbIndex = (verbIndex + 1) % actionVerbs.length;
                loadingText.textContent = actionVerbs[verbIndex] + '...';
            }, 4000);
        }
        
        function stopVerbCycle() {
            if (verbInterval) {
                clearInterval(verbInterval);
                verbInterval = null;
            }
        }
        
        function setQuery(q) {
            input.value = q;
            submitQuery();
        }
        
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') submitQuery();
        });
        
        async function submitQuery() {
            const query = input.value.trim();
            if (!query) return;
            
            loading.classList.add('active');
            results.classList.remove('active');
            startVerbCycle();
            
            // Scroll to loading area immediately
            setTimeout(() => {
                window.scrollTo({ top: loading.offsetTop - 100, behavior: 'smooth' });
            }, 100);
            
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query })
                });
                
                if (!response.ok) throw new Error('No API credits left!');
                
                const data = await response.json();
                
                stopVerbCycle();
                await new Promise(r => setTimeout(r, 200));
                
                parsedQueryEl.textContent = data.parsed_query.search_query || query;
                filtersDisplay.textContent = Object.keys(data.filters).length > 0 
                    ? '| Filters: ' + JSON.stringify(data.filters) 
                    : '';
                
                answerDiv.innerHTML = marked.parse(data.answer);
                sourcesDiv.innerHTML = data.sources.map(s => 
                    '<span class="source-tag">' + s.country + ' (' + s.year + ') p.' + s.page + '</span>'
                ).join('');
                
                loading.classList.remove('active');
                results.classList.add('active');
                
                window.scrollTo({ top: results.offsetTop - 100, behavior: 'smooth' });
                
            } catch (err) {
                stopVerbCycle();
                loading.classList.remove('active');
                answerDiv.innerHTML = '<p style="color: var(--red);">Error: ' + err.message + '</p>';
                results.classList.add('active');
            }
        }
    </script>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
