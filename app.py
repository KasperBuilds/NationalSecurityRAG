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

# Initialize clients
app = FastAPI(title="NSS Document Search")

# Global clients (initialized on startup)
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

@app.on_event("startup")
async def startup():
    global openai_client, claude_client, collection
    
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY environment variable")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY environment variable")
    
    # Download chroma_db if not present
    if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
        print("📥 Database not found. Downloading from R2...")
        zip_path = "/tmp/chroma_db.zip"
        
        # Download
        print(f"   Downloading {CHROMA_DB_URL}...")
        urllib.request.urlretrieve(CHROMA_DB_URL, zip_path)
        print("   Download complete!")
        
        # Extract
        print("   Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("   Extraction complete!")
        
        # Cleanup
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
    
    # Step 1: Understand the query
    parsed = understand_query(request.query)
    filters = build_filters(parsed)
    search_query = parsed.get("search_query", request.query)
    
    # Step 2: Retrieve relevant chunks
    results = retrieve(search_query, filters=filters if filters else None)
    
    if not results['documents'][0]:
        return QueryResponse(
            answer="No relevant documents found for your query. Try broadening your search or checking the country/year filters.",
            sources=[],
            parsed_query=parsed,
            filters=filters
        )
    
    # Step 3: Generate answer
    context = format_context(results)
    answer = generate_answer(request.query, context)
    
    # Format sources
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
    """Get database statistics"""
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

@app.get("/", response_class=HTMLResponse)
async def root():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NSS Document Search</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --bg: #0a0a0a;
            --surface: #141414;
            --border: #2a2a2a;
            --text: #e5e5e5;
            --text-muted: #737373;
            --accent: #3b82f6;
            --accent-dim: #1e3a5f;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Source Serif 4', Georgia, serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            line-height: 1.6;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        header {
            padding: 3rem 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 2rem;
        }
        
        h1 {
            font-size: 1.5rem;
            font-weight: 600;
            letter-spacing: -0.02em;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.8rem;
            color: var(--text-muted);
        }
        
        .search-box {
            position: relative;
            margin-bottom: 2rem;
        }
        
        #query-input {
            width: 100%;
            padding: 1rem 1.25rem;
            font-family: inherit;
            font-size: 1rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        
        #query-input:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-dim);
        }
        
        #query-input::placeholder {
            color: var(--text-muted);
        }
        
        .examples {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.75rem;
        }
        
        .examples span {
            cursor: pointer;
            padding: 0.25rem 0.5rem;
            background: var(--surface);
            border-radius: 4px;
            margin-right: 0.5rem;
            transition: color 0.2s;
        }
        
        .examples span:hover {
            color: var(--accent);
        }
        
        #loading {
            display: none;
            padding: 2rem;
        }
        
        #loading.active {
            display: block;
        }
        
        .loading-container {
            max-width: 400px;
            margin: 0 auto;
        }
        
        .loading-status {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.85rem;
            color: var(--text);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .loading-icon {
            width: 20px;
            height: 20px;
            position: relative;
        }
        
        .loading-icon::before,
        .loading-icon::after {
            content: '';
            position: absolute;
            width: 8px;
            height: 10px;
            border: 1.5px solid var(--accent);
            border-radius: 1px;
        }
        
        .loading-icon::before {
            top: 0;
            left: 0;
            animation: docShuffle 0.6s ease-in-out infinite;
        }
        
        .loading-icon::after {
            top: 4px;
            left: 6px;
            opacity: 0.5;
            animation: docShuffle 0.6s ease-in-out infinite 0.15s;
        }
        
        @keyframes docShuffle {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-3px); }
        }
        
        .progress-track {
            height: 4px;
            background: var(--surface);
            border-radius: 2px;
            overflow: hidden;
            margin-bottom: 0.75rem;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--accent), #60a5fa);
            border-radius: 2px;
            width: 0%;
            transition: width 0.4s ease-out;
        }
        
        .progress-stages {
            display: flex;
            justify-content: space-between;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.65rem;
            color: var(--text-muted);
        }
        
        .progress-stages span {
            opacity: 0.4;
            transition: opacity 0.3s, color 0.3s;
        }
        
        .progress-stages span.active {
            opacity: 1;
            color: var(--accent);
        }
        
        .progress-stages span.done {
            opacity: 0.7;
            color: var(--text);
        }
        
        #results {
            display: none;
        }
        
        #results.active {
            display: block;
        }
        
        .meta-info {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-muted);
            padding: 1rem;
            background: var(--surface);
            border-radius: 6px;
            margin-bottom: 1.5rem;
            border-left: 3px solid var(--accent);
        }
        
        .answer {
            padding: 1.5rem 0;
            border-bottom: 1px solid var(--border);
        }
        
        .answer h2, .answer h3 {
            font-size: 1.1rem;
            margin: 1.5rem 0 0.75rem;
            font-weight: 600;
        }
        
        .answer h2:first-child, .answer h3:first-child {
            margin-top: 0;
        }
        
        .answer p {
            margin-bottom: 1rem;
        }
        
        .answer ul, .answer ol {
            margin: 1rem 0;
            padding-left: 1.5rem;
        }
        
        .answer li {
            margin-bottom: 0.5rem;
        }
        
        .answer strong {
            color: #fff;
        }
        
        .answer code {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.85em;
            background: var(--surface);
            padding: 0.15rem 0.4rem;
            border-radius: 3px;
        }
        
        .sources {
            padding: 1.5rem 0;
        }
        
        .sources h4 {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
            margin-bottom: 1rem;
        }
        
        .source-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        
        .source-tag {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            padding: 0.35rem 0.6rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 4px;
            color: var(--text-muted);
        }
        
        @media (max-width: 640px) {
            .container {
                padding: 1rem;
            }
            header {
                padding: 2rem 0;
            }
            .examples span {
                display: block;
                margin: 0.5rem 0;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>National Security Strategy Search</h1>
            <p class="subtitle">RAG-powered document analysis</p>
        </header>
        
        <div class="search-box">
            <input 
                type="text" 
                id="query-input" 
                placeholder="Ask about national security strategies..."
                autocomplete="off"
            >
            <div class="examples">
                Try: 
                <span onclick="setQuery('What is the latest US NSS about?')">Latest US strategy</span>
                <span onclick="setQuery('How does Japan view China as a threat?')">Japan on China</span>
                <span onclick="setQuery('Compare cyber security approaches across countries')">Cyber comparison</span>
            </div>
        </div>
        
        <div id="loading">
            <div class="loading-container">
                <div class="loading-status">
                    <div class="loading-icon"></div>
                    <span id="loading-text">Understanding query...</span>
                </div>
                <div class="progress-track">
                    <div class="progress-bar" id="progress-bar"></div>
                </div>
                <div class="progress-stages">
                    <span id="stage-0" class="active">Parse</span>
                    <span id="stage-1">Search</span>
                    <span id="stage-2">Retrieve</span>
                    <span id="stage-3">Analyze</span>
                </div>
            </div>
        </div>
        
        <div id="results">
            <div class="meta-info" id="meta-info"></div>
            <div class="answer" id="answer"></div>
            <div class="sources">
                <h4>Sources</h4>
                <div class="source-list" id="sources"></div>
            </div>
        </div>
    </div>
    
    <script>
        const input = document.getElementById('query-input');
        const loading = document.getElementById('loading');
        const loadingText = document.getElementById('loading-text');
        const progressBar = document.getElementById('progress-bar');
        const results = document.getElementById('results');
        const metaInfo = document.getElementById('meta-info');
        const answerDiv = document.getElementById('answer');
        const sourcesDiv = document.getElementById('sources');
        
        const stages = [
            { text: 'Understanding query...', progress: 15 },
            { text: 'Searching documents...', progress: 35 },
            { text: 'Retrieving chunks...', progress: 55 },
            { text: 'Analyzing content...', progress: 75 },
            { text: 'Generating answer...', progress: 90 }
        ];
        
        let stageInterval = null;
        let currentStage = 0;
        
        function resetProgress() {
            currentStage = 0;
            progressBar.style.width = '0%';
            for (let i = 0; i < 4; i++) {
                const el = document.getElementById(`stage-${i}`);
                el.classList.remove('active', 'done');
            }
            document.getElementById('stage-0').classList.add('active');
            loadingText.textContent = stages[0].text;
        }
        
        function advanceStage() {
            if (currentStage >= stages.length - 1) return;
            
            // Mark current as done
            if (currentStage < 4) {
                document.getElementById(`stage-${currentStage}`).classList.remove('active');
                document.getElementById(`stage-${currentStage}`).classList.add('done');
            }
            
            currentStage++;
            const stage = stages[currentStage];
            
            // Update progress bar
            progressBar.style.width = stage.progress + '%';
            loadingText.textContent = stage.text;
            
            // Mark new stage as active
            if (currentStage < 4) {
                document.getElementById(`stage-${currentStage}`).classList.add('active');
            }
        }
        
        function startProgress() {
            resetProgress();
            progressBar.style.width = stages[0].progress + '%';
            
            // Advance through stages with varying delays
            const delays = [800, 600, 700, 1500]; // Time for each stage
            let totalDelay = 0;
            
            delays.forEach((delay, i) => {
                totalDelay += delay;
                setTimeout(() => advanceStage(), totalDelay);
            });
        }
        
        function completeProgress() {
            if (stageInterval) clearInterval(stageInterval);
            progressBar.style.width = '100%';
            
            // Mark all as done
            for (let i = 0; i < 4; i++) {
                const el = document.getElementById(`stage-${i}`);
                el.classList.remove('active');
                el.classList.add('done');
            }
        }
        
        function setQuery(q) {
            input.value = q;
            input.focus();
            submitQuery();
        }
        
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                submitQuery();
            }
        });
        
        async function submitQuery() {
            const query = input.value.trim();
            if (!query) return;
            
            // Show loading with animation
            loading.classList.add('active');
            results.classList.remove('active');
            startProgress();
            
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query })
                });
                
                if (!response.ok) {
                    throw new Error('Query failed');
                }
                
                const data = await response.json();
                
                // Complete progress animation
                completeProgress();
                
                // Small delay to show completion
                await new Promise(r => setTimeout(r, 300));
                
                // Display meta info
                let meta = `Search: "${data.parsed_query.search_query || query}"`;
                if (Object.keys(data.filters).length > 0) {
                    meta += ` | Filters: ${JSON.stringify(data.filters)}`;
                }
                metaInfo.textContent = meta;
                
                // Display answer (render markdown)
                answerDiv.innerHTML = marked.parse(data.answer);
                
                // Display sources
                sourcesDiv.innerHTML = data.sources.map(s => 
                    `<span class="source-tag">${s.country} (${s.year}) p.${s.page}</span>`
                ).join('');
                
                // Show results
                loading.classList.remove('active');
                results.classList.add('active');
                
            } catch (err) {
                completeProgress();
                loading.classList.remove('active');
                answerDiv.innerHTML = `<p style="color: #ef4444;">Error: ${err.message}</p>`;
                results.classList.add('active');
            }
        }
    </script>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
