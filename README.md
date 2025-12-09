# NSS Document Search - Web App

A RAG-powered web interface for searching National Security Strategy documents.

## Features

- **Natural language queries** - Ask questions like "What is the latest US strategy about?"
- **LLM-powered query understanding** - Automatically extracts country, year, and search intent
- **Smart filtering** - Understands "latest", date ranges, and country references
- **Source citations** - See exactly which documents were used

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set environment variables:**
```bash
export OPENAI_API_KEY='sk-...'
export ANTHROPIC_API_KEY='sk-ant-...'
```

3. **Ensure ChromaDB is populated:**
The app expects a ChromaDB at `./chroma_db` with a collection named `nss_documents`.

4. **Run the server:**
```bash
python app.py
```

Or with uvicorn directly:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

5. **Open in browser:**
```
http://localhost:8000
```

## API Endpoints

### POST /api/query
Query the document database.

**Request:**
```json
{
  "query": "What is the latest US NSS about?"
}
```

**Response:**
```json
{
  "answer": "The 2025 US National Security Strategy focuses on...",
  "sources": [
    {"country": "United States", "year": 2025, "doc_name": "us_2025_NationalSecurityStrategy", "page": 10}
  ],
  "parsed_query": {"country": "United States", "wants_latest": true, "search_query": "..."},
  "filters": {"country": "United States", "year_min": 2025, "year_max": 2025}
}
```

### GET /api/stats
Get database statistics (countries, years, chunk count).

## Deploy to Railway

### Option 1: Quick Deploy (DB in repo)

If your `chroma_db` is small enough (<500MB), just include it in the repo:

```bash
# Initialize git repo
git init
git add .
git commit -m "Initial commit"

# Push to GitHub
gh repo create nss-search --public --push

# Deploy on Railway
# 1. Go to railway.app → New Project → Deploy from GitHub
# 2. Select your repo
# 3. Add environment variables:
#    - OPENAI_API_KEY
#    - ANTHROPIC_API_KEY
# 4. Deploy!
```

### Option 2: Persistent Volume (Large DB)

For larger databases, use Railway volumes:

1. Create project on Railway
2. Add a **Volume** and mount it at `/data`
3. Update `app.py`:
   ```python
   CHROMA_PATH = "/data/chroma_db"  # Use volume path
   ```
4. Upload your `chroma_db` folder to the volume (via Railway CLI):
   ```bash
   railway volume upload ./chroma_db /data/chroma_db
   ```

### Environment Variables

Set these in Railway dashboard → Variables:

| Variable | Value |
|----------|-------|
| `OPENAI_API_KEY` | `sk-...` |
| `ANTHROPIC_API_KEY` | `sk-ant-...` |

### Custom Domain

Railway gives you a `.railway.app` URL automatically. To add custom domain:
1. Settings → Domains → Add Custom Domain
2. Add CNAME record pointing to your Railway URL

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│ understand_query │ ← LLM extracts: country, year, intent
│   (Claude)       │   rewrites query for semantic search
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  build_filters   │ ← Converts "latest" → max(year) lookup
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    retrieve      │ ← ChromaDB vector search + metadata filters
│   (OpenAI emb)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ generate_answer  │ ← LLM synthesizes answer from chunks
│    (Claude)      │
└────────┬────────┘
         │
         ▼
    Response
```
