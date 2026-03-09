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

## Deploy to DigitalOcean

### 1. Create a Droplet

1. Go to [cloud.digitalocean.com](https://cloud.digitalocean.com) → **Create → Droplets**
2. Choose **Ubuntu 22.04 LTS**, **$12/mo** (2 GB RAM / 1 vCPU / 50 GB disk)
3. Add your **SSH key** and create

### 2. SSH in and deploy

```bash
ssh root@<your-droplet-ip>

# Download and run the deploy script
git clone https://github.com/KasperBuilds/NationalSecurityRAG.git
cd NationalSecurityRAG
bash deploy.sh
```

### 3. Set your API key

```bash
sudo nano /etc/nationalsecurityrag.env
# Set: OPENROUTER_API_KEY=sk-or-your-real-key
sudo systemctl restart nssrag
```

### 4. Verify

```bash
curl http://localhost/api/stats
```

Or visit `http://<your-droplet-ip>` in your browser.

### Environment Variables

| Variable | Value |
|----------|-------|
| `OPENROUTER_API_KEY` | `sk-or-...` |
| `CHROMA_PATH` | `./chroma_db` (default) |

### Custom Domain + HTTPS

1. Point your domain's **A record** to the Droplet IP
2. Edit Nginx config: `sudo nano /etc/nginx/sites-available/nssrag` → set `server_name yourdomain.com`
3. Run: `sudo certbot --nginx -d yourdomain.com`

### Useful Commands

| Action | Command |
|--------|---------|
| Restart app | `sudo systemctl restart nssrag` |
| View logs | `sudo journalctl -u nssrag -f` |
| Check status | `sudo systemctl status nssrag` |
| Update code | `cd ~/NationalSecurityRAG && git pull && sudo systemctl restart nssrag` |

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│ understand_query │ ← LLM extracts: country, year, intent
│  (OpenRouter)    │   rewrites query for semantic search
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
│  (OpenRouter)    │
└────────┬────────┘
         │
         ▼
    Response
```
