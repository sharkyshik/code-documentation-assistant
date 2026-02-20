# Code Documentation Assistant

A local, privacy-first RAG (Retrieval-Augmented Generation) system for answering questions about your codebase. No external APIs, no data leaving your machine.

## What It Does

Point it at a Python codebase, and it will:
1. Parse your code into semantic chunks (functions, classes)
2. Generate embeddings and store them locally
3. Answer natural language questions grounded in your actual code

Example:
```
POST /query
{"question": "How does the authentication middleware work?"}

Response:
{
  "answer": "The authentication middleware in `app/middleware/auth.py` validates JWT tokens...",
  "sources": [{"file_path": "app/middleware/auth.py", "start_line": 45, "end_line": 78}]
}
```

## Quick Start (Windows)

### 1. Install Ollama

Download and install from [ollama.com](https://ollama.com/download/windows).

After installation, open a terminal and pull the required models:

```powershell
ollama pull nomic-embed-text
ollama pull llama3
```

Verify Ollama is running:
```powershell
curl http://localhost:11434/api/tags
```

### 2. Set Up the Project

```powershell
# Clone or copy the project
cd code-doc-assistant

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Server

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Index Your Code

```powershell
# Using curl
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d "{\"directory_path\": \"C:\\path\\to\\your\\code\"}"

# Or using PowerShell
Invoke-RestMethod -Uri "http://localhost:8000/ingest" -Method Post -ContentType "application/json" -Body '{"directory_path": "C:\\path\\to\\your\\code"}'
```

### 5. Ask Questions

```powershell
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"question\": \"What does the main function do?\"}"
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check system status and model availability |
| `/ingest` | POST | Index a codebase directory |
| `/query` | POST | Ask a question about the indexed code |
| `/stats` | GET | Get indexing statistics |
| `/clear` | DELETE | Clear all indexed data |

### POST /ingest
```json
{
  "directory_path": "C:\\projects\\my-app",
  "clear_existing": false
}
```

### POST /query
```json
{
  "question": "How is error handling implemented?",
  "top_k": 5
}
```

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   FastAPI    │────▶│   Chunker    │────▶│   ChromaDB   │
│   Server     │     │  (AST-based) │     │ Vector Store │
└──────────────┘     └──────────────┘     └──────────────┘
       │                                         │
       ▼                                         ▼
┌──────────────┐                         ┌──────────────┐
│    Ollama    │◀────────────────────────│  Retriever   │
│  (LLM + Emb) │                         │              │
└──────────────┘                         └──────────────┘
```

**Data Flow:**
1. **Ingestion**: Python files → AST parsing → Code chunks → Embeddings → ChromaDB
2. **Query**: Question → Embedding → Vector search → Context + Question → LLM → Answer

## Design Decisions

### Why Ollama?

I chose Ollama over alternatives like llama.cpp directly or vLLM for several reasons:

1. **Dead simple setup**: One install, `ollama pull model`, done. No CUDA toolkit wrestling, no model format conversions.

2. **API-first design**: REST API out of the box means our app stays decoupled from the inference engine. If you want to swap in a different backend later, just change the URL.

3. **Model management**: Ollama handles model downloads, updates, and storage. You're not manually managing 4GB+ files.

4. **Windows support**: Actually works on Windows without WSL gymnastics (though WSL works too if you prefer).

The tradeoff: Ollama adds a layer between you and the raw model. If you need maximum inference speed or custom quantization, you might want llama.cpp directly. But for this use case, the convenience wins.

### Why ChromaDB?

Considered alternatives:
- **FAISS**: Faster for pure similarity search, but no persistence out of the box, less Pythonic API
- **Pinecone/Weaviate**: Cloud services, violates our "local-only" requirement
- **Milvus**: Overkill for this scale, complex setup

ChromaDB hits the sweet spot:
- Embedded mode (no separate server)
- Persistent storage with one line of config
- Python-native, good type hints
- Reasonable performance up to ~100k documents

The tradeoff: ChromaDB isn't the fastest option. For a million+ document corpus, you'd want FAISS or Milvus. But for typical codebases (thousands of files), it's plenty fast.

### Chunking Strategy

I use Python's AST module to extract functions and classes as chunks. Why not simpler approaches?

**Rejected approaches:**
- **Fixed-size chunks** (e.g., 512 tokens): Splits code mid-function, loses semantic meaning
- **Line-based chunks**: Same problem
- **File-level chunks**: Too coarse, retrieval becomes noisy

**Current approach:**
- Parse AST, extract function and class definitions
- Include decorators with their functions
- Fall back to whole-file chunk if parsing fails (for non-Python or invalid syntax)

**Tradeoffs:**
- Only works for Python (other languages need different parsers)
- Nested classes/functions create duplicate content (class includes method, method exists separately)
- Very long functions might benefit from further splitting

A production system might use Tree-sitter for multi-language support, but for Python-only, AST is clean and stdlib.

### No LangChain

I deliberately avoided LangChain despite its popularity. Reasons:

1. **Abstraction debt**: LangChain adds layers that make debugging harder. When your RAG pipeline returns garbage, you want to see exactly what prompt went to the LLM.

2. **Simplicity**: Our use case needs ~50 lines of actual logic. LangChain would add hundreds of lines of framework code.

3. **Control**: Direct Ollama API calls mean we control timeout, retry, streaming behavior exactly.

The tradeoff: If you need to rapidly prototype with many different LLMs/vector stores, LangChain's abstractions help. For a focused tool like this, they just get in the way.

## Tradeoffs and Limitations

### What This Does Well
- Simple, understandable codebase
- Fast setup, works offline
- Accurate retrieval for specific code questions
- File references in answers

### Current Limitations
- **Python only**: AST parsing is Python-specific
- **No incremental updates**: Changing one file requires re-indexing (could add file hashing)
- **No cross-file reasoning**: Asks about individual chunks, not "how does module A interact with module B"
- **Memory usage**: ChromaDB loads embeddings into memory
- **Single-user**: No authentication or multi-tenancy

### Known Issues
- Very large functions (500+ lines) might hit embedding token limits
- Heavily nested code creates somewhat redundant chunks
- First query is slow (model loading) if Ollama wasn't warmed up

## Configuration

Copy `.env.example` to `.env` and adjust:

```bash
# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
LLM_MODEL=llama3

# ChromaDB settings
CHROMA_PERSIST_DIR=./chroma_db
COLLECTION_NAME=code_docs

# Retrieval
TOP_K=5
```

## Running with Docker

The app runs in Docker, but Ollama runs on the host (GPU access):

```powershell
# Start Ollama on host
ollama serve

# Build and run the app
docker build -t code-doc-assistant .
docker run -p 8000:8000 code-doc-assistant
```

Note: On Windows, Docker uses `host.docker.internal` to reach host services. This is configured in the Dockerfile.

For a fully containerized setup, you'd need Ollama in a separate container with GPU passthrough, which varies by system.

## Testing

```powershell
pytest tests/ -v
```

Tests use mocking to avoid requiring Ollama during CI.

## Production Scaling Plan

If I were deploying this at scale, here's the evolution path:

### Phase 1: Small Team (< 10 users)
- Current architecture is fine
- Add basic auth (API key header)
- Maybe Nginx in front for HTTPS

### Phase 2: Medium Scale (10-100 users)
- Move ChromaDB to client-server mode (separate process)
- Add Redis for query caching
- Run multiple FastAPI workers
- Consider GPU instance for Ollama (p3.2xlarge on AWS, n1-standard-4 + T4 on GCP)

### Phase 3: Large Scale (100+ users)
- Replace ChromaDB with Milvus or Qdrant (horizontal scaling)
- Ollama → vLLM or TGI for batched inference
- Kubernetes deployment
- Add proper observability (Prometheus, Grafana, tracing)
- Implement queue-based ingestion (Celery + Redis)

### Cost Estimates (AWS)
- Small: t3.medium + EBS = ~$50/month
- Medium: p3.2xlarge (GPU) = ~$1000/month
- Large: EKS cluster + managed services = $3000+/month

The beauty of starting local-only is you can run this on a $0 laptop until you actually need scale.

## What I'd Improve With More Time

**Week 1:**
- Add file hash tracking for incremental updates
- Support more languages via Tree-sitter
- Add simple web UI (just a search box, really)

**Week 2:**
- Implement query caching
- Add conversation memory (follow-up questions)
- Better chunking for huge files (split long functions)

**Month 1:**
- Multi-repo support
- Git integration (index specific branches/commits)
- Streaming responses in API
- Proper auth and rate limiting

**Longer term:**
- Fine-tune embedding model on code
- Add code execution sandbox for "run this function" queries
- IDE plugin (VS Code extension)

## License

MIT

---

Built because I got tired of grep-ing through unfamiliar codebases. Sometimes you just want to ask "where does this thing get initialized?" and get a straight answer.
