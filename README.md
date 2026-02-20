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

### No Reranking

Reranking is a common RAG pattern where a second-pass model (typically a cross-encoder like Cohere Rerank or BGE Reranker) rescores an initial set of retrieved candidates before passing them to the LLM. I chose not to include it here.

**Why it's not needed for this use case:**

1. **AST chunks are already semantically tight**: Each chunk is a complete function or class — not an arbitrary 512-token window that might start mid-sentence. A query like "how does the embeddings module work?" maps cleanly to `embeddings.py` chunks. The first-pass retrieval is already high precision.

2. **Small corpus, high signal**: Typical codebases indexed here span hundreds to low thousands of chunks. Reranking pays off when you retrieve a large noisy candidate set (top 50–100) and need to find the best 5. With a small, well-structured corpus, the initial cosine similarity search is already accurate enough.

3. **Code queries are keyword-rich**: Natural language questions about code ("where is X initialized?", "what does function Y return?") contain the exact identifiers that appear in the source. This reduces the semantic gap between query and document that rerankers are designed to bridge.

4. **The LLM reranks implicitly**: With only `top_k` (default: 5) chunks in context, the LLM naturally weighs relevance when synthesizing an answer. It will ignore a tangentially related chunk if a more relevant one is present. This is cheap and requires no extra model.

5. **Local-only constraint**: Effective hosted rerankers (Cohere Rerank) send data to external APIs, violating the privacy-first design. Local cross-encoders (BGE, ms-marco) add another model to download, manage, and run — significant overhead for marginal gain at this scale.

**When you'd want reranking:**
- Multi-repo indexing with tens of thousands of chunks
- Queries that are vague or use natural language rather than code identifiers
- You've already tuned chunking and retrieval and are chasing the last few percent of accuracy

### No LangChain

I deliberately avoided LangChain despite its popularity. Reasons:

1. **Abstraction debt**: LangChain adds layers that make debugging harder. When your RAG pipeline returns garbage, you want to see exactly what prompt went to the LLM.

2. **Simplicity**: Our use case needs ~50 lines of actual logic. LangChain would add hundreds of lines of framework code.

3. **Control**: Direct Ollama API calls mean we control timeout, retry, streaming behavior exactly.

The tradeoff: If you need to rapidly prototype with many different LLMs/vector stores, LangChain's abstractions help. For a focused tool like this, they just get in the way.

## Guardrails

The following guardrails are applied to protect the system from misuse and resource exhaustion. All limits are configurable via environment variables.

### Input Validation

Applied by Pydantic on every request — invalid inputs are rejected with HTTP 422 before reaching any business logic.

| Field | Rule | Default |
|-------|------|---------|
| `question` | 1–1000 characters | `MAX_QUESTION_LENGTH=1000` |
| `top_k` | Integer between 1 and 20 | `MAX_TOP_K=20` |

### Rate Limiting

Enforced per IP address using [slowapi](https://github.com/laurentS/slowapi).

| Endpoint | Limit |
|----------|-------|
| `POST /query` | 10 requests / minute |
| `POST /ingest` | 5 requests / minute |

Exceeding the limit returns HTTP 429. Limits are enforced in-memory and reset on server restart.

### Directory Path Security

Applied before any file is read during ingestion:

- **Path resolution**: The provided path is resolved to its absolute canonical form (symlinks expanded). This prevents traversal attacks like `../../etc`.
- **System directory blocklist**: Ingestion is refused if the resolved path starts with a known system directory:
  - Unix: `/etc`, `/proc`, `/sys`, `/dev`, `/root`, `/usr`, `/bin`, `/sbin`
  - Windows: `C:\Windows`, `C:\Program Files`, `C:\Program Files (x86)`

### Resource Limits

Prevent runaway ingestion jobs from exhausting memory or disk I/O.

| Limit | Default | Env var |
|-------|---------|---------|
| Max Python files per ingestion job | 500 | `MAX_FILES_PER_INGEST` |
| Max file size per file | 1 MB (1,048,576 bytes) | `MAX_FILE_SIZE_BYTES` |

Files exceeding the size limit are skipped with a warning log rather than failing the entire ingestion job.

## Known Hallucination Scenarios

The following scenarios are structural — they arise from how the pipeline is built, not from model quality. Each entry includes a test prompt, what the system is likely to return, and the root cause in the code.

---

### 1. No Relevance Threshold — Forced Retrieval on Unrelated Questions

**Test prompt:**
```json
{"question": "How does user authentication work in this codebase?"}
```

**What happens:** ChromaDB always returns exactly `top_k` results regardless of how dissimilar they are to the query (`retriever.py:129` — no distance cutoff). If the codebase has no authentication code, the retriever returns the closest chunks it has (e.g., the `config.py` chunk or an unrelated function). The LLM then tries to answer from those chunks and may fabricate an auth flow by over-interpreting unrelated code — or correctly say "I don't know", depending on how well the model follows the system prompt.

**Root cause:** `VectorStore.query()` passes no distance filter to ChromaDB. Every query returns results.

**Mitigation:** Set a maximum distance threshold (e.g., cosine distance > 0.5 = not relevant). Chunks above the threshold should be dropped before passing context to the LLM.

---

### 2. Module-Level Code Not Indexed

**Test prompt:**
```json
{"question": "What file extensions does the system support?"}
```

**What happens:** The answer lives in `config.py` as `SUPPORTED_EXTENSIONS = [".py"]` — a module-level constant, not inside any function or class. The AST chunker in `chunking.py:70` only extracts `FunctionDef`, `AsyncFunctionDef`, and `ClassDef` nodes. Module-level assignments are only captured via the whole-file fallback, which only activates when **no** functions or classes exist in the file (`chunking.py:88`). Since `config.py` has no functions, it does get the fallback chunk — but any file that mixes functions with important module-level constants (global vars, instantiation code) will have those constants silently excluded from the index.

For example, `main.py` has `app = FastAPI(...)` and `limiter = Limiter(...)` at module level. Those lines are not in any chunk. A question like "how is the FastAPI app configured?" will miss the CORS middleware setup and lifespan handler instantiation.

**Root cause:** `chunking.py:88` — the fallback only fires when the chunk list is empty. Files with any functions/classes skip module-level code entirely.

**Mitigation:** Always add a module-level chunk regardless of whether functions/classes were found. Or extract `ast.Assign` / `ast.AugAssign` nodes for top-level constants.

---

### 3. Cross-File Reasoning Collapses Into One File

**Test prompt:**
```json
{"question": "How does a query flow from the API endpoint to the LLM response?"}
```

**What happens:** This question spans `main.py → retriever.py → llm.py`. The embedding for "query flow from API endpoint to LLM" most closely matches the `query_endpoint` function in `main.py`. With `top_k=5`, all five returned chunks may come from `main.py`, leaving out the retrieval and generation steps entirely. The LLM then describes only the endpoint layer and either stops there or invents the rest of the flow from its training data.

**Root cause:** There is no per-file diversity enforcement in retrieval. ChromaDB returns the globally top-k closest chunks with no constraint on source file spread.

**Mitigation:** Implement MMR (Maximal Marginal Relevance) or file-level de-duplication so retrieved chunks span multiple files. Alternatively, raise `top_k` for architecture questions — though this risks noise.

---

### 4. Training Data Bleed on Generic Questions

**Test prompt:**
```json
{"question": "What is a vector database?"}
```

**What happens:** The system prompt in `config.py:25` instructs the LLM to "Answer ONLY based on the provided context." However, general conceptual questions retrieve tangentially related chunks (e.g., the `VectorStore` class) and then the LLM answers with a mix of the retrieved code and its pre-trained knowledge about vector databases. The answer will be accurate but is no longer grounded in the indexed codebase — it's a blended response that violates the RAG contract.

**Root cause:** LLMs routinely ignore system prompt scope constraints when they have strong training knowledge on the topic. The instruction is a soft guardrail, not a hard one.

**Mitigation:** This is difficult to fully prevent with prompt engineering alone. Adding a distance threshold (see scenario 1) reduces it by refusing to answer when no close match exists.

---

### 5. Stale Index After Code Changes

**Test prompt** (after modifying `app/llm.py` without re-ingesting):
```json
{"question": "What is the LLM request timeout?"}
```

**What happens:** If `llm.py` was modified (e.g., timeout changed from `120` to `30`) but `/ingest` was not re-run, the vector store still holds the old embedding and content. The LLM answers `120 seconds` from the stale chunk. There is no file hash tracking or modification-time check in `ingestion.py`.

**Root cause:** `ingestion.py` has no mechanism to detect changed files. The only way to refresh is `clear_existing=true` on a full re-ingest.

**Mitigation:** Track file modification times or content hashes at ingest time. On subsequent ingests, skip unchanged files and re-embed only changed or new ones.

---

### 6. Duplicate Class and Method Chunks Cause Contradictory Context

**Test prompt:**
```json
{"question": "What does the add method in Calculator do?"}
```

**What happens:** The AST chunker (`chunking.py:70`) walks the entire tree with `ast.walk()`, which visits all nested nodes. For a class `Calculator` with method `add`, both the `ClassDef` node (containing the full class body including `add`) and the `FunctionDef` node for `add` are extracted as separate chunks. Both end up in the vector store. A query about `add` may retrieve both. The LLM now sees the method body twice — once in isolation and once embedded inside the class chunk. While typically harmless, this can cause the LLM to over-count the method's importance or generate redundant references.

**Root cause:** `chunking.py:70` uses `ast.walk()` which recurses into class bodies, so inner methods are visited as top-level `FunctionDef` nodes.

**Mitigation:** Use `ast.iter_child_nodes()` instead of `ast.walk()` at the top level to avoid recursing into class bodies, or de-duplicate by tracking parent node membership.

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

