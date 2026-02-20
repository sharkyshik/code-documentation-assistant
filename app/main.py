"""
FastAPI entrypoint for the Code Documentation Assistant.
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .config import TOP_K, EMBEDDING_MODEL, LLM_MODEL, MAX_QUESTION_LENGTH, MAX_TOP_K
from .ingestion import ingest_codebase, get_ingestion_stats
from .retriever import get_vector_store, format_context
from .llm import generate_response
from .embeddings import check_ollama_connection, check_model_available

# Configure logging with file and console output
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'queries.log'

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    logger.info("Starting Code Documentation Assistant")

    # Check Ollama connection
    if not check_ollama_connection():
        logger.warning(
            "Cannot connect to Ollama. Make sure it's running at localhost:11434"
        )
    else:
        logger.info("Connected to Ollama")

        # Check models
        if not check_model_available(EMBEDDING_MODEL):
            logger.warning(f"Embedding model '{EMBEDDING_MODEL}' not found. Run: ollama pull {EMBEDDING_MODEL}")
        if not check_model_available(LLM_MODEL):
            logger.warning(f"LLM model '{LLM_MODEL}' not found. Run: ollama pull {LLM_MODEL}")

    yield
    # Shutdown (nothing to clean up)


# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(
    title="Code Documentation Assistant",
    description="RAG-based system for answering questions about codebases",
    version="1.0.0",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class IngestRequest(BaseModel):
    """Request model for ingestion endpoint."""
    directory_path: str = Field(..., description="Path to the codebase directory")
    clear_existing: bool = Field(False, description="Clear existing data before ingestion")


class IngestResponse(BaseModel):
    """Response model for ingestion endpoint."""
    success: bool
    files_processed: int
    chunks_created: int
    files_failed: int
    errors: List[str]
    message: str


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH, description="Question about the codebase")
    top_k: int = Field(TOP_K, ge=1, le=MAX_TOP_K, description="Number of chunks to retrieve")


class Source(BaseModel):
    """Source reference in query response."""
    file_path: str
    chunk_type: str
    name: Optional[str]
    start_line: Optional[int]
    end_line: Optional[int]


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    sources: List[Source]
    retrieval_time_ms: float
    llm_time_ms: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    ollama_connected: bool
    embedding_model_available: bool
    llm_model_available: bool
    documents_indexed: int


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Verifies Ollama connection and model availability.
    """
    ollama_connected = check_ollama_connection()
    embedding_available = check_model_available(EMBEDDING_MODEL) if ollama_connected else False
    llm_available = check_model_available(LLM_MODEL) if ollama_connected else False

    vector_store = get_vector_store()
    doc_count = vector_store.collection.count()

    status = "healthy" if (ollama_connected and embedding_available and llm_available) else "degraded"

    return HealthResponse(
        status=status,
        ollama_connected=ollama_connected,
        embedding_model_available=embedding_available,
        llm_model_available=llm_available,
        documents_indexed=doc_count
    )


@app.post("/ingest", response_model=IngestResponse)
@limiter.limit("5/minute")
async def ingest_endpoint(http_request: Request, request: IngestRequest):
    """
    Ingest a codebase into the vector store.

    Recursively scans the directory for Python files,
    chunks them, and stores embeddings.
    """
    logger.info(f"Starting ingestion of: {request.directory_path}")

    try:
        result = ingest_codebase(
            request.directory_path,
            clear_existing=request.clear_existing
        )

        message = f"Ingested {result.files_processed} files with {result.chunks_created} chunks"
        if result.files_failed > 0:
            message += f" ({result.files_failed} files failed)"

        logger.info(message)

        return IngestResponse(
            success=result.files_failed == 0,
            files_processed=result.files_processed,
            chunks_created=result.chunks_created,
            files_failed=result.files_failed,
            errors=result.errors,
            message=message
        )

    except ValueError as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
async def query_endpoint(http_request: Request, request: QueryRequest):
    """
    Query the codebase.

    Retrieves relevant code chunks and generates an answer
    using the LLM.
    """
    logger.info(f"Query: {request.question}")

    # Check if we have any documents
    vector_store = get_vector_store()
    if vector_store.collection.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Please ingest a codebase first using POST /ingest"
        )

    try:
        # Retrieve relevant chunks
        start_time = time.time()
        results = vector_store.query(request.question, top_k=request.top_k)
        retrieval_time = (time.time() - start_time) * 1000

        logger.info(f"Retrieved {len(results)} chunks in {retrieval_time:.2f}ms")

        if not results:
            return QueryResponse(
                answer="No relevant code found for your question.",
                sources=[],
                retrieval_time_ms=retrieval_time,
                llm_time_ms=0
            )

        # Format context for LLM
        context = format_context(results)

        # Generate response
        start_time = time.time()
        answer = generate_response(request.question, context)
        llm_time = (time.time() - start_time) * 1000

        logger.info(f"Generated response in {llm_time:.2f}ms")

        # Build sources list
        sources = [
            Source(
                file_path=r.file_path,
                chunk_type=r.chunk_type,
                name=r.name,
                start_line=r.start_line,
                end_line=r.end_line
            )
            for r in results
        ]

        return QueryResponse(
            answer=answer,
            sources=sources,
            retrieval_time_ms=round(retrieval_time, 2),
            llm_time_ms=round(llm_time, 2)
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


@app.get("/stats")
async def stats_endpoint():
    """
    Get statistics about the indexed codebase.
    """
    return get_ingestion_stats()


@app.delete("/clear")
async def clear_endpoint():
    """
    Clear all indexed data.
    """
    vector_store = get_vector_store()
    vector_store.clear()
    logger.info("Cleared all indexed data")
    return {"message": "All indexed data cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
