"""
Configuration constants for the Code Documentation Assistant.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")

# ChromaDB configuration
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "code_docs")

# Retrieval configuration
TOP_K = int(os.getenv("TOP_K", "5"))

# Supported file extensions
SUPPORTED_EXTENSIONS = [".py"]

# Guardrails
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "1000"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "20"))
MAX_FILES_PER_INGEST = int(os.getenv("MAX_FILES_PER_INGEST", "500"))
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_BYTES", str(1 * 1024 * 1024)))  # 1MB

BLOCKED_INGEST_PATHS = [
    # Unix system directories
    "/etc", "/proc", "/sys", "/dev", "/root", "/usr", "/bin", "/sbin",
    # Windows system directories
    "c:/windows", "c:/program files", "c:/program files (x86)",
]

# LLM prompt template
SYSTEM_PROMPT = """You are a code documentation assistant. Answer ONLY based on the provided context.
If the answer is not found in the context, say "I don't know based on the provided code."
Always include file paths when referencing code.
Be concise and accurate."""

QUERY_PROMPT_TEMPLATE = """{system_prompt}

Context:
{context}

Question: {question}

Provide a clear answer with file references where applicable."""
