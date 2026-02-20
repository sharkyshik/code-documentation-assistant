"""
Vector storage and retrieval using ChromaDB.
"""
import chromadb
from chromadb.config import Settings
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .config import CHROMA_PERSIST_DIR, COLLECTION_NAME, TOP_K
from .embeddings import get_embedding
from .chunking import CodeChunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a retrieval result."""
    content: str
    file_path: str
    chunk_type: str
    name: Optional[str]
    start_line: Optional[int]
    end_line: Optional[int]
    distance: float


class VectorStore:
    """ChromaDB-based vector store for code chunks."""

    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIR):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory for ChromaDB persistence
        """
        self.persist_directory = persist_directory

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(
            f"Initialized ChromaDB with {self.collection.count()} documents"
        )

    def add_chunks(self, chunks: List[CodeChunk]) -> int:
        """
        Add code chunks to the vector store.

        Args:
            chunks: List of CodeChunk objects

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        documents = []
        embeddings = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            # Generate embedding
            try:
                embedding = get_embedding(chunk.content)
            except Exception as e:
                logger.error(f"Failed to embed chunk {i}: {e}")
                continue

            # Create unique ID
            chunk_id = f"{chunk.file_path}:{chunk.chunk_index}"

            documents.append(chunk.content)
            embeddings.append(embedding)
            metadatas.append({
                "file_path": chunk.file_path,
                "chunk_index": chunk.chunk_index,
                "chunk_type": chunk.chunk_type,
                "name": chunk.name or "",
                "start_line": chunk.start_line or 0,
                "end_line": chunk.end_line or 0
            })
            ids.append(chunk_id)

        if documents:
            # Upsert to handle duplicates
            self.collection.upsert(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} chunks to vector store")

        return len(documents)

    def query(
        self,
        query_text: str,
        top_k: int = TOP_K
    ) -> List[RetrievalResult]:
        """
        Query the vector store for similar chunks.

        Args:
            query_text: Query string
            top_k: Number of results to return

        Returns:
            List of RetrievalResult objects
        """
        # Generate query embedding
        query_embedding = get_embedding(query_text)

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # Parse results
        retrieval_results = []
        if results and results["documents"]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            for doc, meta, dist in zip(documents, metadatas, distances):
                retrieval_results.append(RetrievalResult(
                    content=doc,
                    file_path=meta.get("file_path", ""),
                    chunk_type=meta.get("chunk_type", ""),
                    name=meta.get("name") or None,
                    start_line=meta.get("start_line") or None,
                    end_line=meta.get("end_line") or None,
                    distance=dist
                ))

        return retrieval_results

    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Cleared vector store")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "total_documents": self.collection.count(),
            "collection_name": COLLECTION_NAME,
            "persist_directory": self.persist_directory
        }


# Global vector store instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def format_context(results: List[RetrievalResult]) -> str:
    """
    Format retrieval results into a context string for the LLM.

    Args:
        results: List of RetrievalResult objects

    Returns:
        Formatted context string
    """
    context_parts = []
    for i, result in enumerate(results, 1):
        location = f"{result.file_path}"
        if result.start_line:
            location += f" (lines {result.start_line}-{result.end_line})"

        header = f"[{i}] {result.chunk_type.upper()}"
        if result.name:
            header += f": {result.name}"
        header += f" | {location}"

        context_parts.append(f"{header}\n```python\n{result.content}\n```")

    return "\n\n".join(context_parts)
