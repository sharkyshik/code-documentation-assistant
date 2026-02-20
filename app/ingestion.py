"""
Codebase ingestion and indexing.
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .config import SUPPORTED_EXTENSIONS, BLOCKED_INGEST_PATHS, MAX_FILES_PER_INGEST, MAX_FILE_SIZE_BYTES
from .chunking import chunk_code, CodeChunk
from .retriever import get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of ingestion process."""
    files_processed: int
    chunks_created: int
    files_failed: int
    errors: List[str]


class CodebaseIngester:
    """Handles ingestion of codebases into the vector store."""

    def __init__(self):
        self.vector_store = get_vector_store()

    def ingest_directory(
        self,
        directory_path: str,
        clear_existing: bool = False
    ) -> IngestionResult:
        """
        Ingest all Python files from a directory.

        Args:
            directory_path: Path to the directory to ingest
            clear_existing: If True, clear existing data before ingestion

        Returns:
            IngestionResult with statistics
        """
        path = self._validate_ingest_path(directory_path)

        if clear_existing:
            self.vector_store.clear()
            logger.info("Cleared existing vector store data")

        # Find all Python files
        python_files = self._find_python_files(path)
        logger.info(f"Found {len(python_files)} Python files to process")

        if len(python_files) > MAX_FILES_PER_INGEST:
            raise ValueError(
                f"Directory contains {len(python_files)} Python files, exceeding the limit of "
                f"{MAX_FILES_PER_INGEST}. Use a more specific path or increase MAX_FILES_PER_INGEST."
            )

        files_processed = 0
        chunks_created = 0
        files_failed = 0
        errors = []

        for file_path in python_files:
            try:
                chunks = self._process_file(file_path)
                if chunks:
                    added = self.vector_store.add_chunks(chunks)
                    chunks_created += added
                    files_processed += 1
                    logger.info(f"Processed {file_path}: {added} chunks")
            except Exception as e:
                files_failed += 1
                error_msg = f"Failed to process {file_path}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)

        return IngestionResult(
            files_processed=files_processed,
            chunks_created=chunks_created,
            files_failed=files_failed,
            errors=errors
        )

    def ingest_file(self, file_path: str) -> int:
        """
        Ingest a single Python file.

        Args:
            file_path: Path to the file

        Returns:
            Number of chunks created
        """
        path = Path(file_path)

        if not path.exists():
            raise ValueError(f"File does not exist: {file_path}")

        if path.suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        chunks = self._process_file(str(path))
        if chunks:
            return self.vector_store.add_chunks(chunks)
        return 0

    def _validate_ingest_path(self, directory_path: str) -> Path:
        """
        Resolve and validate that the directory path is safe to ingest.

        Raises ValueError for non-existent, non-directory, or blocked paths.
        """
        path = Path(directory_path).resolve()

        if not path.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")

        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        path_normalized = str(path).lower().replace("\\", "/")
        for blocked in BLOCKED_INGEST_PATHS:
            if path_normalized.startswith(blocked.lower()):
                raise ValueError(
                    f"Ingestion of system directory is not permitted: {directory_path}"
                )

        return path

    def _find_python_files(self, directory: Path) -> List[str]:
        """Find all Python files in a directory recursively."""
        python_files = []

        for root, dirs, files in os.walk(directory):
            # Skip common non-code directories
            dirs[:] = [
                d for d in dirs
                if not d.startswith('.')
                and d not in ['__pycache__', 'node_modules', 'venv', '.venv', 'env', '.env']
            ]

            for file in files:
                if any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    python_files.append(os.path.join(root, file))

        return python_files

    def _process_file(self, file_path: str) -> List[CodeChunk]:
        """Read and chunk a single file."""
        try:
            file_size = os.path.getsize(file_path)
            if file_size > MAX_FILE_SIZE_BYTES:
                logger.warning(
                    f"Skipping {file_path}: size {file_size // 1024}KB exceeds "
                    f"limit of {MAX_FILE_SIZE_BYTES // 1024}KB"
                )
                return []

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if not content.strip():
                logger.debug(f"Skipping empty file: {file_path}")
                return []

            return chunk_code(file_path, content)

        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            raise


def ingest_codebase(
    directory_path: str,
    clear_existing: bool = False
) -> IngestionResult:
    """
    Convenience function to ingest a codebase.

    Args:
        directory_path: Path to the directory
        clear_existing: Whether to clear existing data

    Returns:
        IngestionResult with statistics
    """
    ingester = CodebaseIngester()
    return ingester.ingest_directory(directory_path, clear_existing)


def get_ingestion_stats() -> Dict[str, Any]:
    """Get current ingestion statistics."""
    vector_store = get_vector_store()
    return vector_store.get_stats()
