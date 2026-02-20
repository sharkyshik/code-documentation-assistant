"""
Code chunking logic using AST parsing.
Extracts functions and classes as semantic chunks.
"""
import ast
from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata."""
    content: str
    file_path: str
    chunk_index: int
    chunk_type: str  # 'function', 'class', or 'module'
    name: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None


class CodeChunker:
    """Chunks Python code by semantic boundaries (functions/classes)."""

    def __init__(self, min_chunk_size: int = 50):
        """
        Initialize chunker.

        Args:
            min_chunk_size: Minimum characters for a chunk to be included
        """
        self.min_chunk_size = min_chunk_size

    def chunk_file(self, file_path: str, content: str) -> List[CodeChunk]:
        """
        Parse a Python file and extract chunks.

        Args:
            file_path: Path to the file
            content: File content as string

        Returns:
            List of CodeChunk objects
        """
        chunks = []
        lines = content.split('\n')

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            # Fall back to treating entire file as one chunk
            if len(content.strip()) >= self.min_chunk_size:
                chunks.append(CodeChunk(
                    content=content,
                    file_path=file_path,
                    chunk_index=0,
                    chunk_type="module",
                    name=file_path.split("\\")[-1].split("/")[-1],
                    start_line=1,
                    end_line=len(lines)
                ))
            return chunks

        chunk_index = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                chunk = self._extract_node_chunk(
                    node, lines, file_path, chunk_index, "function"
                )
                if chunk and len(chunk.content) >= self.min_chunk_size:
                    chunks.append(chunk)
                    chunk_index += 1

            elif isinstance(node, ast.ClassDef):
                chunk = self._extract_node_chunk(
                    node, lines, file_path, chunk_index, "class"
                )
                if chunk and len(chunk.content) >= self.min_chunk_size:
                    chunks.append(chunk)
                    chunk_index += 1

        # If no chunks found, use entire file
        if not chunks and len(content.strip()) >= self.min_chunk_size:
            chunks.append(CodeChunk(
                content=content,
                file_path=file_path,
                chunk_index=0,
                chunk_type="module",
                name=file_path.split("\\")[-1].split("/")[-1],
                start_line=1,
                end_line=len(lines)
            ))

        logger.info(f"Extracted {len(chunks)} chunks from {file_path}")
        return chunks

    def _extract_node_chunk(
        self,
        node: ast.AST,
        lines: List[str],
        file_path: str,
        chunk_index: int,
        chunk_type: str
    ) -> Optional[CodeChunk]:
        """Extract a chunk from an AST node."""
        try:
            start_line = node.lineno
            end_line = node.end_lineno or start_line

            # Include decorators if present
            if hasattr(node, 'decorator_list') and node.decorator_list:
                first_decorator = node.decorator_list[0]
                start_line = first_decorator.lineno

            # Extract the code (line numbers are 1-indexed)
            chunk_lines = lines[start_line - 1:end_line]
            content = '\n'.join(chunk_lines)

            return CodeChunk(
                content=content,
                file_path=file_path,
                chunk_index=chunk_index,
                chunk_type=chunk_type,
                name=node.name,
                start_line=start_line,
                end_line=end_line
            )
        except Exception as e:
            logger.error(f"Error extracting chunk: {e}")
            return None


def chunk_code(file_path: str, content: str) -> List[CodeChunk]:
    """
    Convenience function to chunk code.

    Args:
        file_path: Path to the file
        content: File content

    Returns:
        List of CodeChunk objects
    """
    chunker = CodeChunker()
    return chunker.chunk_file(file_path, content)
