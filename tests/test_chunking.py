"""
Tests for the chunking module.
"""
import pytest
from app.chunking import CodeChunker, chunk_code, CodeChunk


class TestCodeChunker:
    """Tests for CodeChunker class."""

    def test_chunk_function(self):
        """Test chunking a simple function."""
        code = '''
def hello_world():
    """A simple function."""
    print("Hello, World!")
    return True
'''
        chunker = CodeChunker(min_chunk_size=10)
        chunks = chunker.chunk_file("test.py", code)

        assert len(chunks) == 1
        assert chunks[0].chunk_type == "function"
        assert chunks[0].name == "hello_world"
        assert chunks[0].file_path == "test.py"
        assert "def hello_world" in chunks[0].content

    def test_chunk_class(self):
        """Test chunking a class definition."""
        code = '''
class Calculator:
    """A simple calculator."""

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
'''
        chunker = CodeChunker(min_chunk_size=10)
        chunks = chunker.chunk_file("calc.py", code)

        # Should find the class and its methods
        chunk_types = [c.chunk_type for c in chunks]
        assert "class" in chunk_types

        # Find the class chunk
        class_chunk = next(c for c in chunks if c.chunk_type == "class")
        assert class_chunk.name == "Calculator"

    def test_chunk_with_decorator(self):
        """Test that decorators are included with functions."""
        code = '''
@staticmethod
def decorated_function():
    pass
'''
        chunker = CodeChunker(min_chunk_size=10)
        chunks = chunker.chunk_file("test.py", code)

        assert len(chunks) == 1
        assert "@staticmethod" in chunks[0].content

    def test_empty_file(self):
        """Test handling of empty file."""
        chunker = CodeChunker()
        chunks = chunker.chunk_file("empty.py", "")

        assert len(chunks) == 0

    def test_syntax_error_fallback(self):
        """Test fallback for files with syntax errors."""
        code = '''
def broken_function(
    # Missing closing paren
    pass
'''
        chunker = CodeChunker(min_chunk_size=10)
        chunks = chunker.chunk_file("broken.py", code)

        # Should fall back to whole file as one chunk
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "module"

    def test_multiple_functions(self):
        """Test chunking multiple functions."""
        code = '''
def func_one():
    return 1

def func_two():
    return 2

def func_three():
    return 3
'''
        chunker = CodeChunker(min_chunk_size=10)
        chunks = chunker.chunk_file("multi.py", code)

        function_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(function_chunks) == 3

        names = {c.name for c in function_chunks}
        assert names == {"func_one", "func_two", "func_three"}

    def test_min_chunk_size(self):
        """Test that chunks below min_chunk_size are filtered."""
        code = '''
def tiny():
    pass
'''
        # Set high min_chunk_size
        chunker = CodeChunker(min_chunk_size=1000)
        chunks = chunker.chunk_file("test.py", code)

        assert len(chunks) == 0

    def test_chunk_code_convenience_function(self):
        """Test the convenience function."""
        code = '''
def example():
    """Example function."""
    return 42
'''
        chunks = chunk_code("example.py", code)

        assert len(chunks) >= 1
        assert any(c.name == "example" for c in chunks)


class TestCodeChunk:
    """Tests for CodeChunk dataclass."""

    def test_code_chunk_creation(self):
        """Test creating a CodeChunk."""
        chunk = CodeChunk(
            content="def foo(): pass",
            file_path="test.py",
            chunk_index=0,
            chunk_type="function",
            name="foo",
            start_line=1,
            end_line=1
        )

        assert chunk.content == "def foo(): pass"
        assert chunk.file_path == "test.py"
        assert chunk.chunk_index == 0
        assert chunk.chunk_type == "function"
        assert chunk.name == "foo"

    def test_code_chunk_optional_fields(self):
        """Test CodeChunk with optional fields as None."""
        chunk = CodeChunk(
            content="# comment",
            file_path="test.py",
            chunk_index=0,
            chunk_type="module"
        )

        assert chunk.name is None
        assert chunk.start_line is None
        assert chunk.end_line is None
