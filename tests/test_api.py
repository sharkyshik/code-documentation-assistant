"""
Tests for the FastAPI endpoints.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check_ollama_down(self, client):
        """Test health check when Ollama is not running."""
        with patch('app.main.check_ollama_connection', return_value=False):
            with patch('app.main.check_model_available', return_value=False):
                response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["ollama_connected"] is False


class TestIngestEndpoint:
    """Tests for /ingest endpoint."""

    def test_ingest_nonexistent_directory(self, client):
        """Test ingestion with non-existent directory."""
        response = client.post(
            "/ingest",
            json={"directory_path": "/nonexistent/path/12345"}
        )

        assert response.status_code == 400
        assert "does not exist" in response.json()["detail"]


class TestQueryEndpoint:
    """Tests for /query endpoint."""

    def test_query_no_documents(self, client):
        """Test query when no documents are indexed."""
        # Clear the vector store first
        with patch('app.main.get_vector_store') as mock_store:
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_store.return_value.collection = mock_collection

            response = client.post(
                "/query",
                json={"question": "What does this code do?"}
            )

        assert response.status_code == 400
        assert "No documents indexed" in response.json()["detail"]


class TestStatsEndpoint:
    """Tests for /stats endpoint."""

    def test_stats(self, client):
        """Test getting statistics."""
        response = client.get("/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert "collection_name" in data


class TestClearEndpoint:
    """Tests for /clear endpoint."""

    def test_clear(self, client):
        """Test clearing indexed data."""
        response = client.delete("/clear")

        assert response.status_code == 200
        assert "cleared" in response.json()["message"].lower()
