"""
Ollama embedding integration.
"""
import requests
import logging
from typing import List

from .config import OLLAMA_BASE_URL, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


def get_embedding(text: str) -> List[float]:
    """
    Generate embedding for a single text using Ollama.

    Args:
        text: Text to embed

    Returns:
        List of floats representing the embedding vector

    Raises:
        EmbeddingError: If the API call fails
    """
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {
        "model": EMBEDDING_MODEL,
        "prompt": text
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        if "embedding" not in result:
            raise EmbeddingError(f"No embedding in response: {result}")

        return result["embedding"]

    except requests.exceptions.ConnectionError:
        raise EmbeddingError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            "Make sure Ollama is running."
        )
    except requests.exceptions.Timeout:
        raise EmbeddingError("Embedding request timed out")
    except requests.exceptions.RequestException as e:
        raise EmbeddingError(f"Embedding request failed: {e}")


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts.

    Note: Ollama doesn't have a native batch endpoint,
    so we process sequentially.

    Args:
        texts: List of texts to embed

    Returns:
        List of embedding vectors
    """
    embeddings = []
    for i, text in enumerate(texts):
        logger.debug(f"Generating embedding {i+1}/{len(texts)}")
        embedding = get_embedding(text)
        embeddings.append(embedding)
    return embeddings


def check_ollama_connection() -> bool:
    """
    Check if Ollama is running and accessible.

    Returns:
        True if Ollama is accessible, False otherwise
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def check_model_available(model_name: str) -> bool:
    """
    Check if a specific model is available in Ollama.

    Args:
        model_name: Name of the model to check

    Returns:
        True if model is available, False otherwise
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            return False

        models = response.json().get("models", [])
        model_names = [m.get("name", "").split(":")[0] for m in models]
        return model_name in model_names
    except requests.exceptions.RequestException:
        return False
