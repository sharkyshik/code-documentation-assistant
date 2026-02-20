"""
Ollama LLM integration for response generation.
"""
import requests
import logging
from typing import Optional

from .config import OLLAMA_BASE_URL, LLM_MODEL, SYSTEM_PROMPT, QUERY_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when LLM generation fails."""
    pass


def generate_response(
    question: str,
    context: str,
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 0.1,
    max_tokens: int = 1024
) -> str:
    """
    Generate a response using Ollama.

    Args:
        question: User question
        context: Retrieved context from vector store
        system_prompt: System prompt for the LLM
        temperature: Sampling temperature (lower = more deterministic)
        max_tokens: Maximum tokens in response

    Returns:
        Generated response string

    Raises:
        LLMError: If generation fails
    """
    # Build the full prompt
    prompt = QUERY_PROMPT_TEMPLATE.format(
        system_prompt=system_prompt,
        context=context,
        question=question
    )

    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }

    try:
        logger.debug(f"Sending request to Ollama LLM")
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()

        if "response" not in result:
            raise LLMError(f"No response in LLM output: {result}")

        return result["response"].strip()

    except requests.exceptions.ConnectionError:
        raise LLMError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            "Make sure Ollama is running."
        )
    except requests.exceptions.Timeout:
        raise LLMError("LLM request timed out")
    except requests.exceptions.RequestException as e:
        raise LLMError(f"LLM request failed: {e}")


def generate_with_streaming(
    question: str,
    context: str,
    system_prompt: str = SYSTEM_PROMPT
):
    """
    Generate a response with streaming output.

    Yields response chunks as they are generated.

    Args:
        question: User question
        context: Retrieved context
        system_prompt: System prompt

    Yields:
        Response chunks as strings
    """
    prompt = QUERY_PROMPT_TEMPLATE.format(
        system_prompt=system_prompt,
        context=context,
        question=question
    )

    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.1,
            "num_predict": 1024
        }
    }

    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    import json
                    chunk = json.loads(line)
                    if "response" in chunk:
                        yield chunk["response"]
                    if chunk.get("done", False):
                        break
    except requests.exceptions.RequestException as e:
        raise LLMError(f"Streaming request failed: {e}")


def check_llm_ready() -> bool:
    """
    Check if the LLM model is loaded and ready.

    Returns:
        True if LLM is ready, False otherwise
    """
    try:
        # Try a minimal generation to check if model is loaded
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": "Hi",
                "stream": False,
                "options": {"num_predict": 1}
            },
            timeout=30
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
