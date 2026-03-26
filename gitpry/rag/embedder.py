"""
Handles generating text embeddings using Ollama's local embedding endpoint.
Model: nomic-embed-text (optimized for long-form text retrieval)
"""
import httpx
from typing import List
from gitpry.utils.logger import logger
from gitpry.config import settings

EMBED_MODEL = "nomic-embed-text"


def get_embedding(text: str) -> List[float]:
    """
    Generate a single embedding vector for the given text using Ollama.
    Returns an empty list on failure.
    """
    host = settings.llm.base_url
    url = f"{host}/api/embeddings"

    try:
        response = httpx.post(
            url,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=settings.llm.timeout,
        )
        response.raise_for_status()
        embedding = response.json().get("embedding", [])
        if not embedding:
            logger.warning(f"Ollama returned an empty embedding for text snippet.")
        return embedding
    except httpx.ConnectError:
        logger.error(f"Cannot connect to Ollama at {host}. Is Ollama running?")
    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama embedding request failed: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.error(f"Unexpected error during embedding: {e}")
    return []
