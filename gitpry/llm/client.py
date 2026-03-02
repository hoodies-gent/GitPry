import json
import httpx
from typing import Iterator, List, Dict, Optional
from gitpry.utils.logger import logger

from gitpry.config import settings

def stream_ollama(prompt: str, system: str, model: str = None) -> Optional[Iterator[str]]:
    """
    Sends a prompt to the local Ollama instance and yields response chunks.
    Gracefully degrades if the server is unreachable or the model is missing.
    """
    target_model = model or settings.llm.model
    # The client handles appending the routing implementation detail (/api/generate)
    # The user config only specifies the machine endpoint.
    api_url = f"{settings.llm.base_url.rstrip('/')}/api/generate"
    
    payload = {
        "model": target_model,
        "prompt": prompt,
        "system": system,
        "stream": True,
        "options": {
            "temperature": settings.llm.temperature,
        }
    }

    try:
        # We use a 30s connection timeout to allow Ollama time to load the model into VRAM (cold start),
        # and a dynamic configurable read timeout for token generation.
        with httpx.stream("POST", api_url, json=payload, timeout=httpx.Timeout(30.0, read=settings.llm.timeout)) as response:
            if response.status_code == 404:
                logger.error(f"✗ Model '{model}' not found in local Ollama. Please run: ollama run {model}")
                return None
            
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        yield chunk["response"]

    except httpx.ConnectError:
        logger.error("✗ Could not connect to Ollama. Ensure the Ollama app is running locally (localhost:11434).")
        return None
    except httpx.TimeoutException:
        logger.error("✗ Ollama took too long to respond. The model might be too large for current system memory.")
        return None
    except Exception as e:
        logger.error(f"✗ Unexpected error communicating with Ollama: {str(e)}")
        return None


def stream_ollama_chat(
    messages: List[Dict[str, str]],
    model: str = None,
) -> Optional[Iterator[str]]:
    """
    Send a multi-turn conversation to Ollama's /api/chat endpoint and stream the response.

    Args:
        messages: Standard chat messages list, e.g.:
            [
                {"role": "system",    "content": "You are ..."},
                {"role": "user",      "content": "Q1 with context"},
                {"role": "assistant", "content": "A1"},
                {"role": "user",      "content": "Q2 with context"},   # current turn
            ]
        model: Override model name (defaults to settings.llm.model).

    Yields:
        Response token strings as they stream in.

    Returns None on connection failure or timeout.
    """
    target_model = model or settings.llm.model
    api_url = f"{settings.llm.base_url.rstrip('/')}/api/chat"

    payload = {
        "model": target_model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": settings.llm.temperature,
        },
    }

    try:
        with httpx.stream(
            "POST", api_url, json=payload,
            timeout=httpx.Timeout(30.0, read=settings.llm.timeout)
        ) as response:
            if response.status_code == 404:
                logger.error(f"✗ Model '{target_model}' not found. Run: ollama pull {target_model}")
                return None
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    # /api/chat wraps tokens under chunk["message"]["content"]
                    msg = chunk.get("message", {})
                    content = msg.get("content", "")
                    if content:
                        yield content

    except httpx.ConnectError:
        logger.error("✗ Could not connect to Ollama. Ensure the Ollama app is running locally.")
        return None
    except httpx.TimeoutException:
        logger.error("✗ Ollama timed out during chat response.")
        return None
    except Exception as e:
        logger.error(f"✗ Unexpected Ollama chat error: {str(e)}")
        return None
