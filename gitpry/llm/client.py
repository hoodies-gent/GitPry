import json
import httpx
from typing import Iterator, Optional
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
        # We use a short connection timeout to fail fast if Ollama isn't running,
        # but a dynamic configurable read timeout.
        with httpx.stream("POST", api_url, json=payload, timeout=httpx.Timeout(10.0, read=settings.llm.timeout)) as response:
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
