import json
import httpx
from typing import Iterator, Optional
from gitpry.utils.logger import logger

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def stream_ollama(prompt: str, system: str, model: str = "qwen2.5-coder:7b") -> Optional[Iterator[str]]:
    """
    Sends a prompt to the local Ollama instance and yields response chunks.
    Gracefully degrades if the server is unreachable or the model is missing.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": True,
        "options": {
            "temperature": 0.1,  # Keep reasoning deterministic
        }
    }

    try:
        # We use a short connection timeout to fail fast if Ollama isn't running,
        # but a long read timeout since local models might take time to process the first token.
        with httpx.stream("POST", OLLAMA_API_URL, json=payload, timeout=httpx.Timeout(10.0, read=60.0)) as response:
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
