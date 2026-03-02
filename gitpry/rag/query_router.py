"""
gitpry/rag/query_router.py

Language-agnostic query intent classifier using local Ollama.

Makes a single, minimal LLM call asking for a one-word classification:
  "structured" → route to filtered git scan (count/date/author queries)
  "semantic"   → route to RAG vector search (conceptual queries)

Defaults to "semantic" on error or timeout (safe fallback).
"""
import httpx
from typing import Literal

QueryRoute = Literal["structured", "semantic", "conversational"]

_CLASSIFY_PROMPT = """\
Classify a git history question as either "structured" or "semantic".

Rules:
- structured: LISTS or SHOWS commits by COUNT (show me N, last N), DATE (yesterday, this week), or AUTHOR (by John)
  Examples: "show me the last 5 commits", "last 3", "what changed yesterday", "commits by John",
            "list recent commits", "显示最近5个commit", "昨天改了什么", "最新的3条"
- semantic: asks WHY/HOW/WHEN something happened involving specific code or features
  Examples: "when did we add the login feature", "why was auth rewritten"
- conversational: asks for TOTALS/COUNTS (how many), or is a conversational greeting/follow-up
  Examples: "how many commits do we have", "who are the contributors", "hello", "explain that again"

Key rule: "last N commits" = structured. "how many commits" = conversational (answered by statistics).

Reply with ONLY one lowercase word — either: structured, semantic, or conversational

Question: {question}
Answer:"""


def classify_query(
    question: str,
    base_url: str = "http://localhost:11434",
    model: str = "qwen2.5-coder:7b",
    timeout: float = 8.0,
) -> QueryRoute:
    """
    Classify a user question as 'structured' or 'semantic' using a local LLM.

    Language-agnostic: works for any language the local model understands.
    Falls back to 'semantic' on timeout or any error.
    """
    try:
        response = httpx.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": _CLASSIFY_PROMPT.format(question=question),
                "stream": False,
                "options": {"temperature": 0, "num_predict": 5},  # One-word, deterministic
            },
            timeout=timeout,
        )
        response.raise_for_status()
        text = response.json().get("response", "").strip().lower()
        if "structured" in text:
            return "structured"
        if "conversational" in text:
            return "conversational"
        return "semantic"
    except Exception:
        return "semantic"  # Safe fallback — RAG is always correct-ish
