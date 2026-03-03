"""
gitpry/git_utils/scanner.py

Structured query executor: parses filter intent from the user question
and runs a filtered git scan directly (no vector search).

Handles:
  - Count/recency: "last N commits"
  - Time-based:    "yesterday", "this week", "last month"
  - Author-based:  "by <name>", "from <name>"

Returns a (context_str, description) tuple that plugs directly into
build_user_prompt() the same way the RAG path does.
"""
import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
import git
from gitpry.utils.logger import logger


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_limit(question: str, default: int = 10) -> int:
    """Extract an explicit N from 'last N', 'show me N', etc."""
    m = re.search(r'\b(?:last|latest|recent|first|show(?:\s+me)?)\s+(\d+)\b', question, re.IGNORECASE)
    return int(m.group(1)) if m else default


def _extract_author(question: str, repo_path: str = ".") -> Optional[str]:
    """Extract author name by matching the question against known git authors.
    
    Queries `git shortlog -s` to get the list of exact known authors in the repo.
    This safely handles multi-word names (like "Yifan Ke") and special characters 
    without relying on fragile regex.
    """
    try:
        repo = git.Repo(repo_path)
        # Get all authors: "   123  Yifan Ke"
        shortlog = repo.git.shortlog("-s", "HEAD")
        
        # Extract just the names
        authors = []
        for line in shortlog.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Split by the first whitespace block after the commit count
            parts = line.split("\t", 1)
            if len(parts) == 2:
                authors.append(parts[1].strip())
            else:
                # Fallback if tab parsing fails
                name = re.sub(r'^\d+\s+', '', line)
                authors.append(name.strip())
                
        # Sort by length descending, so "Yifan Ke" matches before just "Yifan"
        authors.sort(key=len, reverse=True)
        
        q_lower = question.lower()
        for author in authors:
            if not author:
                continue
            # Look for exact substring match of the author's name in the query
            if author.lower() in q_lower:
                return author
    except Exception as e:
        logger.debug(f"Failed to extract author via shortlog: {e}")
        
    # Fallback to the old basic regex if git fails
    m = re.search(r'\b(?:by|from)\s+([A-Za-z]+)\b', question, re.IGNORECASE)
    return m.group(1) if m else None


def _extract_since(question: str) -> Optional[datetime]:
    """Convert time keywords to an absolute datetime boundary (UTC)."""
    now = datetime.now(timezone.utc)
    q = question.lower()
    if "yesterday" in q:
        return (now - timedelta(days=1)).replace(hour=0, minute=0, second=0)
    if "today" in q:
        return now.replace(hour=0, minute=0, second=0)
    if "this week" in q:
        return (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0)
    if "last week" in q:
        start = now - timedelta(days=now.weekday() + 7)
        return start.replace(hour=0, minute=0, second=0)
    if "this month" in q:
        return now.replace(day=1, hour=0, minute=0, second=0)
    if "last month" in q:
        first_this = now.replace(day=1)
        last_month_end = first_this - timedelta(days=1)
        return last_month_end.replace(day=1, hour=0, minute=0, second=0)
    m = re.search(r'in\s+the\s+last\s+(\d+)\s+(day|week|month)s?', q)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {"day": timedelta(days=n), "week": timedelta(weeks=n), "month": timedelta(days=n * 30)}[unit]
        return now - delta
    return None


def _format_commit(commit) -> str:
    """Format a single gitpython commit object into a readable block."""
    return (
        f"[{commit.hexsha[:8]}] {str(commit.author)} @ "
        f"{commit.committed_datetime.strftime('%Y-%m-%d %H:%M')}\n"
        f"Message: {commit.message.strip().splitlines()[0]}"
    )


# ── Main entry ────────────────────────────────────────────────────────────────

def scan_structured(
    question: str,
    repo_path: str = ".",
    branch: str = "HEAD",
    max_results: int = 500,
) -> Tuple[str, str]:
    """
    Execute a structured filtered git scan based on the user question.

    Returns:
        context_str:  Formatted commit blocks for the LLM prompt.
        description:  Human-readable summary of the filters applied.
    """
    try:
        repo = git.Repo(repo_path)
    except Exception as e:
        logger.error(f"Failed to open repo: {e}")
        return "", "unknown filters"

    limit = _extract_limit(question, default=10)
    author_filter = _extract_author(question, repo_path)
    since = _extract_since(question)

    parts = []
    if since:
        parts.append(f"since {since.strftime('%Y-%m-%d')}")
    if author_filter:
        parts.append(f"author ≈ '{author_filter}'")
    parts.append(f"limit {limit}")
    description = ", ".join(parts)

    kwargs = {"max_count": max_results}
    if since:
        kwargs["after"] = since.strftime("%Y-%m-%d")

    collected = []
    try:
        for commit in repo.iter_commits(branch, **kwargs):
            if author_filter and author_filter.lower() not in str(commit.author).lower():
                continue
            collected.append(_format_commit(commit))
            if len(collected) >= limit:
                break
    except Exception as e:
        logger.error(f"Structured scan failed: {e}")
        return "", description

    if not collected:
        return "(No commits matched the specified filters.)", description

    context_str = "\n\n---\n\n".join(collected)
    return context_str, description
