from typing import List, Dict, Optional
import git
import tiktoken
from git.exc import InvalidGitRepositoryError, NoSuchPathError
from gitpry.utils.logger import logger
from gitpry.config import settings

def count_tokens(text: str) -> int:
    """Estimate the number of tokens in a string using cl100k_base encoding."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback estimation if tiktoken fails
        return len(text) // 4

def get_repo_stats(repo_path: str = ".", scan_limit: int = 10000) -> dict:
    """
    Compute aggregate repository statistics for the metadata block.
    Scans up to scan_limit commits to build ground-truth counters.
    Returns a dict with total_commits, date_range, authors, current_branch.
    """
    try:
        repo = git.Repo(repo_path)
    except Exception:
        return {}

    try:
        current_branch = repo.active_branch.name
    except TypeError:
        current_branch = repo.head.commit.hexsha[:8] + " (detached HEAD)"

    authors: dict[str, int] = {}
    earliest_date = None
    latest_date = None
    total = 0

    for commit in repo.iter_commits('HEAD', max_count=scan_limit):
        total += 1
        author = str(commit.author)
        authors[author] = authors.get(author, 0) + 1
        dt = commit.committed_datetime
        if earliest_date is None or dt < earliest_date:
            earliest_date = dt
        if latest_date is None or dt > latest_date:
            latest_date = dt

    sorted_authors = sorted(authors.items(), key=lambda x: x[1], reverse=True)

    return {
        "total_commits": total,
        "current_branch": current_branch,
        "date_range": (
            f"{earliest_date.strftime('%Y-%m-%d')} → {latest_date.strftime('%Y-%m-%d')}"
            if earliest_date and latest_date else "unknown"
        ),
        "top_authors": sorted_authors[:5],
        "total_authors": len(authors),
    }

def format_repo_stats_block(stats: dict) -> str:
    """
    Format repo stats as a compact context block to prepend to every prompt.
    Gives the LLM ground-truth for aggregate queries.
    """
    if not stats:
        return ""

    authors_str = ", ".join(
        f"{name} ({count})" for name, count in stats.get("top_authors", [])
    )
    return (
        f"[Repository Overview]\n"
        f"Current Branch: {stats.get('current_branch', 'unknown')}\n"
        f"Total Commits (this branch): {stats.get('total_commits', '?')}\n"
        f"Date Range: {stats.get('date_range', 'unknown')}\n"
        f"Contributors: {authors_str}\n"
        f"Total Authors: {stats.get('total_authors', '?')}\n"
    )

def get_recent_commits(repo_path: str = ".", limit: int = 500) -> Optional[List[Dict]]:
    """
    Extract the most recent commits from the specified repository path.
    Defensively checks if the directory is a valid git repository.
    
    Returns a list of commit dictionaries, or None if the repository is invalid.
    """
    try:
        repo = git.Repo(repo_path)
    except (InvalidGitRepositoryError, NoSuchPathError):
        # Graceful degradation: Log a friendly error instead of crashing
        logger.error(f"✗ The path '{repo_path}' is not a valid git repository.")
        return None

    commits = []
    
    # Extract commits iteratively to avoid loading massive history into memory at once
    try:
        for commit in repo.iter_commits('HEAD', max_count=limit):
            commit_data = {
                "hash": commit.hexsha[:8],
                "author": str(commit.author),
                "date": commit.committed_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                "message": commit.message.strip().replace("\n", " "), # Flatten multiline messages
            }
            
            # Fetch associated diff if enabled
            if settings.git.include_diff:
                try:
                    # Extracts both the file change stats and the actual patch, but strips the duplicated commit log format part
                    diff_text = repo.git.show(commit.hexsha, "--stat", "-p", "--format=")
                    diff_text = diff_text.strip()
                    
                    if diff_text:
                        diff_lines = diff_text.split("\n")
                        # Truncate to prevent enormous diffs (e.g. package-lock.json) from blowing out token limits
                        if len(diff_lines) > settings.git.max_diff_lines:
                            commit_data["diff"] = "\n".join(diff_lines[:settings.git.max_diff_lines]) + f"\n... [{len(diff_lines) - settings.git.max_diff_lines} lines truncated]"
                        else:
                            commit_data["diff"] = diff_text
                except Exception as diff_e:
                    logger.debug(f"Failed to extract diff for {commit.hexsha[:8]}: {str(diff_e)}")

            commits.append(commit_data)
    except Exception as e:
        logger.error(f"Failed to read commit history: {str(e)}")
        return None
        
    logger.debug(f"Successfully extracted {len(commits)} commits from '{repo_path}'.")
    return commits

def build_prompt_context(commits: List[Dict], max_tokens: int = None, base_tokens: int = 0) -> tuple[str, int]:
    """
    Convert a list of commit dictionaries into the structured text format, embedding diffs if available.
    If max_tokens is provided, older commits (at the end of the list) will be truncated
    to ensure the total prompt size stays safely within the LLM context window limits.
    Returns the built context string and the number of commits included.
    """
    blocks = []
    current_tokens = base_tokens
    
    for c in commits:
        header = f"[{c['hash']}] {c['author']} @ {c['date']}\nMessage: {c['message']}"
        if c.get("diff"):
            diff_block = f"Diff Details:\n```diff\n{c['diff']}\n```"
            block = f"{header}\n{diff_block}"
        else:
            block = header
            
        if max_tokens is not None:
            # +3 for the \n\n---\n\n separator
            block_tokens = count_tokens(block) + 3
            if current_tokens + block_tokens > max_tokens:
                logger.warning(f"⚠ Token limit reached ({current_tokens}/{max_tokens}). Truncated {len(commits) - len(blocks)} older commits to prevent overflow.")
                break
            current_tokens += block_tokens
            
        blocks.append(block)
            
    # Reverse the blocks array so the LLM reads them in chronological order
    # (Oldest at the top, Newest (HEAD) at the absolute bottom).
    # This prevents the LLM from confusing "last commit" in the text with the oldest commit.
    blocks.reverse()
    
    # Separate independent commit records with clear demarcations
    return "\n\n---\n\n".join(blocks), len(blocks)
