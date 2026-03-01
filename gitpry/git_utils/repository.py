from typing import List, Dict, Optional
import git
from git.exc import InvalidGitRepositoryError, NoSuchPathError
from gitpry.utils.logger import logger

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
            commits.append({
                "hash": commit.hexsha[:8],
                "author": str(commit.author),
                "date": commit.committed_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                "message": commit.message.strip().replace("\n", " "), # Flatten multiline messages
            })
    except Exception as e:
        logger.error(f"Failed to read commit history: {str(e)}")
        return None
        
    logger.debug(f"Successfully extracted {len(commits)} commits from '{repo_path}'.")
    return commits

def build_prompt_context(commits: List[Dict]) -> str:
    """
    Convert a list of commit dictionaries into the structured text format.
    """
    lines = []
    for c in commits:
        line = f"{c['hash']} | {c['author']} | {c['date']} | {c['message']}"
        lines.append(line)
        
    return "\n".join(lines)
