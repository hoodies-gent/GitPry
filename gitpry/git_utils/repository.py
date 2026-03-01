from typing import List, Dict, Optional
import git
from git.exc import InvalidGitRepositoryError, NoSuchPathError
from gitpry.utils.logger import logger
from gitpry.config import settings

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

def build_prompt_context(commits: List[Dict]) -> str:
    """
    Convert a list of commit dictionaries into the structured text format, embedding diffs if available.
    """
    blocks = []
    for c in commits:
        header = f"[{c['hash']}] {c['author']} @ {c['date']}\nMessage: {c['message']}"
        if c.get("diff"):
            diff_block = f"Diff Details:\n```diff\n{c['diff']}\n```"
            blocks.append(f"{header}\n{diff_block}")
        else:
            blocks.append(header)
            
    # Separate independent commit records with clear demarcations
    return "\n\n---\n\n".join(blocks)
