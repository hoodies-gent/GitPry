"""
Provides surgical, line-level code attribution tools.
Used by the MCP server to answer "why was this specific line written?"
"""

import git
from typing import Optional, Dict, List
from gitpry.utils.logger import logger

def get_file_blame(
    repo_path: str, 
    filepath: str, 
    start_line: Optional[int] = None, 
    end_line: Optional[int] = None
) -> str:
    """
    Runs git blame on a specific file (and optional line range),
    extracts the unique commit hashes responsible for the current state,
    and returns a clean, token-efficient mapping of the origin commit contexts.
    """
    try:
        repo = git.Repo(repo_path)
    except Exception as e:
        logger.error(f"Failed to open repo for blame: {e}")
        return f"Error: Failed to open repository at {repo_path}"

    args = []
    if start_line is not None and end_line is not None:
        args.extend(["-L", f"{start_line},{end_line}"])
    elif start_line is not None:
        args.extend(["-L", f"{start_line},{start_line}"])
    
    args.extend(["--line-porcelain", filepath])

    try:
        blame_output = repo.git.blame(*args)
    except Exception as e:
        logger.error(f"Failed to execute git blame on {filepath}: {e}")
        return f"Error: Failed to execute git blame on {filepath}. Make sure the file exists and is tracked by Git."

    if not blame_output:
        return f"No blame information found for {filepath}."

    unique_hashes = set()
    for line in blame_output.split("\n"):
        if not line:
            continue
        parts = line.split(" ")
        if len(parts[0]) == 40 and all(c in '0123456789abcdefABCDEF' for c in parts[0]):
            unique_hashes.add(parts[0])

    if not unique_hashes:
        return "Could not extract origin commits from the blame output."

    result_blocks = []
    result_blocks.append(f"Blame Analysis for `{filepath}`" + 
                         (f" (Lines {start_line}-{end_line})" if start_line else "") + 
                         ":\n")
    
    result_blocks.append(f"Found {len(unique_hashes)} unique commit(s) responsible for these lines:\n")

    for commit_hash in unique_hashes:
        try:
            commit = repo.commit(commit_hash)
            short_hash = commit.hexsha[:8]
            author = str(commit.author)
            date = commit.committed_datetime.strftime("%Y-%m-%d %H:%M")
            subject = commit.message.strip().split("\n")[0]
            
            message_lines = commit.message.strip().split("\n")
            body = "\n".join(message_lines[1:]).strip() if len(message_lines) > 1 else ""
            
            block = f"- Commit: [{short_hash}]\n  Author: {author} @ {date}\n  Summary: {subject}"
            if body:
                indented_body = "\n".join([f"    {line}" for line in body.split("\n")])
                block += f"\n  Details:\n{indented_body}"
            
            result_blocks.append(block)
        except Exception as e:
            logger.debug(f"Failed to fetch commit details for {commit_hash}: {e}")
            result_blocks.append(f"- Commit: [{commit_hash[:8]}] (Details unavailable)")

    return "\n\n".join(result_blocks)
