"""
Model Context Protocol (MCP) Server for GitPry.
Exposes core Git history extraction and RAG retrieval capabilities as Tools 
for external AI Agents (like Claude Desktop, Cursor, or Windsurf).

Tools exposed:
1. get_repo_stats: Return high-level repository aggregation
2. semantic_search: Query the local LanceDB vector store for relevant chunks
3. git_log_scan: Perform a structured metadata search (author, date, limit)
4. get_commit_diff: Extract the full code diff for a specific commit hash
"""
import os
from typing import Optional, Any
from mcp.server.fastmcp import FastMCP
from gitpry.utils.logger import logger

# Initialize the FastMCP server
mcp = FastMCP("GitPry", dependencies=["gitpython", "lancedb", "httpx"])

@mcp.tool()
def get_repo_stats(repo_path: str = ".", branch: str = "HEAD") -> str:
    """
    Get high-level repository statistics including total commits, top authors, 
    and date range for a specific branch. Use this when the user asks 
    aggregate questions like 'how many commits do we have?' or 'who are the top contributors?'
    """
    from gitpry.git_utils.repository import get_repo_stats, format_repo_stats_block
    stats = get_repo_stats(repo_path=repo_path, branch=branch)
    if not stats:
        return f"Error: Could not retrieve stats for repository at {repo_path} (branch: {branch}). Is it a valid Git repository?"
    return format_repo_stats_block(stats)

@mcp.tool()
def semantic_search(query: str, repo_path: str = ".", branch: Optional[str] = None, top_k: int = 5) -> str:
    """
    Search the local GitPry vector index for commit chunks conceptually related to a natural language query.
    Use this for "WHY" or "HOW" questions like "why was the auth module rewritten?" or "when did we add the login feature?".
    Requires `git pry index` to have been run previously on the repository.
    """
    from gitpry.rag.embedder import get_embedding
    from gitpry.rag.vector_store import search_similar

    query_vector = get_embedding(query)
    if not query_vector:
        return "Error: Failed to generate query embedding. Make sure Ollama or the embedding backend is running."

    results = search_similar(repo_path, query_vector, top_k=top_k, branch_filter=branch)
    if not results:
        branch_msg = f" for branch '{branch}'" if branch else ""
        return f"No semantic matches found in the index{branch_msg}. Please ensure `git pry index` has been run."

    # Format the results into a readable context block
    blocks = []
    for r in results:
        block = (
            f"[{r['commit_hash_short']}] {r['author']} @ {r['date']}\n"
            f"Message: {r['message']}\n"
            f"Relevant Chunk:\n{r['chunk_text']}"
        )
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)

@mcp.tool()
def git_log_scan(repo_path: str = ".", branch: str = "HEAD", limit: int = 10, author: Optional[str] = None, since: Optional[str] = None) -> str:
    """
    Scan recent Git commits using strict metadata filters. 
    Use this for structurally bounded queries like "show me the last 5 commits by John" or "what changed yesterday".
    Does NOT search the message body or code diffs, only the commit metadata.
    `since` should refer to an absolute or relative time recognizable by Git (e.g. 'yesterday', '2024-01-01', '1 week ago').
    """
    import git
    from gitpry.git_utils.scanner import _format_commit
    try:
        repo = git.Repo(repo_path)
    except Exception as e:
        return f"Error: Invalid git repository at {repo_path} ({e})"

    kwargs: dict[str, Any] = {"max_count": limit}
    if since:
        kwargs["after"] = since
    if author:
        kwargs["author"] = author

    collected = []
    try:
        for commit in repo.iter_commits(branch, **kwargs):
            collected.append(_format_commit(commit))
    except Exception as e:
        return f"Error executing git log block: {e}"

    if not collected:
        return "(No commits matched the specified filters.)"

    return "\n\n---\n\n".join(collected)

@mcp.tool()
def get_commit_diff(commit_hash: str, repo_path: str = ".") -> str:
    """
    Extract the full patch diff for a specific commit hash.
    Use this when you need to see exactly what code lines were added/removed in a given commit.
    Provide the full or short hash (e.g., 'a1b2c3d4').
    """
    import git
    try:
        repo = git.Repo(repo_path)
    except Exception as e:
        return f"Error: Invalid git repository at {repo_path} ({e})"

    try:
        # Extract the commit patch but do not include the commit metadata header again 
        # (format= suppresses the log header, leaving only the diff)
        diff_text = repo.git.show(commit_hash, "--stat", "-p", "--format=")
        return diff_text.strip() if diff_text else "(Empty diff for this commit)"
    except Exception as e:
        return f"Error extracting diff for commit {commit_hash}: {e}"

@mcp.tool()
def get_file_blame(filepath: str, start_line: Optional[int] = None, end_line: Optional[int] = None, repo_path: str = ".") -> str:
    """
    Surgically inspect a file to understand why specific lines of code were written.
    Returns the origin commit messages and authors for the given file or line range.
    Use this when you need to understand the historical context or intent behind a specific code block.
    """
    from gitpry.git_utils.blame import get_file_blame as builtin_blame
    return builtin_blame(repo_path, filepath, start_line, end_line)

@mcp.tool()
def compare_branches(base: str, target: str, repo_path: str = ".") -> str:
    """
    Analyze the divergence between two branches (e.g., 'main' vs 'feature').
    Returns a structural summary of the unique commits, their intents, and aggregate diff stats introduced in the target branch.
    Use this to understand the macro-level intent and scope of an entire Pull Request before diving into specific files.
    """
    from gitpry.git_utils.repository import compare_branches as builtin_compare
    return builtin_compare(repo_path, base, target)

def serve_stdio():
    """Start the MCP server using standard input/output transport."""
    mcp.run()
