"""
Responsible for converting raw commit dictionaries into embeddable text chunks.

Chunking Strategy:
- Each commit produces a "header" chunk: hash + author + date + message.
- If a diff is present, it is split into per-file "diff" chunks (max ~400 tokens each).
  This prevents huge diffs (e.g., package-lock.json) from swamping a single embedding.
"""
from typing import List, Dict
from gitpry.git_utils.repository import count_tokens
from gitpry.config import settings

def _get_max_chunk_tokens() -> int:
    return settings.rag.max_chunk_tokens


def _chunk_diff(diff_text: str) -> List[str]:
    """
    Split a large diff text into smaller per-file chunks.
    Tries to break at file boundaries (lines starting with 'diff --git').
    Falls back to token-based chunking if no file boundaries are found.
    """
    if not diff_text:
        return []

    file_sections = []
    current_section = []

    for line in diff_text.split("\n"):
        if line.startswith("diff --git") and current_section:
            file_sections.append("\n".join(current_section))
            current_section = [line]
        else:
            current_section.append(line)

    if current_section:
        file_sections.append("\n".join(current_section))

    result_chunks = []
    max_chunk_tokens = _get_max_chunk_tokens()
    for section in file_sections:
        if count_tokens(section) <= max_chunk_tokens:
            result_chunks.append(section)
        else:
            lines = section.split("\n")
            temp_chunk = []
            temp_tokens = 0
            for line in lines:
                line_tokens = count_tokens(line) + 1
                if temp_tokens + line_tokens > max_chunk_tokens and temp_chunk:
                    result_chunks.append("\n".join(temp_chunk))
                    temp_chunk = [line]
                    temp_tokens = line_tokens
                else:
                    temp_chunk.append(line)
                    temp_tokens += line_tokens
            if temp_chunk:
                result_chunks.append("\n".join(temp_chunk))

    return result_chunks


def commits_to_chunks(commits: List[Dict], branch: str = "HEAD") -> List[Dict]:
    """
    Convert a list of commit dicts into a list of embeddable chunk dicts.
    Each returned chunk dict has:
      - chunk_id:          Unique identifier
      - commit_hash:       Full 40-char SHA
      - commit_hash_short: First 8 chars
      - author:            Commit author
      - date:              Formatted date string
      - message:           Commit message
      - chunk_text:        The actual text to embed
      - chunk_type:        "header" or "diff"
      - branch:            Branch name used when this chunk was indexed
    """
    all_chunks = []

    for commit in commits:
        full_hash = commit.get("full_hash", commit["hash"])
        short_hash = commit["hash"]

        # --- Header Chunk (always produced) ---
        header_text = (
            f"Commit: {short_hash}\n"
            f"Author: {commit['author']}\n"
            f"Date: {commit['date']}\n"
            f"Message: {commit['message']}"
        )
        all_chunks.append({
            "chunk_id": f"{full_hash}_header",
            "commit_hash": full_hash,
            "commit_hash_short": short_hash,
            "author": commit["author"],
            "date": commit["date"],
            "message": commit["message"],
            "chunk_text": header_text,
            "chunk_type": "header",
            "branch": branch,
        })

        # --- Diff Chunks (only if diff data is present) ---
        if commit.get("diff"):
            diff_sub_chunks = _chunk_diff(commit["diff"])
            for idx, diff_chunk in enumerate(diff_sub_chunks):
                diff_text = (
                    f"Commit: {short_hash} | {commit['message']}\n"
                    f"Diff Chunk {idx + 1}/{len(diff_sub_chunks)}:\n{diff_chunk}"
                )
                all_chunks.append({
                    "chunk_id": f"{full_hash}_diff_{idx}",
                    "commit_hash": full_hash,
                    "commit_hash_short": short_hash,
                    "author": commit["author"],
                    "date": commit["date"],
                    "message": commit["message"],
                    "chunk_text": diff_text,
                    "chunk_type": "diff",
                    "branch": branch,
                })


    return all_chunks
