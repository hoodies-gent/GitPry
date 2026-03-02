"""
gitpry/rag/vector_store.py

Manages the local LanceDB vector database for Git commit embeddings.

The database is stored at: ~/.gitpry/vectors/<repo_id>/
- repo_id is the first 16 chars of the SHA1 of the repo's remote URL (or its absolute path as fallback).
- This allows multiple different repos to have isolated vector stores.
"""
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
import lancedb
import pyarrow as pa
from gitpry.utils.logger import logger

# V0.3: added "branch" column. Existing V0.2 indexes MUST be rebuilt with --reindex.
SCHEMA_VERSION = "v0.3"

# LanceDB schema for each stored commit chunk
COMMIT_SCHEMA = pa.schema([
    pa.field("chunk_id", pa.string()),          # "{hash}_{chunk_index}"
    pa.field("commit_hash", pa.string()),        # Full 40-char SHA
    pa.field("commit_hash_short", pa.string()),  # First 8 chars
    pa.field("author", pa.string()),
    pa.field("date", pa.string()),
    pa.field("message", pa.string()),
    pa.field("chunk_text", pa.string()),         # The text content that was embedded
    pa.field("chunk_type", pa.string()),         # "header" | "diff"
    pa.field("branch", pa.string()),             # Branch name at index time (V0.3+)
    pa.field("vector", pa.list_(pa.float32())), # The embedding vector
])

TABLE_NAME = "commits"
GITPRY_HOME = Path.home() / ".gitpry" / "vectors"


def get_repo_id(repo_path: str = ".") -> str:
    """
    Derive a unique, stable identifier for the current git repository.
    Uses the remote URL if available, otherwise falls back to the absolute path.
    """
    try:
        import git
        repo = git.Repo(repo_path)
        if repo.remotes:
            identifier = repo.remotes.origin.url
        else:
            identifier = str(Path(repo_path).resolve())
    except Exception:
        identifier = str(Path(repo_path).resolve())
    return hashlib.sha1(identifier.encode()).hexdigest()[:16]


def get_db_path(repo_id: str) -> Path:
    db_path = GITPRY_HOME / repo_id
    db_path.mkdir(parents=True, exist_ok=True)
    return db_path


def _has_branch_column(table) -> bool:
    """Detect if the existing table has the V0.3 'branch' column."""
    try:
        schema = table.schema
        return "branch" in [field.name for field in schema]
    except Exception:
        return False


def check_schema_migration(db) -> bool:
    """
    Check if the existing table has an outdated schema (missing `branch` column).
    Returns True if migration is needed, False if schema is current or table doesn't exist.
    """
    if TABLE_NAME not in db.table_names():
        return False
    table = db.open_table(TABLE_NAME)
    return not _has_branch_column(table)


def drop_table(db):
    """Drop the commits table (used by --reindex)."""
    try:
        db.drop_table(TABLE_NAME)
        logger.debug("Dropped existing commits table for reindex.")
    except Exception as e:
        logger.error(f"Failed to drop table: {e}")


def open_or_create_table(db, vector_dim: int):
    """Open the commits table or create it with the correct schema."""
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)

    schema = pa.schema([
        pa.field("chunk_id", pa.string()),
        pa.field("commit_hash", pa.string()),
        pa.field("commit_hash_short", pa.string()),
        pa.field("author", pa.string()),
        pa.field("date", pa.string()),
        pa.field("message", pa.string()),
        pa.field("chunk_text", pa.string()),
        pa.field("chunk_type", pa.string()),
        pa.field("branch", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), vector_dim)),
    ])
    table = db.create_table(TABLE_NAME, schema=schema)
    # TODO(V0.3 - Performance): After inserting chunks, call table.create_index()
    # to build an ANN (Approximate Nearest Neighbor) index for faster vector search
    # on repos with > 5000 chunks.
    return table


def get_indexed_hashes(table) -> set:
    """
    Return the set of globally indexed full commit hashes for incremental dedup.

    DESIGN NOTE: Deduplication is always global (cross-branch).
    If commit X is already indexed under branch 'main', it will NOT be re-indexed
    when running `git pry index --branch feature/foo`, even if it's reachable from
    that branch. It keeps its original branch tag ('main').
    This is intentional — "first indexed branch wins."

    # TODO(V0.4 - Multi-branch tagging): To properly associate one chunk with multiple
    # branches, we'd need to store `branches: list[str]` and support in-place append.
    # That requires a delete+reinsert strategy since LanceDB has no native field-level update.
    """
    try:
        rows = table.search().select(["commit_hash"]).limit(100_000).to_list()
        return {row["commit_hash"] for row in rows}
    except Exception:
        return set()


def upsert_chunks(table, chunks: List[Dict]):
    """Add new commit chunks to the vector store table."""
    if not chunks:
        return
    try:
        table.add(chunks)
        logger.debug(f"Inserted {len(chunks)} chunks into vector store.")
    except Exception as e:
        logger.error(f"Failed to insert chunks into vector store: {e}")


def search_similar(
    repo_path: str,
    query_vector: List[float],
    top_k: int = 5,
    branch_filter: Optional[str] = None,
) -> List[Dict]:
    """
    Search the vector store for the top-K most semantically similar commit chunks.
    If branch_filter is set, only return chunks indexed from that branch.
    Returns a list of result dicts with commit metadata.
    """
    repo_id = get_repo_id(repo_path)
    db_path = get_db_path(repo_id)

    try:
        db = lancedb.connect(str(db_path))
        if TABLE_NAME not in db.table_names():
            logger.warning("No index found for this repository. Run `git pry index` first.")
            return []

        table = db.open_table(TABLE_NAME)
        query = (
            table.search(query_vector)
            .limit(top_k if not branch_filter else top_k * 3)  # Fetch more when filtering
            .select(["chunk_id", "commit_hash_short", "author", "date", "message",
                     "chunk_text", "chunk_type", "branch", "_distance"])
        )

        results = query.to_list()

        # Post-filter by branch if requested
        # TODO(V0.4 - Native Branch Filter): LanceDB .where() support for string columns is
        # version-dependent. Until stable, we over-fetch (top_k * 3) and post-filter in Python.
        # Limitation: if fewer than top_k chunks exist for the target branch, we silently
        # return less than top_k. No warning is shown to the user.
        if branch_filter:
            results = [r for r in results if r.get("branch") == branch_filter]
            results = results[:top_k]

        return results
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []
