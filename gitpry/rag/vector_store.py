"""
gitpry/rag/vector_store.py

Manages the local LanceDB vector database for Git commit embeddings.

The database is stored at: ~/.gitpry/vectors/<repo_id>/
- repo_id is the first 8 chars of the SHA1 of the repo's remote URL (or its absolute path as fallback).
- This allows multiple different repos to have isolated vector stores.
"""
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
import lancedb
import pyarrow as pa
from gitpry.utils.logger import logger

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
    pa.field("vector", pa.list_(pa.float32())), # The embedding vector
    # TODO(V0.3 - P1 Branch Awareness): Add pa.field("branch", pa.list_(pa.string()))
    # to tag each chunk with which branches it belongs to, enabling cross-branch queries.
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


def open_or_create_table(db, vector_dim: int):
    """Open the commits table or create it with the correct schema."""
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
    
    # LanceDB infers the vector dimension from the first write.
    # We create it empty with our fixed schema.
    schema = pa.schema([
        pa.field("chunk_id", pa.string()),
        pa.field("commit_hash", pa.string()),
        pa.field("commit_hash_short", pa.string()),
        pa.field("author", pa.string()),
        pa.field("date", pa.string()),
        pa.field("message", pa.string()),
        pa.field("chunk_text", pa.string()),
        pa.field("chunk_type", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), vector_dim)),
    ])
    table = db.create_table(TABLE_NAME, schema=schema)
    # TODO(V0.3 - Performance): After inserting chunks, call table.create_index()
    # to build an ANN (Approximate Nearest Neighbor) index for faster vector search
    # on repos with > 5000 chunks.
    return table


def get_indexed_hashes(table) -> set:
    """Return the set of already-indexed full commit hashes for incremental indexing."""
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


def search_similar(repo_path: str, query_vector: List[float], top_k: int = 5) -> List[Dict]:
    """
    Search the vector store for the top-K most semantically similar commit chunks.
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
        results = (
            table.search(query_vector)
            .limit(top_k)
            .select(["chunk_id", "commit_hash_short", "author", "date", "message", "chunk_text", "chunk_type", "_distance"])
            .to_list()
        )
        return results
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []
