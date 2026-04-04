<h1 align="center">GitPry</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Experimental-yellow.svg" alt="Status" />
  <img src="https://img.shields.io/badge/Version-0.7.1-blue.svg" alt="Version" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-blueviolet.svg" alt="Python" />
</p>

**Talk to Git History with Natural Language.** GitPry is a local-first Python CLI and MCP server that transforms your raw commit history into an interactive, AI-powered knowledge base.

## Prerequisites

GitPry requires [Ollama](https://ollama.ai/) running locally to power its semantic search and codebase analysis. Install Ollama and pull the required models:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:7b  # Or your preferred chat model
```

## Installation

```bash
pipx install git+https://github.com/hoodies-gent/GitPry.git
```

> Requires **Python 3.10+**. `pipx` handles environment isolation automatically.

## Usage

**First, navigate to ANY local Git repository you want to analyze:**
```bash
cd /path/to/your/project
```

### 1. CLI: Talk to your repo
```bash
# Semantic "Why/How" queries (better w/ RAG)
gitpry ask "Why was the auth module rewritten?"

# Structured metadata queries
gitpry ask "Show me the last 5 commits by John"
gitpry ask "What changed yesterday?"

# Aggregate & analytical queries
gitpry ask "How many commits do we have in total?"
```
*(Tips: Run `gitpry index` first to enable semantic search; use `--no-rag` to bypass it at any time. For more options, run `gitpry --help` or `gitpry <command> --help`.)*

### 2. MCP Server: AI IDE Integration
Start the standard I/O server to grant agents (Cursor, Claude Desktop, etc.) access to git history tools:
```bash
gitpry serve
```
*Exposes: `semantic_search`, `git_log_scan`, `get_commit_diff`, `get_file_blame`, `compare_branches`, `get_repo_stats`.*

## Configuration

GitPry auto-generates `~/.config/gitpry/config.toml` on first run. Edit it to customize the model, timeout, RAG settings, and more.

For all available options, see [`.gitpry.example.toml`](.gitpry.example.toml).

## Uninstallation

```bash
pipx uninstall gitpry
rm -rf ~/.gitpry  # optional: clear vector database
```

> ### Known Limitations
> 
> As an experimental proof-of-concept, GitPry currently operates within the following boundaries:
> 
> - **Oversized Commits:** Massive file changes (like lockfiles) are automatically truncated to prevent local LLM context overflow.
> - **Merge Commits:** These are excluded from RAG indexing to keep semantic search results highly accurate and noise-free.
> - **Hardware Demand:** Cross-file chronological analysis requires you to select an LLM with a sufficiently large context window.