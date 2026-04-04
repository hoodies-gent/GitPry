"""
Configuration Management for GitPry.
Handles hierarchical loading of configurations from TOML files and environment variables.
"""
import os
import sys
from pathlib import Path
from dataclasses import dataclass

# Python 3.11+ comes with tomllib, earlier versions need tomli
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# TODO(Future - Cloud Fallback):
# Add support for cloud LLM providers (OpenAI, Anthropic) as a fallback when
# Ollama is not available. Controlled via `provider` field and corresponding API key env vars.
@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "qwen2.5-coder:7b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    timeout: float = 60.0
    max_tokens: int = 6000


@dataclass
class GitConfig:
    limit: int = 500
    include_diff: bool = False
    max_diff_lines: int = 150

@dataclass
class RagConfig:
    enabled: bool = True
    embed_model: str = "nomic-embed-text"
    max_chunk_tokens: int = 400
    top_k: int = 5

@dataclass
class GitPryConfig:
    llm: LLMConfig
    git: GitConfig
    rag: RagConfig

def _load_toml_dict(filepath: Path) -> dict:
    if not filepath.exists() or not filepath.is_file():
        return {}
    try:
        with open(filepath, "rb") as f:
            return tomllib.load(f)
    except Exception:
        # If the file is malformed, we just ignore it gracefully and use defaults
        return {}

_GLOBAL_CONFIG_TEMPLATE = """\
# GitPry Global Configuration
# This file was auto-generated on first run. Edit as needed.
# For all available options, see: https://github.com/hoodies-gent/GitPry/blob/main/.gitpry.example.toml

[llm]
# The Ollama model to use for answering questions.
# Run 'ollama list' to see what models you have available.
model = "qwen2.5-coder:7b"

# Ollama server address (default: local)
# base_url = "http://localhost:11434"

[git]
# Max commits to scan in legacy (non-RAG) mode
# limit = 500

[rag]
# Embedding model for semantic indexing (must be pulled via ollama)
# embed_model = "nomic-embed-text"
"""

def _ensure_global_config() -> None:
    """Write a default config to ~/.config/gitpry/config.toml on first run."""
    global_path = Path.home() / ".config" / "gitpry" / "config.toml"
    if not global_path.exists():
        try:
            global_path.parent.mkdir(parents=True, exist_ok=True)
            global_path.write_text(_GLOBAL_CONFIG_TEMPLATE, encoding="utf-8")
            print(f"💡 First run detected! Created default config at: {global_path}")
            print("   Edit it to change the Ollama model or other settings.\n")
        except Exception:
            pass  # Fail silently; defaults will still apply

def load_config() -> GitPryConfig:
    """
    Loads configuration with the following priority (highest to lowest):
    1. Environment Variables (e.g., GITPRY_LLM_MODEL)
    2. Local Project Config (./.gitpry.toml)
    3. Global User Config (~/.config/gitpry/config.toml)
    4. Code Defaults
    """
    # 0. Bootstrap global config on first run
    _ensure_global_config()

    # 1. Start with hardcoded defaults
    config_dict = {
        "llm": {
            "provider": "ollama", 
            "model": "qwen2.5-coder:7b", 
            "base_url": "http://localhost:11434",
            "temperature": 0.1,
            "timeout": 60.0,
            "max_tokens": 6000
        },
        "git": {
            "limit": 500,
            "include_diff": False,
            "max_diff_lines": 150
        },
        "rag": {
            "enabled": True,
            "embed_model": "nomic-embed-text",
            "max_chunk_tokens": 400,
            "top_k": 5,
        }
    }

    # 2. Merge Global Config (lowest file priority)
    global_path = Path.home() / ".config" / "gitpry" / "config.toml"
    global_dict = _load_toml_dict(global_path)
    for section, values in global_dict.items():
        if section in config_dict and isinstance(values, dict):
            config_dict[section].update(values)

    # 3. Merge Local Config (highest file priority)
    local_path = Path.cwd() / ".gitpry.toml"
    local_dict = _load_toml_dict(local_path)
    for section, values in local_dict.items():
        if section in config_dict and isinstance(values, dict):
            config_dict[section].update(values)

    # 4. Merge Environment Variables (absolute highest priority)
    # E.g., export GITPRY_LLM_MODEL="llama3.1"
    env_model = os.getenv("GITPRY_LLM_MODEL")
    if env_model:
        config_dict["llm"]["model"] = env_model
        
    env_base_url = os.getenv("GITPRY_LLM_BASE_URL")
    if env_base_url:
        config_dict["llm"]["base_url"] = env_base_url

    # Build typed dataclasses
    return GitPryConfig(
        llm=LLMConfig(**config_dict["llm"]),
        git=GitConfig(**config_dict["git"]),
        rag=RagConfig(**config_dict["rag"]),
    )

# Singleton global instance
# This gets evaluated once upon module import, reading the disk/envs at CLI startup.
settings = load_config()
