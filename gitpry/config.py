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

@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "qwen2.5-coder:7b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    timeout: float = 60.0

@dataclass
class GitConfig:
    limit: int = 500
    include_diff: bool = True
    max_diff_lines: int = 150

@dataclass
class GitPryConfig:
    llm: LLMConfig
    git: GitConfig

def _load_toml_dict(filepath: Path) -> dict:
    if not filepath.exists() or not filepath.is_file():
        return {}
    try:
        with open(filepath, "rb") as f:
            return tomllib.load(f)
    except Exception:
        # If the file is malformed, we just ignore it gracefully and use defaults
        return {}

def load_config() -> GitPryConfig:
    """
    Loads configuration with the following priority (highest to lowest):
    1. Environment Variables (e.g., GITPRY_LLM_MODEL)
    2. Local Project Config (./.gitpry.toml)
    3. Global User Config (~/.config/gitpry/config.toml)
    4. Code Defaults
    """
    # 1. Start with hardcoded defaults
    config_dict = {
        "llm": {
            "provider": "ollama", 
            "model": "qwen2.5-coder:7b", 
            "base_url": "http://localhost:11434",
            "temperature": 0.1,
            "timeout": 60.0
        },
        "git": {
            "limit": 500,
            "include_diff": True,
            "max_diff_lines": 150
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
        git=GitConfig(**config_dict["git"])
    )

# Singleton global instance
# This gets evaluated once upon module import, reading the disk/envs at CLI startup.
settings = load_config()
