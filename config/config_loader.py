"""
config_loader.py
-----------------
Single source of truth for loading config.yaml + .env across the whole
project. Every module (data_generator, preprocessing, spark, ml, hotspot,
alerts, database, dashboard) imports `get_config()` / `get_project_root()`
from here so paths and parameters stay consistent.
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

_CONFIG_CACHE = None


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def load_env():
    """Loads variables from .env file at project root (if present)."""
    env_path = get_project_root() / ".env"
    load_dotenv(dotenv_path=env_path)


def get_config() -> dict:
    """Loads config/config.yaml once and caches it."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        config_path = get_project_root() / "config" / "config.yaml"
        with open(config_path, "r") as f:
            _CONFIG_CACHE = yaml.safe_load(f)
        load_env()
    return _CONFIG_CACHE


def get_path(key: str) -> Path:
    """
    Resolves a path defined under config.yaml -> paths.<key>
    to an absolute Path object rooted at the project root.
    """
    cfg = get_config()
    rel_path = cfg["paths"][key]
    return get_project_root() / rel_path


def ensure_dirs():
    """Creates all directories referenced in config.yaml paths if missing."""
    cfg = get_config()
    root = get_project_root()
    for key in ["raw_dir", "synthetic_dir", "processed_dir", "models_dir"]:
        d = root / cfg["paths"][key]
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    cfg = get_config()
    print("Project root:", get_project_root())
    print("Loaded config sections:", list(cfg.keys()))
    ensure_dirs()
    print("Ensured data directories exist.")
