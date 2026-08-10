from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENV_LOADED = False


def repo_root() -> Path:
    return _REPO_ROOT


def load_project_env(*, override: bool = False) -> Path | None:
    """Load ``.env`` from the repo root. Shell env wins unless ``override=True``."""
    global _ENV_LOADED
    env_path = _REPO_ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=override)
    _ENV_LOADED = True
    return env_path if env_path.is_file() else None


def first_env(*names: str) -> str | None:
    if not _ENV_LOADED:
        load_project_env()
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value
    return None


load_project_env()
