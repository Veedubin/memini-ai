"""Settings. Env vars are MEMINI_* only; an explicit config file supplies defaults."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_FILE = Path("~/.config/memini-ai/config.env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEMINI_", extra="ignore")

    db_url: str = "postgresql://memini:memini@localhost:5555/memini"
    model: str = "BAAI/bge-m3"
    device: str = "cpu"
    project: str | None = None
    config_file: Path = DEFAULT_CONFIG_FILE
    log_level: str = "INFO"
    timeout_s: float = 30.0
    max_text_bytes: int = 32768


def _config_file_path() -> Path:
    raw = os.environ.get("MEMINI_CONFIG_FILE")
    return Path(raw).expanduser() if raw else DEFAULT_CONFIG_FILE.expanduser()


def load_settings(**overrides: Any) -> Settings:
    """Precedence: explicit overrides > process env > config file > defaults."""
    path = _config_file_path()
    env_file = str(path) if path.is_file() else None
    return Settings(_env_file=env_file, **overrides)  # type: ignore[call-arg]
