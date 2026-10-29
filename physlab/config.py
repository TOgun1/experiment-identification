"""Settings loaded from the environment (and `.env` at the repo root)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]

# Values from the Meta Model API cookbook (github.com/meta-models/meta-model-cookbook).
DEFAULT_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_SAM_MODEL = "sam-3.1"
DEFAULT_MUSE_MODEL = "muse-spark-1.3"


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str = DEFAULT_BASE_URL
    sam_model: str = DEFAULT_SAM_MODEL
    muse_model: str = DEFAULT_MUSE_MODEL
    cache_dir: Path = REPO_ROOT / "data" / "cache"
    # When true, a cache miss raises instead of calling the API (protects credits).
    offline: bool = False


def load_settings() -> Settings:
    load_dotenv(REPO_ROOT / ".env")
    cache_dir = Path(os.environ.get("PHYSLAB_CACHE_DIR", REPO_ROOT / "data" / "cache"))
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    return Settings(
        api_key=os.environ.get("MODEL_API_KEY") or None,
        base_url=os.environ.get("META_API_BASE_URL", DEFAULT_BASE_URL),
        sam_model=os.environ.get("SAM_MODEL", DEFAULT_SAM_MODEL),
        muse_model=os.environ.get("MUSE_MODEL", DEFAULT_MUSE_MODEL),
        cache_dir=cache_dir,
        offline=os.environ.get("PHYSLAB_OFFLINE", "0").lower() in ("1", "true", "yes"),
    )
