"""Settings loaded once from the environment (.env), no globals elsewhere."""

import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

HELSINKI = ZoneInfo("Europe/Helsinki")
UTC = ZoneInfo("UTC")
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "anthropic:claude-sonnet-5"


@dataclass(frozen=True, slots=True, kw_only=True)
class Settings:
    fingrid_api_key: str | None
    entsoe_api_key: str | None
    copilot_model: str
    cache_dir: Path


def load_settings(env_file: Path | None = None) -> Settings:
    """Read keys and paths from `.env` (if present) and the process environment."""
    load_dotenv(env_file or ROOT / ".env", override=False)
    cache_dir = Path(os.environ.get("COPILOT_CACHE_DIR", ROOT / "data" / "cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        fingrid_api_key=os.environ.get("FINGRID_API_KEY") or None,
        entsoe_api_key=os.environ.get("ENTSOE_API_KEY") or None,
        copilot_model=os.environ.get("COPILOT_MODEL") or DEFAULT_MODEL,
        cache_dir=cache_dir,
    )
