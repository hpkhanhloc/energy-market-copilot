from pathlib import Path

import pytest

from copilot.config import DEFAULT_MODEL, load_settings


def test_load_settings_reads_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("FINGRID_API_KEY", "ENTSOE_API_KEY", "COPILOT_MODEL", "COPILOT_CACHE_DIR"):
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / ".env"
    env.write_text("FINGRID_API_KEY=abc\nENTSOE_API_KEY=\n")
    monkeypatch.setenv("COPILOT_CACHE_DIR", str(tmp_path / "cache"))

    settings = load_settings(env)

    assert settings.fingrid_api_key == "abc"
    assert settings.entsoe_api_key is None  # empty string means missing
    assert settings.copilot_model == DEFAULT_MODEL
    assert settings.cache_dir == tmp_path / "cache"
    assert settings.cache_dir.is_dir()
