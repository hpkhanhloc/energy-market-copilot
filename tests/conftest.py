import pytest


@pytest.fixture(autouse=True)
def _trace_to_tmp(tmp_path, monkeypatch) -> None:
    """Every LLM call is logged; keep test logs out of data/logs/."""
    monkeypatch.setenv("COPILOT_TRACE", str(tmp_path / "llm.jsonl"))
