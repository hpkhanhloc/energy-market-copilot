import json
from dataclasses import replace
from pathlib import Path

from copilot.trace import LlmCall, last_call, record, timed, trace_path

BASE = LlmCall(
    kind="narrative",
    model="test",
    input="FACTS",
    output="{}",
    ok=True,
    guard=None,
    fallback=False,
    latency_ms=12,
    ts="2026-09-10T00:00:00.000+00:00",
)


def test_trace_path_from_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("COPILOT_TRACE", str(tmp_path / "x.jsonl"))
    assert trace_path() == tmp_path / "x.jsonl"
    monkeypatch.delenv("COPILOT_TRACE")
    assert trace_path().name == "llm.jsonl"


def test_record_appends_and_last_call_reads_back(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "llm.jsonl"
    assert last_call(path) is None
    record(BASE, path)
    record(replace(BASE, kind="intent", guard="exception", ok=False, fallback=True), path)
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["kind"] == "narrative"
    last = last_call(path)
    assert last is not None
    assert last.kind == "intent"
    assert last.guard == "exception"
    assert last.fallback is True


def test_record_never_raises(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    record(BASE, blocker / "cannot" / "nest.jsonl")  # parent is a file


def test_last_call_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "llm.jsonl"
    record(BASE, path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n   \n")  # trailing blank lines, as a partial/flushed write might leave
    last = last_call(path)
    assert last is not None
    assert last.kind == "narrative"


def test_last_call_returns_none_when_file_is_all_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "llm.jsonl"
    path.write_text("\n\n   \n")
    assert last_call(path) is None


def test_timed_returns_result_or_exception() -> None:
    out, ms = timed(lambda: 3)
    assert out == 3
    assert isinstance(ms, int)

    def boom() -> int:
        raise RuntimeError("down")

    err, _ = timed(boom)
    assert isinstance(err, RuntimeError)


def test_last_call_skips_a_truncated_or_foreign_line(tmp_path: Path) -> None:
    """The sidebar reads this on every rerun; one bad line must not take the app down."""
    path = tmp_path / "llm.jsonl"
    record(BASE, path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"kind": "intent", "model": "m", "inp\n')  # killed mid-write
        fh.write('{"kind": "intent", "model": "m"}\n')  # older schema, fields missing
    last = last_call(path)
    assert last is not None
    assert last.kind == "narrative"
    path.write_text("not json at all\n")
    assert last_call(path) is None


def test_record_call_derives_ok_and_fallback(tmp_path: Path, monkeypatch) -> None:
    from copilot.trace import record_call

    monkeypatch.setenv("COPILOT_TRACE", str(tmp_path / "t.jsonl"))
    record_call("intent", "m", "p", "o", 5, guard=None)
    call = last_call()
    assert call is not None
    assert (call.ok, call.fallback) == (True, False)
    record_call("answer", "m", "p", "o", 5, guard="banned_phrase")
    call = last_call()
    assert call is not None
    assert (call.ok, call.fallback) == (False, True)
    record_call("narrative", "m", "p", "o", 5, guard="invented_hypotheses", fallback=False)
    call = last_call()
    assert call is not None
    assert (call.ok, call.guard, call.fallback) == (False, "invented_hypotheses", False)
