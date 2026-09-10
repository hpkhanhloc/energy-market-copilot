from datetime import date, datetime
from pathlib import Path

import pytest
from pydantic_ai import models
from pydantic_ai.models.test import TestModel

from copilot.intent import (
    FALLBACK_TEXT,
    Ask,
    Context,
    Investigate,
    Reply,
    Scan,
    build_intent_agent,
    data_reach,
    guard_intent,
    parse_intent,
    render_context,
)
from copilot.trace import last_call

models.ALLOW_MODEL_REQUESTS = False

TODAY = date(2026, 9, 10)


@pytest.fixture
def ctx() -> Context:
    return Context(today=TODAY, reach_start=date(2023, 11, 1), reach_end=date(2026, 9, 10))


def parse(seed: int, args: dict, ctx: Context):
    agent = build_intent_agent("test")
    with agent.override(model=TestModel(custom_output_args=args, seed=seed)):
        return parse_intent("anything", ctx, model="test", agent=agent)


def test_data_reach_from_month_files(tmp_path: Path) -> None:
    for m in ("202311", "202401", "202402"):
        (tmp_path / f"entsoe_price_FI_{m}.parquet").write_bytes(b"")
    (tmp_path / "entsoe_load_FI_202512.parquet").write_bytes(b"")  # other series ignored
    assert data_reach(tmp_path, TODAY) == (date(2023, 11, 1), date(2024, 2, 29))


def test_data_reach_caps_at_today_and_handles_empty(tmp_path: Path) -> None:
    assert data_reach(tmp_path, TODAY) == (TODAY, TODAY)
    (tmp_path / "entsoe_price_FI_202609.parquet").write_bytes(b"")
    assert data_reach(tmp_path, TODAY)[1] == TODAY


def test_render_context_lists_anchors(ctx: Context) -> None:
    full = Context(
        today=ctx.today,
        reach_start=ctx.reach_start,
        reach_end=ctx.reach_end,
        last_hour=datetime(2024, 1, 5, 19, 0),
        last_range=(date(2023, 12, 8), date(2024, 1, 8)),
        transcript=(("user", "hi"), ("assistant", "hello")),
    )
    text = render_context(full)
    assert "today: 2026-09-10" in text
    assert "data available: 2023-11-01 to 2026-09-10" in text
    assert "last investigated hour: 2024-01-05 19:00" in text
    assert "last scanned range: 2023-12-08 to 2024-01-08" in text
    assert "  user: hi" in text
    assert "last investigated" not in render_context(ctx)


def test_parse_intent_each_union_member(ctx: Context) -> None:
    scan = parse(0, {"start": "2023-12-08", "end": "2024-01-08"}, ctx)
    assert scan == Scan(start=date(2023, 12, 8), end=date(2024, 1, 8))

    inv = parse(1, {"when": "2024-01-05T19:00"}, ctx)
    assert isinstance(inv, Investigate)
    assert inv.when.tzinfo is not None
    assert inv.when.isoformat() == "2024-01-05T19:00:00+02:00"

    assert parse(2, {"question": "what is residual load"}, ctx) == Ask(
        question="what is residual load"
    )
    assert parse(3, {"text": "I only do power prices"}, ctx) == Reply(text="I only do power prices")


def test_parse_intent_records_trace(ctx: Context) -> None:
    parse(0, {"start": "2023-12-08", "end": "2024-01-08"}, ctx)
    call = last_call()
    assert call is not None
    assert call.kind == "intent"
    assert call.ok is True
    assert "USER: anything" in call.input


def test_parse_intent_falls_back_to_reply_on_error(ctx: Context, monkeypatch) -> None:
    agent = build_intent_agent("test")

    def boom(*_a, **_k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(agent, "run_sync", boom)
    out = parse_intent("hi", ctx, model="test", agent=agent)
    assert out == Reply(text=FALLBACK_TEXT)
    call = last_call()
    assert call is not None
    assert call.guard == "exception"
    assert call.fallback is True


def test_guard_scan_rules(ctx: Context) -> None:
    ok = Scan(start=date(2023, 12, 8), end=date(2024, 1, 8))
    assert guard_intent(ok, ctx) == ok
    swapped = Scan(start=date(2024, 1, 8), end=date(2023, 12, 8))
    assert guard_intent(swapped, ctx) == ok
    too_long = guard_intent(Scan(start=date(2024, 1, 1), end=date(2024, 3, 15)), ctx)
    assert isinstance(too_long, Reply)
    assert "60 days" in too_long.text
    early = guard_intent(Scan(start=date(2023, 11, 5), end=date(2023, 11, 20)), ctx)
    assert isinstance(early, Reply)
    assert "01 Dec 2023" in early.text  # reach_start + 30 baseline days


def test_guard_investigate_rules(ctx: Context) -> None:
    ok = Investigate(when=datetime(2024, 1, 5, 19))
    assert guard_intent(ok, ctx) is ok
    future = guard_intent(Investigate(when=datetime(2027, 1, 1, 12)), ctx)
    assert isinstance(future, Reply)
    assert "future" in future.text
    early = guard_intent(Investigate(when=datetime(2023, 11, 10, 19)), ctx)
    assert isinstance(early, Reply)
    assert "01 Dec 2023" in early.text


def test_guard_passes_ask_and_reply(ctx: Context) -> None:
    ask = Ask(question="q")
    reply = Reply(text="r")
    assert guard_intent(ask, ctx) is ask
    assert guard_intent(reply, ctx) is reply
