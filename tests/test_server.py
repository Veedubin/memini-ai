import asyncio
import json

import pytest
import structlog
from fastmcp import Client

from memini_ai.config import Settings
from memini_ai.server import configure_logging, create_app


@pytest.fixture
async def app(test_dsn, db):
    settings = Settings(model="hash", db_url=test_dsn, project="proj", timeout_s=5)
    application = create_app(settings)
    yield application
    await application.memini_state.close()  # type: ignore[attr-defined]


def _schema(tool):
    """FastMCP 4 renamed Tool.inputSchema to input_schema; accept either."""
    schema = getattr(tool, "input_schema", None)
    return schema if schema is not None else tool.inputSchema


def _data(result):
    if getattr(result, "data", None) is not None:
        return result.data
    return json.loads(result.content[0].text)


async def test_tools_are_exactly_three_and_small(app):
    async with Client(app) as c:
        tools = await c.list_tools()
    names = sorted(t.name for t in tools)
    assert names == ["orient", "recall", "remember"]
    schema = [{"name": t.name, "description": t.description, "inputSchema": _schema(t)} for t in tools]
    assert len(json.dumps(schema)) // 4 < 700
    remember = _schema(next(t for t in tools if t.name == "remember"))
    assert "description" in remember["properties"]["kind"]
    assert '"decision"' in json.dumps(remember["properties"]["kind"])


async def test_round_trip(app):
    async with Client(app) as c:
        r = _data(await c.call_tool("remember", {"text": "We use port 5555", "kind": "fact"}))
        assert r["duplicate"] is False and "error" not in r
        q = _data(await c.call_tool("recall", {"query": "which port"}))
        assert q["results"][0]["id"] == r["id"]
        o = _data(await c.call_tool("orient", {}))
        assert o["status"]["db"] == "ok" and o["status"]["memories"] == 1


async def test_errors_have_only_error_key(app):
    async with Client(app) as c:
        r = _data(await c.call_tool("remember", {"text": "   "}))
        assert set(r) == {"error"} and "empty" in r["error"]
        r = _data(await c.call_tool("recall", {"query": ""}))
        assert set(r) == {"error"}


async def test_thought_chain_over_mcp(app):
    async with Client(app) as c:
        t1 = _data(await c.call_tool("remember", {
            "text": "first", "kind": "thought",
            "chain": {"number": 1, "total": 2, "next_needed": True}}))
        t2 = _data(await c.call_tool("remember", {
            "text": "second", "kind": "thought",
            "chain": {"chain_id": t1["chain_id"], "number": 2, "total": 2, "next_needed": False}}))
        assert t2["chain_id"] == t1["chain_id"]
        r = _data(await c.call_tool("recall", {"query": "", "chain_id": t1["chain_id"]}))
        assert [x["chain"]["number"] for x in r["results"]] == [1, 2]


async def test_db_down_is_reported_not_raised(test_dsn, missing_db_dsn, db):
    settings = Settings(model="hash", db_url=missing_db_dsn, timeout_s=5)
    app = create_app(settings)
    try:
        async with Client(app) as c:
            o = _data(await c.call_tool("orient", {}))
            assert o["status"]["db"] == "error"
            r = _data(await c.call_tool("recall", {"query": "x"}))
            assert set(r) == {"error"}

            # The failed attempt above must not be cached: pointing the same app's state at
            # a working DSN and retrying should succeed, proving get_store() retries rather
            # than remembering the earlier failure.
            app.memini_state.settings = Settings(model="hash", db_url=test_dsn, timeout_s=5)  # type: ignore[attr-defined]
            o2 = _data(await c.call_tool("orient", {}))
            assert o2["status"]["db"] == "ok"
    finally:
        await app.memini_state.close()  # type: ignore[attr-defined]


async def test_unexpected_exception_logs_traceback_and_reports_error(app, capsys):
    configure_logging("INFO")
    try:
        state = app.memini_state  # type: ignore[attr-defined]
        store = await state.get_store()

        async def boom(*args, **kwargs):
            raise RuntimeError("boom")

        store.recall = boom  # type: ignore[method-assign]

        async with Client(app) as c:
            r = _data(await c.call_tool("recall", {"query": "x"}))
        assert set(r) == {"error"}

        captured = capsys.readouterr()
        assert "Traceback" in captured.err
    finally:
        # configure_logging binds structlog's PrintLoggerFactory to the current sys.stderr,
        # which under capsys is a capture buffer that gets closed when the test ends. Reset
        # to defaults so later tests' log calls don't write to that closed file.
        structlog.reset_defaults()


async def test_configure_logging_falls_back_on_an_invalid_level(capsys):
    import logging

    try:
        configure_logging("chatty")
        assert logging.getLogger().level == logging.INFO
        assert "chatty" in capsys.readouterr().err
    finally:
        structlog.reset_defaults()


async def test_timeout_returns_only_an_error(test_dsn, db):
    settings = Settings(model="hash", db_url=test_dsn, timeout_s=0.01)
    app = create_app(settings)
    try:
        store = await app.memini_state.get_store()  # type: ignore[attr-defined]

        async def slow(*args, **kwargs):
            await asyncio.sleep(1)
            return {"results": []}

        store.recall = slow  # type: ignore[method-assign]
        async with Client(app) as c:
            r = _data(await c.call_tool("recall", {"query": "x"}))
        assert r == {"error": "timeout after 0.01s"}
    finally:
        await app.memini_state.close()  # type: ignore[attr-defined]
