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


def _data(result):
    if getattr(result, "data", None) is not None:
        return result.data
    return json.loads(result.content[0].text)


async def test_tools_are_exactly_three_and_small(app):
    async with Client(app) as c:
        tools = await c.list_tools()
    names = sorted(t.name for t in tools)
    assert names == ["orient", "recall", "remember"]
    schema = [{"name": t.name, "description": t.description, "inputSchema": t.inputSchema} for t in tools]
    assert len(json.dumps(schema)) // 4 < 700
    remember = next(t for t in tools if t.name == "remember")
    assert "description" in remember.inputSchema["properties"]["kind"]
    assert '"decision"' in json.dumps(remember.inputSchema["properties"]["kind"])


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


async def test_db_down_is_reported_not_raised(test_dsn, db):
    settings = Settings(model="hash", db_url="postgresql://memini:memini@localhost:5555/does_not_exist", timeout_s=5)
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
