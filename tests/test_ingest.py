from pathlib import Path

import pytest

from memini_ai.config import Settings
from memini_ai.embed import HashEmbedder
from memini_ai.ingest import (
    Turn,
    chunk_turns,
    ingest_claude_sessions,
    parse_claude_transcript,
    project_from_slug,
)
from memini_ai.store import Store

FIX = Path(__file__).parent / "fixtures" / "claude_session.jsonl"


def test_parse_keeps_only_human_readable_turns():
    turns = parse_claude_transcript(FIX)
    assert [t.role for t in turns] == ["user", "assistant", "user", "assistant"]
    assert "secret reasoning" not in " ".join(t.text for t in turns)
    assert "file contents here" not in " ".join(t.text for t in turns)
    assert "generated while running a local command" not in " ".join(t.text for t in turns)
    assert turns[0].ts == "2026-09-16T01:28:54.888Z"


def test_chunk_turns_splits_at_message_boundaries_with_overlap():
    turns = [Turn("user", "a" * 600, None), Turn("assistant", "b" * 600, None), Turn("user", "c" * 600, None)]
    chunks = chunk_turns(turns, max_chars=1500)
    assert len(chunks) == 2
    assert chunks[0].startswith("user: aaa") and "assistant: bbb" in chunks[0]
    assert chunks[1].startswith("assistant: bbb") and "user: ccc" in chunks[1]


def test_chunk_turns_hard_splits_one_giant_turn():
    chunks = chunk_turns([Turn("user", "x" * 4000, None)], max_chars=1500)
    assert len(chunks) == 3 and all(len(c) <= 1500 + len("user: ") for c in chunks)


def test_project_from_slug():
    assert project_from_slug("-home-jcharles-Projects-food-index") == "food-index"
    assert project_from_slug("-home-jcharles-Projects-MCP-Servers") == "MCP-Servers"


@pytest.fixture
def store(db):
    return Store(db, HashEmbedder(), Settings(model="hash"))


async def test_ingest_is_idempotent_and_searchable(store, db, tmp_path):
    root = tmp_path / "projects" / "-home-jcharles-Projects-food-index"
    root.mkdir(parents=True)
    (root / "s1.jsonl").write_text(FIX.read_text())
    first = await ingest_claude_sessions(store, root=tmp_path / "projects", project=None, since=None)
    assert first["sessions"] == 1 and first["chunks"] >= 1 and first["duplicates"] == 0
    second = await ingest_claude_sessions(store, root=tmp_path / "projects", project=None, since=None)
    assert second["chunks"] == 0 and second["duplicates"] == first["chunks"]
    r = await store.recall("why delta pairs", kind="session", project="food-index")
    assert r["results"] and "T-SCANDEP-001" in r["results"][0]["text"]
    row = await db.fetchrow("SELECT source FROM memories WHERE kind='session' LIMIT 1")
    src = row["source"] if isinstance(row["source"], dict) else __import__("json").loads(row["source"])
    assert src["client"] == "claude-code" and src["session_id"] == "s1"
    assert src["project_slug"] == "-home-jcharles-Projects-food-index"


async def test_ingest_since_skips_old_files(store, tmp_path):
    import os
    import time

    root = tmp_path / "projects" / "-x"
    root.mkdir(parents=True)
    f = root / "s1.jsonl"
    f.write_text(FIX.read_text())
    old = time.time() - 40 * 86400
    os.utime(f, (old, old))
    r = await ingest_claude_sessions(store, root=tmp_path / "projects", project="p", since="30d")
    assert r == {"sessions": 0, "chunks": 0, "duplicates": 0}


async def test_ingest_skips_unreadable_file_and_continues(store, tmp_path):
    import os
    import stat

    if os.geteuid() == 0:
        pytest.skip("running as root; chmod 0 does not block reads")

    root = tmp_path / "projects" / "-home-jcharles-Projects-p"
    root.mkdir(parents=True)
    bad = root / "a-bad.jsonl"
    bad.write_text(FIX.read_text())
    bad.chmod(0)
    good = root / "b-good.jsonl"
    good.write_text(FIX.read_text())
    try:
        r = await ingest_claude_sessions(store, root=tmp_path / "projects", project=None, since=None)
    finally:
        bad.chmod(stat.S_IRUSR | stat.S_IWUSR)
    assert r["sessions"] == 1 and r["chunks"] >= 1
    assert os.access(good, os.R_OK)
