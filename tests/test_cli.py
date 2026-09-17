import json
import subprocess
import sys

import pytest

from memini_ai.cli import build_parser


def test_parser_has_subcommands():
    p = build_parser()
    for argv in (["serve"], ["migrate"], ["db", "up"], ["db", "status"], ["import", "x.jsonl"], ["reembed"]):
        assert p.parse_args(argv).cmd == argv[0]


def test_serve_writes_nothing_to_stdout_before_transport(test_dsn):
    # Start serve with stdin closed: the MCP transport reads EOF and exits. stdout must stay empty.
    env = {"MEMINI_DB_URL": test_dsn, "MEMINI_MODEL": "hash", "PATH": "/usr/bin:/bin",
           "HOME": "/tmp", "MEMINI_CONFIG_FILE": "/nonexistent"}
    proc = subprocess.run([sys.executable, "-m", "memini_ai.cli", "serve"], input=b"", capture_output=True,
                          env=env, timeout=60)
    assert proc.stdout == b"", proc.stdout[:200]


async def test_import_jsonl(db, tmp_path):
    from memini_ai.config import Settings
    from memini_ai.embed import HashEmbedder
    from memini_ai.ingest import import_jsonl
    from memini_ai.store import Store

    p = tmp_path / "in.jsonl"
    p.write_text("\n".join([
        json.dumps({"text": "alpha decision", "kind": "decision", "tags": ["a"]}),
        json.dumps({"text": "alpha decision", "kind": "decision"}),
        json.dumps({"text": "", "kind": "note"}),
        json.dumps({"kind": "note"}),
        "not json",
        json.dumps({"text": "kitchen", "project": "kitchen"}),
    ]))
    store = Store(db, HashEmbedder(), Settings(model="hash", project="proj"))
    r = await import_jsonl(store, p, project=None)
    assert r == {"imported": 2, "duplicates": 1, "skipped": 3}
    rows = await db.fetch("SELECT project FROM memories ORDER BY created_at")
    assert [x["project"] for x in rows] == ["proj", "kitchen"]


@pytest.mark.parametrize("argv", [["db", "bogus"], ["nope"]])
def test_bad_args_exit_2(argv):
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(argv)
    assert e.value.code == 2


def test_parser_has_warm():
    assert build_parser().parse_args(["warm"]).cmd == "warm"


def test_db_commands_name_the_compose_project(monkeypatch):
    """Without -p the project name comes from the packaged compose.yaml's directory."""
    from memini_ai import cli

    seen = []
    monkeypatch.setattr(cli, "_compose_cmd", lambda: ["podman", "compose"])
    monkeypatch.setattr(cli.subprocess, "call", lambda argv: seen.append(argv) or 0)
    for action in ("up", "down", "status"):
        assert cli.cmd_db(action) == 0
    assert [a[:4] for a in seen] == [["podman", "compose", "-p", "memini-ai"]] * 3
    assert [a[-1] for a in seen] == ["-d", "down", "ps"]


def test_warm_loads_the_embedder_and_reports_it(monkeypatch, capsys):
    from memini_ai.cli import main

    monkeypatch.setenv("MEMINI_MODEL", "hash")
    monkeypatch.setenv("MEMINI_CONFIG_FILE", "/nonexistent")
    assert main(["warm"]) == 0
    out = capsys.readouterr().out
    assert "hash" in out and "1024" in out
