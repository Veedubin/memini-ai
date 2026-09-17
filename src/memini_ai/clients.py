"""Client adapters: merge-only config edits, skill install, and the memory protocol block."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

SERVER_NAME = "memini-ai"
START, END = "<!-- memini-ai:start -->", "<!-- memini-ai:end -->"

PROTOCOL_BLOCK = """## Memory (memini-ai)
- Start every session with `orient`. Read it before planning.
- Before re-deriving anything about this project, `recall` it first.
- After a decision, a finished task, or a handoff, call `remember` with the matching `kind`.
  One paragraph, stating what and why. Include file paths and commit ids when relevant.
- When a fact changes, `remember` the new one with `supersedes=<old id>`.
- For multi-step reasoning you want to survive the session, use `kind="thought"` with `chain`.
- Never store secrets, tokens, or full file contents.
"""


@dataclass
class InitReport:
    config_path: Path | None
    backup_path: Path | None
    skill_path: Path | None
    instructions_path: Path
    protocol_added: bool
    snippet: str | None


def merge_json(path: Path, key_path: list[str], value: dict[str, Any]) -> Path | None:
    """Set data[key_path...] = value, keeping every other key. Returns the backup path if one was made."""
    data: dict[str, Any] = {}
    existed = path.exists()
    if existed:
        raw = path.read_text(encoding="utf-8")
        try:
            data = json.loads(raw or "{}")
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: not valid JSON ({e}); fix or remove it and re-run") from e
    node = data
    for key in key_path[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"{path}: {key!r} is not an object")
    backup: Path | None = None
    if existed:
        backup = path.with_name(f"{path.name}.bak-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}")
        shutil.copy2(path, backup)
    node[key_path[-1]] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return backup


def ensure_protocol(path: Path) -> bool:
    """Append the protocol block once. Returns True if the file changed."""
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if START in existing:
        return False
    block = f"\n{START}\n{PROTOCOL_BLOCK}{END}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(existing.rstrip("\n") + "\n" + block if existing else block.lstrip("\n"), encoding="utf-8")
    return True


def server_entry(client: str, command: list[str], env: dict[str, str]) -> dict[str, Any]:
    exe, args = command[0], command[1:]
    if client == "claude-code":
        return {"type": "stdio", "command": exe, "args": args, "env": env}
    if client == "opencode":
        return {"type": "local", "command": command, "environment": env, "enabled": True}
    if client == "kimi-code":
        return {"command": exe, "args": args, "env": env, "enabled": True}
    return {"command": exe, "args": args, "env": env}


def install_skill(dest_dir: Path) -> Path:
    src = resources.files("memini_ai") / "skill" / "init-memini" / "SKILL.md"
    out = dest_dir / "init-memini" / "SKILL.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return out


def init_client(
    client: str,
    scope: str,
    project: str | None,
    command: str | None,
    cwd: Path,
    home: Path,
) -> InitReport:
    cmd = shlex.split(command) if command else ["memini-ai", "serve"]
    env: dict[str, str] = {"MEMINI_PROJECT": project or cwd.name}
    entry = server_entry(client, cmd, env)

    config_path: Path | None
    key_path: list[str]
    skill_dir: Path | None
    instructions = cwd / "AGENTS.md"
    if client == "claude-code":
        config_path = (home / ".claude.json") if scope == "user" else (cwd / ".mcp.json")
        key_path = ["mcpServers", SERVER_NAME]
        skill_dir = home / ".claude" / "skills"
        instructions = cwd / "CLAUDE.md"
    elif client == "opencode":
        config_path = (
            (home / ".config" / "opencode" / "opencode.json")
            if scope == "user"
            else (cwd / ".opencode" / "opencode.json")
        )
        key_path = ["mcp", SERVER_NAME]
        skill_dir = (home / ".config" / "opencode" / "skills") if scope == "user" else (cwd / ".opencode" / "skills")
    elif client == "kimi-code":
        config_path = cwd / ".kimi-code" / "mcp.json"
        key_path = ["mcpServers", SERVER_NAME]
        skill_dir = None
    elif client == "generic":
        config_path, key_path, skill_dir = None, [], None
    else:
        raise ValueError(f"unknown client {client!r}")

    backup = merge_json(config_path, key_path, entry) if config_path is not None else None
    skill_path = install_skill(skill_dir) if skill_dir is not None else None
    added = ensure_protocol(instructions)
    snippet = json.dumps({"mcpServers": {SERVER_NAME: entry}}, indent=2) if client == "generic" else None
    return InitReport(config_path, backup, skill_path, instructions, added, snippet)


def run_init(args: argparse.Namespace) -> int:
    try:
        r = init_client(args.client, args.scope, args.project, args.command, args.cwd.resolve(), Path.home())
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    if r.config_path:
        print(f"wrote {r.config_path}" + (f" (backup {r.backup_path})" if r.backup_path else ""))
    if r.skill_path:
        print(f"installed skill {r.skill_path}")
    print(f"{'added' if r.protocol_added else 'kept'} memory protocol in {r.instructions_path}")
    if r.snippet:
        print("add this to your client's MCP config:\n" + r.snippet)
    print("restart the client so it reloads MCP servers, then call orient.")
    return 0
