import json

from memini_ai.clients import (
    PROTOCOL_BLOCK,
    ensure_protocol,
    init_client,
    install_skill,
    merge_json,
    server_entry,
)


def test_merge_json_preserves_other_keys_and_backs_up(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}, "theme": "dark"}))
    backup = merge_json(p, ["mcpServers", "memini-ai"], {"command": "memini-ai"})
    data = json.loads(p.read_text())
    assert data["theme"] == "dark" and data["mcpServers"]["other"] == {"command": "x"}
    assert data["mcpServers"]["memini-ai"] == {"command": "memini-ai"}
    assert backup is not None and backup.exists() and "theme" in backup.read_text()


def test_merge_json_creates_missing_file(tmp_path):
    p = tmp_path / "sub" / "c.json"
    assert merge_json(p, ["mcp", "memini-ai"], {"a": 1}) is None
    assert json.loads(p.read_text()) == {"mcp": {"memini-ai": {"a": 1}}}


def test_ensure_protocol_is_idempotent(tmp_path):
    p = tmp_path / "CLAUDE.md"
    p.write_text("# My project\n")
    assert ensure_protocol(p) is True
    assert ensure_protocol(p) is False
    text = p.read_text()
    assert text.count("<!-- memini-ai:start -->") == 1 and PROTOCOL_BLOCK in text
    assert text.startswith("# My project\n")


def test_server_entry_shapes():
    cmd = ["memini-ai", "serve"]
    env = {"MEMINI_PROJECT": "p"}
    assert server_entry("claude-code", cmd, env) == {"type": "stdio", "command": "memini-ai", "args": ["serve"], "env": env}
    assert server_entry("opencode", cmd, env) == {"type": "local", "command": cmd, "environment": env, "enabled": True}
    assert server_entry("kimi-code", cmd, env) == {"command": "memini-ai", "args": ["serve"], "env": env, "enabled": True}


def test_install_skill_copies_file(tmp_path):
    out = install_skill(tmp_path / "skills")
    assert out == tmp_path / "skills" / "init-memini" / "SKILL.md"
    assert "name: init-memini" in out.read_text()


def test_init_claude_code_project_scope(tmp_path):
    cwd = tmp_path / "repo"
    cwd.mkdir()
    home = tmp_path / "home"
    r = init_client("claude-code", "project", "repo", None, cwd, home)
    assert r.config_path == cwd / ".mcp.json"
    cfg = json.loads(r.config_path.read_text())
    assert cfg["mcpServers"]["memini-ai"]["env"] == {"MEMINI_PROJECT": "repo"}
    assert r.skill_path == home / ".claude" / "skills" / "init-memini" / "SKILL.md" and r.skill_path.exists()
    assert r.instructions_path == cwd / "CLAUDE.md" and r.protocol_added is True
    assert r.snippet is None


def test_init_claude_code_user_scope_and_custom_command(tmp_path):
    cwd = tmp_path / "repo"
    cwd.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    (home / ".claude.json").write_text(json.dumps({"projects": {}}))
    r = init_client("claude-code", "user", "repo", "uv run --directory /x memini-ai serve", cwd, home)
    assert r.config_path == home / ".claude.json"
    cfg = json.loads(r.config_path.read_text())
    assert cfg["projects"] == {}
    assert cfg["mcpServers"]["memini-ai"]["command"] == "uv"
    assert cfg["mcpServers"]["memini-ai"]["args"] == ["run", "--directory", "/x", "memini-ai", "serve"]


def test_init_opencode_and_kimi(tmp_path):
    cwd = tmp_path / "repo"
    cwd.mkdir()
    home = tmp_path / "home"
    r = init_client("opencode", "project", "repo", None, cwd, home)
    assert r.config_path == cwd / ".opencode" / "opencode.json"
    assert json.loads(r.config_path.read_text())["mcp"]["memini-ai"]["type"] == "local"
    assert r.instructions_path == cwd / "AGENTS.md"
    assert r.skill_path == cwd / ".opencode" / "skills" / "init-memini" / "SKILL.md"
    r = init_client("kimi-code", "project", "repo", None, cwd, home)
    assert r.config_path == cwd / ".kimi-code" / "mcp.json"
    assert json.loads(r.config_path.read_text())["mcpServers"]["memini-ai"]["enabled"] is True
    assert r.skill_path is None


def test_init_generic_prints_snippet(tmp_path):
    cwd = tmp_path / "repo"
    cwd.mkdir()
    r = init_client("generic", "project", "repo", None, cwd, tmp_path / "home")
    assert r.config_path is None and r.snippet is not None
    assert json.loads(r.snippet)["mcpServers"]["memini-ai"]["args"] == ["serve"]
    assert (cwd / "AGENTS.md").exists() and r.protocol_added
