from pathlib import Path

from memini_ai.config import Settings, load_settings


def test_defaults(monkeypatch, tmp_path):
    for k in list(__import__("os").environ):
        if k.startswith("MEMINI_"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("MEMINI_CONFIG_FILE", str(tmp_path / "missing.env"))
    s = load_settings()
    assert s.db_url == "postgresql://memini:memini@localhost:5555/memini"
    assert s.model == "BAAI/bge-m3"
    assert s.device == "cpu"
    assert s.project is None
    assert s.timeout_s == 30.0
    assert s.max_text_bytes == 32768


def test_config_file_then_env_precedence(monkeypatch, tmp_path):
    cfg = tmp_path / "config.env"
    cfg.write_text("MEMINI_MODEL=hash\nMEMINI_PROJECT=from-file\n")
    monkeypatch.setenv("MEMINI_CONFIG_FILE", str(cfg))
    monkeypatch.setenv("MEMINI_PROJECT", "from-env")
    monkeypatch.delenv("MEMINI_MODEL", raising=False)
    s = load_settings()
    assert s.model == "hash"
    assert s.project == "from-env"


def test_overrides_win(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMINI_CONFIG_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setenv("MEMINI_DEVICE", "cuda")
    s = load_settings(device="cpu")
    assert s.device == "cpu"


def test_settings_type():
    assert Settings.model_fields["config_file"].annotation is Path
