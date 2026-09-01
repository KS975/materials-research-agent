from pathlib import Path

from company_data import resolve_company_data_runtime_root


def test_default_company_runtime_is_project_runtime(monkeypatch):
    monkeypatch.delenv("COMPANY_DATA_RUNTIME_ROOT", raising=False)
    root = resolve_company_data_runtime_root()
    assert root.name == ".runtime"
    assert root.parent == Path(__file__).resolve().parents[2]


def test_environment_override_is_respected(monkeypatch, tmp_path):
    custom = tmp_path / "custom-real-data-runtime"
    monkeypatch.setenv("COMPANY_DATA_RUNTIME_ROOT", str(custom))
    assert resolve_company_data_runtime_root() == custom.resolve()


def test_explicit_runtime_beats_environment(monkeypatch, tmp_path):
    env_root = tmp_path / "env"
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("COMPANY_DATA_RUNTIME_ROOT", str(env_root))
    assert (
        resolve_company_data_runtime_root(explicit)
        == explicit.resolve()
    )
