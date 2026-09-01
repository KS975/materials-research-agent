from pathlib import Path
import subprocess

from demo import MondayDemoService
from scripts.prepare_monday_demo import _configure_utf8_stdio


def test_demo_service_forces_utf8_into_child(monkeypatch):
    service = MondayDemoService(
        project_root=Path(__file__).resolve().parents[2]
    )
    captured = {}

    class Result:
        returncode = 0
        stdout = "target_metric: 冲击强度\n"
        stderr = ""

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = service._run(["-c", "print('冲击强度')"])

    assert result["pass"] is True
    assert captured["env"]["PYTHONUTF8"] == "1"
    assert captured["env"]["PYTHONIOENCODING"] == "utf-8"
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"


def test_actual_child_can_print_chinese():
    service = MondayDemoService(
        project_root=Path(__file__).resolve().parents[2]
    )
    result = service._run(
        ["-c", "print('target_metric: 冲击强度')"],
        timeout=10,
    )
    assert result["pass"] is True
    assert "target_metric: 冲击强度" in result["stdout_tail"]


def test_top_level_utf8_configurator_exists():
    assert callable(_configure_utf8_stdio)
