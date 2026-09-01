from pathlib import Path
import subprocess

from demo import MondayDemoService


def test_demo_child_command_uses_x_utf8(monkeypatch):
    service = MondayDemoService(
        project_root=Path(__file__).resolve().parents[2]
    )
    captured_args = []
    captured_kwargs = {}

    class Result:
        returncode = 0
        stdout = "utf8_mode=1\nmetric=冲击强度\n"
        stderr = ""

    def fake_run(args, **kwargs):
        captured_args.extend(args)
        captured_kwargs.update(kwargs)
        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = service._run(["-c", "print('冲击强度')"])

    assert result["pass"] is True
    assert captured_args[0].endswith("python") or "python" in captured_args[0].lower()
    assert captured_args[1:3] == ["-X", "utf8"]
    assert captured_kwargs["env"]["PYTHONUTF8"] == "1"
    assert captured_kwargs["env"]["PYTHONIOENCODING"] == "utf-8:replace"
    assert captured_kwargs["encoding"] == "utf-8"


def test_real_encoding_preflight_passes():
    service = MondayDemoService(
        project_root=Path(__file__).resolve().parents[2]
    )
    result = service.encoding_preflight()
    assert result["pass"] is True
    assert "utf8_mode=1" in result["stdout"]
    assert "metric=冲击强度" in result["stdout"]


def test_real_child_stdout_is_utf8():
    service = MondayDemoService(
        project_root=Path(__file__).resolve().parents[2]
    )
    result = service._run([
        "-c",
        "import sys; print(sys.stdout.encoding); print('冲击强度')",
    ])
    assert result["pass"] is True
    assert "冲击强度" in result["stdout_tail"]
    assert "utf" in result["stdout_tail"].lower()
