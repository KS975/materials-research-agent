
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = (ROOT/"frontend"/"src"/"App.jsx").read_text(encoding="utf-8")
DEMO_API = (ROOT/"frontend"/"src"/"demo_api.js").read_text(encoding="utf-8")


def test_explicit_demo_projects():
    for text in (
        "modeling:9010",
        "optimization:9018",
        "feedback:9026",
        "autonomy:9036",
        "companyRealData:930066",
    ):
        assert text in APP


def test_top_buttons_use_correct_demo_projects():
    assert "V0.3 · 9036" in APP
    assert "V0.2 · 9026" in APP
    assert "模型 · 9010" in APP


def test_status_functions_default_to_correct_projects():
    assert "projectIdOverride=MONDAY_DEMO_PROJECTS.modeling" in APP
    assert "projectIdOverride=MONDAY_DEMO_PROJECTS.feedback" in APP
    assert "projectIdOverride=MONDAY_DEMO_PROJECTS.autonomy" in APP


def test_failures_are_visible_as_assistant_messages():
    assert "function appendUiFailure" in APP
    assert 'kind:"ui_action_error"' in APP


def test_demo_404_has_restart_guidance():
    assert "HTTP 404" in DEMO_API
    assert "重启 FastAPI 后端" in DEMO_API
