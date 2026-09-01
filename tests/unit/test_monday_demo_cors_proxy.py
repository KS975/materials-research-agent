
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEMO_API = (
    ROOT / "frontend" / "src" / "demo_api.js"
).read_text(encoding="utf-8")
VITE = (
    ROOT / "frontend" / "vite.config.js"
).read_text(encoding="utf-8")


def test_demo_api_defaults_to_same_origin():
    assert 'VITE_API_BASE_URL || ""' in DEMO_API
    assert '|| "http://127.0.0.1:8000"' not in DEMO_API


def test_demo_api_uses_relative_api_path():
    assert "/api/v1/demo-ui/status" in DEMO_API


def test_vite_proxies_api_to_fastapi():
    assert '"/api"' in VITE
    assert 'target: "http://127.0.0.1:8000"' in VITE
