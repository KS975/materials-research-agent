from pathlib import Path
import ast


def _load_route_helper():
    source = (
        Path(__file__).resolve().parents[2] / "api" / "chat_ui.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_looks_like_company_real_data"
    )
    module = ast.Module(body=[node], type_ignores=[])
    ns = {}
    exec(compile(module, "chat_ui_route_helper", "exec"), ns)
    return ns["_looks_like_company_real_data"]


def test_real_sample_count_routes_to_company_data():
    route = _load_route_helper()
    assert route("真实样本多少？") is True
    assert route("真实样品有多少") is True
    assert route("公司真实数据量是多少") is True
    assert route("有几条真实数据") is True


def test_existing_company_markers_still_route():
    route = _load_route_helper()
    assert route("查看公司真实数据") is True
    assert route("海科数据概况") is True


def test_unqualified_sample_question_does_not_hijack_other_routes():
    route = _load_route_helper()
    assert route("训练样本多少？") is False
    assert route("这个模型样本量够吗？") is False
