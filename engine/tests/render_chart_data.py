from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def render_bundle(input_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Development-only renderer; production code exposes data without plotting."""
    import matplotlib.pyplot as plt

    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    datasets = payload.get("datasets", [])
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    for dataset in datasets:
        if dataset.get("dataset_kind") == "table":
            continue
        path = root / f"{_safe_name(dataset['dataset_id'])}.png"
        _render_dataset(dataset, path)
        rendered.append(path)
        plt.close("all")

    manifest = {
        "source": str(Path(input_path).resolve()),
        "rendered_files": [str(path.resolve()) for path in rendered],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return rendered


def _render_dataset(dataset: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    chart_type = dataset.get("chart_type")
    records = dataset.get("records", [])
    if not records:
        return
    if chart_type in {"bar", "horizontal_bar"}:
        _render_bar(dataset, path, horizontal=chart_type == "horizontal_bar")
    elif chart_type == "scatter":
        _render_scatter(dataset, path)
    elif chart_type == "line":
        _render_line(dataset, path)
    elif chart_type == "heatmap":
        _render_heatmap(dataset, path)
    elif chart_type == "parallel_coordinates":
        _render_parallel_coordinates(dataset, path)
    else:
        raise ValueError(f"unsupported test chart type: {chart_type}")


def _render_bar(
    dataset: dict[str, Any],
    path: Path,
    *,
    horizontal: bool,
) -> None:
    import matplotlib.pyplot as plt

    records = dataset.get("records", [])
    x_field = dataset["x_field"]
    y_fields = list(dataset["y_fields"])
    series_field = dataset.get("series_field")
    if horizontal:
        if series_field:
            records = _top_records_by_abs_value(records, x_field, series_field)
        category_field = y_fields[0]
        value_fields = [x_field]
    else:
        category_field = x_field
        value_fields = y_fields
    figure, axes = plt.subplots(
        len(value_fields), 1,
        figsize=(10, max(4, 3 * len(value_fields))),
        squeeze=False,
    )
    x_values = [_label(record.get(category_field)) for record in records]
    for axis, y_field in zip(axes.flat, value_fields):
        if series_field:
            series_names = list(dict.fromkeys(
                _label(record.get(series_field)) for record in records
            ))
            width = 0.8 / max(1, len(series_names))
            for series_index, series_name in enumerate(series_names):
                values_by_x = {
                    _label(record.get(category_field)): _number(record.get(y_field))
                    for record in records
                    if _label(record.get(series_field)) == series_name
                }
                positions = [
                    index - 0.4 + width * (series_index + 0.5)
                    for index in range(len(x_values))
                ]
                values = [values_by_x.get(value) for value in x_values]
                plotted = [
                    (position, value)
                    for position, value in zip(positions, values)
                    if value is not None
                ]
                _bar_plot(
                    axis,
                    [position for position, _ in plotted],
                    [value for _, value in plotted],
                    width,
                    series_name,
                    horizontal,
                )
        if horizontal:
            axis.set_yticks(range(len(x_values)), x_values, fontsize=8)
        else:
            axis.set_xticks(range(len(x_values)), x_values, rotation=35, ha="right")
    else:
        values = [_number(record.get(y_field)) for record in records]
        positions = range(len(x_values))
        plotted = [
            (position, value)
            for position, value in zip(positions, values)
            if value is not None
        ]
        _bar_plot(
            axis,
            [position for position, _ in plotted],
            [value for _, value in plotted],
            0.8,
            y_field,
            horizontal,
        )
        if horizontal:
            axis.set_yticks(range(len(x_values)), x_values, fontsize=8)
        else:
            axis.set_xticks(range(len(x_values)), x_values, rotation=35, ha="right")
        axis.set_title(y_field)
        axis.grid(alpha=0.2)
    figure.suptitle(dataset.get("title", ""), fontsize=13)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")


def _bar_plot(
    axis,
    positions,
    values,
    width: float,
    label: str,
    horizontal: bool,
) -> None:
    if horizontal:
        axis.barh(positions, values, height=width, label=label)
    else:
        axis.bar(positions, values, width=width, label=label)


def _render_scatter(dataset: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    x_field = dataset["x_field"]
    y_fields = list(dataset["y_fields"])
    records = dataset.get("records", [])
    x_values = [_number(record.get(x_field)) for record in records]
    figure, axes = plt.subplots(
        len(y_fields), 1,
        figsize=(8, max(5, 4 * len(y_fields))),
        squeeze=False,
    )
    for axis, y_field in zip(axes.flat, y_fields):
        y_values = [_number(record.get(y_field)) for record in records]
        axis.scatter(x_values, y_values, s=26, alpha=0.75)
        if y_field == "y_pred" and x_field == "y_true":
            minimum = min(x_values + y_values)
            maximum = max(x_values + y_values)
            axis.plot([minimum, maximum], [minimum, maximum], "--", alpha=0.7)
        if y_field == "residual":
            axis.axhline(0, linestyle="--", alpha=0.7)
        axis.set_xlabel(x_field)
        axis.set_ylabel(y_field)
        axis.grid(alpha=0.2)
    figure.suptitle(dataset.get("title", ""), fontsize=13)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")


def _render_line(dataset: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    x_field = dataset["x_field"]
    y_fields = list(dataset["y_fields"])
    records = dataset.get("records", [])
    x_values = [_number(record.get(x_field)) for record in records]
    figure, axis = plt.subplots(figsize=(10, 5))
    for y_field in y_fields:
        axis.plot(x_values, [_number(record.get(y_field)) for record in records], label=y_field)
    axis.set_xlabel(x_field)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.suptitle(dataset.get("title", ""), fontsize=13)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")


def _render_heatmap(dataset: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    records = dataset.get("records", [])
    columns = [column["name"] for column in dataset.get("columns", [])]
    numeric_columns = [
        name for name in columns
        if all(_is_number(record.get(name)) for record in records)
    ]
    matrix = [
        [_number(record.get(name)) for name in numeric_columns]
        for record in records
    ]
    figure, axis = plt.subplots(figsize=(max(8, len(numeric_columns)), max(5, len(records) * 0.25)))
    image = axis.imshow(matrix, aspect="auto", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(numeric_columns)), numeric_columns, rotation=35, ha="right")
    if records:
        row_field = dataset.get("x_field", columns[0] if columns else None)
        if row_field:
            axis.set_yticks(
                range(len(records)),
                [_label(record.get(row_field)) for record in records],
                fontsize=8,
            )
    figure.suptitle(dataset.get("title", ""), fontsize=13)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")


def _render_parallel_coordinates(dataset: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    records = dataset.get("records", [])
    y_fields = list(dataset["y_fields"])
    normalized = []
    for field in y_fields:
        values = [_number(record.get(field)) for record in records]
        minimum, maximum = min(values), max(values)
        span = maximum - minimum
        normalized.append([
            (value - minimum) / span if span else 0.5 for value in values
        ])
    figure, axis = plt.subplots(figsize=(10, 5))
    for row_index in range(len(records)):
        axis.plot(range(len(y_fields)), [field[row_index] for field in normalized], alpha=0.55)
    axis.set_xticks(range(len(y_fields)), y_fields, rotation=25, ha="right")
    axis.set_ylabel("normalized value")
    axis.grid(alpha=0.2)
    figure.suptitle(dataset.get("title", ""), fontsize=13)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")


def _number(value: Any) -> float:
    return 0.0 if value is None else float(value)


def _top_records_by_abs_value(
    records: list[dict[str, Any]],
    value_field: str,
    series_field: str,
    limit: int = 15,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get(series_field)), []).append(record)
    selected: list[dict[str, Any]] = []
    for items in grouped.values():
        selected.extend(
            sorted(
                items,
                key=lambda item: abs(_number(item.get(value_field))),
                reverse=True,
            )[:limit]
        )
    return selected


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _label(value: Any) -> str:
    return str(value)


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"_", "-", "."} else "_"
        for character in value
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="render-chart-data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="engine/artifacts/reports/charts")
    arguments = parser.parse_args(argv)
    rendered = render_bundle(arguments.input, arguments.output_dir)
    print(json.dumps({"rendered_count": len(rendered)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
