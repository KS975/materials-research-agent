from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from engine import __version__
from engine.contracts import QualityThresholdConfig
from engine.contracts import (
    ClosureConfig,
    CleaningConfig,
    LeakageConfig,
    TestConsistencySpec,
)
from engine.dataset.test_data_factory import (
    PerturbationSpec,
    generate_perturbation,
    save_perturbation,
)
from engine.governance.gate import apply_modeling_gate
from engine.governance.quality import run_quality_checks
from engine.ingestion.reader import read_tabular
from engine.ingestion.constraints import read_constraints_xlsx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="engine")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    quality = subparsers.add_parser("quality", help="run data quality checks and modeling gate")
    quality.add_argument("--input", required=True)
    quality.add_argument("--target-field", action="append", required=True)
    quality.add_argument("--feature-field", action="append", default=None)
    quality.add_argument("--identifier-field", action="append", default=None)
    quality.add_argument("--thresholds", default=None)
    quality.add_argument("--closure-required-field", action="append", default=None)
    quality.add_argument("--min-closure-ratio", type=float, default=None)
    quality.add_argument("--post-experiment-field", action="append", default=None)
    quality.add_argument("--leakage-field", action="append", default=None)
    quality.add_argument("--consistency-config", default=None)
    quality.add_argument("--output", default=None)

    test_data = subparsers.add_parser("test-data", help="generate a perturbed test dataset")
    test_data.add_argument("--input", required=True)
    test_data.add_argument("--kind", required=True)
    test_data.add_argument("--target-field", action="append", required=True)
    test_data.add_argument("--output-dir", default="engine/artifacts/datasets")
    test_data.add_argument("--affected-ratio", type=float, default=0.1)
    test_data.add_argument("--magnitude", type=float, default=0.05)
    test_data.add_argument("--seed", type=int, default=42)
    test_data.add_argument("--param", action="append", default=None)
    test_data.add_argument("--component", action="append", default=None)

    build = subparsers.add_parser("build-dataset", help="build a versioned dataset artifact")
    build.add_argument("--input", required=True)
    build.add_argument("--target-field", action="append", required=True)
    build.add_argument("--feature-field", action="append", default=None)
    build.add_argument("--identifier-field", action="append", default=None)
    build.add_argument("--thresholds", default=None)
    build.add_argument("--closure-required-field", action="append", default=None)
    build.add_argument("--min-closure-ratio", type=float, default=None)
    build.add_argument("--post-experiment-field", action="append", default=None)
    build.add_argument("--leakage-field", action="append", default=None)
    build.add_argument("--consistency-config", default=None)
    build.add_argument("--drop-field", action="append", default=None)
    build.add_argument("--no-drop-missing-target", action="store_true")
    build.add_argument("--no-drop-duplicates", action="store_true")
    build.add_argument("--winsorize", action="store_true")
    build.add_argument("--output-dir", default="engine/artifacts/datasets")
    build.add_argument("--output", default=None)

    preprocess = subparsers.add_parser(
        "preprocess",
        help="run preset-driven preprocessing and build one dataset artifact",
    )
    preprocess.add_argument("--input", required=True)
    preprocess.add_argument("--config", default=None)
    preprocess.add_argument("--output-dir", default="engine/artifacts/datasets")
    preprocess.add_argument("--output", default=None)

    train = subparsers.add_parser(
        "train",
        help="train registered targets from a versioned DatasetArtifact",
    )
    train.add_argument("--input", required=True)
    train.add_argument("--config", default=None)
    train.add_argument("--output-dir", default="engine/artifacts/models")
    train.add_argument("--model-registry", default=None)
    train.add_argument("--output", default=None)

    predict = subparsers.add_parser("predict", help="predict with a registered model")
    predict.add_argument("--input", required=True)
    predict.add_argument("--registry", default="engine/artifacts/models/model-registry.json")
    predict.add_argument("--model-id", default=None)
    predict.add_argument("--target-name", default=None)
    predict.add_argument("--dataset-id", default=None)
    predict.add_argument("--version", default=None)
    predict.add_argument("--output", default=None)

    optimize = subparsers.add_parser(
        "optimize",
        help="optimize a formula with registered target models",
    )
    optimize.add_argument("--request", required=True)
    optimize.add_argument("--registry", default=None)
    optimize.add_argument("--output-dir", default="engine/artifacts/optimizations")
    optimize.add_argument("--output", default=None)

    next_experiments = subparsers.add_parser(
        "recommend-experiments",
        help="recommend the next experiments with GP BO or cold-start design",
    )
    next_experiments.add_argument("--request", required=True)
    next_experiments.add_argument("--registry", default=None)
    next_experiments.add_argument("--output-dir", default="engine/artifacts/optimizations")
    next_experiments.add_argument("--output", default=None)

    chart_data = subparsers.add_parser(
        "chart-data",
        help="expose UI-neutral chart and table datasets from an engine report",
    )
    chart_data.add_argument("--input", required=True)
    chart_data.add_argument(
        "--source-kind",
        choices=["auto", "preprocessing", "training", "prediction", "optimization"],
        default="auto",
    )
    chart_data.add_argument("--output", default=None)

    constraints = subparsers.add_parser(
        "constraints",
        help="read a constraints workbook into a canonical JSON contract",
    )
    constraints.add_argument("--input", required=True)
    constraints.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "quality":
        payload = _run_quality(args)
    elif args.command == "test-data":
        payload = _run_test_data(args)
    elif args.command == "build-dataset":
        payload = _run_build_dataset(args)
    elif args.command == "preprocess":
        payload = _run_preprocess(args)
    elif args.command == "train":
        payload = _run_train(args)
    elif args.command == "predict":
        payload = _run_predict(args)
    elif args.command == "optimize":
        payload = _run_optimize(args)
    elif args.command == "recommend-experiments":
        payload = _run_recommend_experiments(args)
    elif args.command == "chart-data":
        payload = _run_chart_data(args)
    elif args.command == "constraints":
        payload = read_constraints_xlsx(args.input)
    else:  # pragma: no cover - argparse enforces valid commands
        parser.error("unknown command")

    output_argument = getattr(args, "output", None)
    if output_argument:
        rendered = json.dumps(
            payload, ensure_ascii=False, indent=2, default=str
        )
        output = Path(output_argument)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        # JSON Unicode escapes avoid GBK/UTF-8 console mismatches on Windows.
        print(json.dumps(payload, ensure_ascii=True, indent=2, default=str))
    return 0


def _run_quality(args: argparse.Namespace) -> dict:
    dataframe = read_tabular(args.input)
    threshold_payload = None
    if args.thresholds:
        threshold_payload = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    thresholds = QualityThresholdConfig.from_dict(threshold_payload)
    context = _quality_context(args)
    report = run_quality_checks(
        dataframe,
        target_fields=args.target_field,
        feature_fields=args.feature_field,
        identifier_fields=args.identifier_field,
        thresholds=thresholds,
        **context,
    )
    gate = apply_modeling_gate(report)
    return {"quality_report": report.to_dict(), "modeling_gate": gate.to_dict()}


def _run_test_data(args: argparse.Namespace) -> dict:
    params = _parse_params(args.param)
    if args.component:
        components = [
            {"kind": kind, "params": dict(params)}
            for kind in args.component
        ]
        params = {"components": components, **params}
    spec = PerturbationSpec(
        kind=args.kind,
        target_fields=args.target_field,
        affected_ratio=args.affected_ratio,
        magnitude=args.magnitude,
        random_seed=args.seed,
        params=params,
    )
    result = generate_perturbation(args.input, spec)
    artifact_dir = save_perturbation(result, args.output_dir, engine_version=__version__)
    return {
        "artifact_dir": str(artifact_dir),
        "source_hash": result.source_hash,
        "data_hash": result.perturbation_hash,
        "kind": spec.kind,
    }


def _parse_params(values: list[str] | None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError("--param must use key=value format")
        key, raw_value = value.split("=", 1)
        try:
            params[key] = json.loads(raw_value)
        except json.JSONDecodeError:
            params[key] = raw_value
    return params


def _run_build_dataset(args: argparse.Namespace) -> dict:
    from engine.dataset.builder import build_dataset
    from engine.ingestion.reader import hash_dataframe

    dataframe = read_tabular(args.input)
    threshold_payload = None
    if args.thresholds:
        threshold_payload = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    thresholds = QualityThresholdConfig.from_dict(threshold_payload)
    context = _quality_context(args)
    report = run_quality_checks(
        dataframe,
        target_fields=args.target_field,
        feature_fields=args.feature_field,
        identifier_fields=args.identifier_field,
        thresholds=thresholds,
        **context,
    )
    gate = apply_modeling_gate(report)
    artifact = build_dataset(
        dataframe,
        target_fields=args.target_field,
        feature_fields=args.feature_field,
        identifier_fields=args.identifier_field,
        quality_report=report,
        gate_result=gate,
        cleaning_config=CleaningConfig(
            drop_missing_target_rows=not args.no_drop_missing_target,
            drop_exact_duplicates=not args.no_drop_duplicates,
            drop_fields=args.drop_field or [],
            winsorize_numeric_outliers=args.winsorize,
        ),
        source_uri=str(Path(args.input).resolve()),
        source_hash=hash_dataframe(dataframe),
        output_dir=args.output_dir,
    )
    return {
        "quality_report": report.to_dict(),
        "modeling_gate": gate.to_dict(),
        "dataset_artifact": artifact.to_dict(),
    }


def _run_preprocess(args: argparse.Namespace) -> dict:
    from engine.dataset.preprocessing import run_dataset_preprocessing

    dataframe = read_tabular(args.input)
    user_config = None
    if args.config:
        user_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_dataset_preprocessing(
        dataframe,
        user_config=user_config,
        source_uri=str(Path(args.input).resolve()),
        output_dir=args.output_dir,
    )
    return result.to_dict()


def _run_train(args: argparse.Namespace) -> dict:
    from engine.dataset.loader import load_dataset_artifact
    from engine.modeling.config import resolve_training_config
    from engine.modeling.trainer import train_models

    loaded = load_dataset_artifact(args.input)
    user_config = None
    if args.config:
        user_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config, resolution = resolve_training_config(
        loaded.dataframe,
        metadata=loaded.metadata,
        user_config=user_config,
    )
    gate = loaded.lineage.get("modeling_gate", {})
    if gate.get("decision") == "FAIL":
        raise ValueError("DatasetArtifact modeling gate is FAIL; training is blocked")

    result = train_models(
        loaded.dataframe,
        config,
        dataset_artifact_id=loaded.artifact.dataset_id,
        dataset_data_hash=loaded.artifact.data_hash,
        source_uri=loaded.artifact.source_uri or str(loaded.artifact_dir),
        output_dir=args.output_dir,
        registry_path=args.model_registry,
    )
    return {
        "training_config_resolution": resolution,
        "modeling_gate": gate,
        "dataset_artifact": loaded.artifact.to_dict(),
        "training_run": result.to_dict(),
    }


def _run_predict(args: argparse.Namespace) -> dict:
    from engine.modeling.predictor import predict_with_model
    from engine.modeling.registry import load_registered_model

    dataframe = read_tabular(args.input)
    loaded = load_registered_model(
        args.registry,
        model_id=args.model_id,
        target_name=args.target_name,
        dataset_artifact_id=args.dataset_id,
        version=args.version,
    )
    predictions = predict_with_model(loaded.bundle, dataframe)
    return {
        "model": {
            "model_id": loaded.bundle["model_id"],
            "version": loaded.bundle["version"],
            "target_name": loaded.bundle["target_name"],
            "algorithm": loaded.bundle["algorithm"],
        },
        "predictions": [item.to_dict() for item in predictions],
    }


def _run_optimize(args: argparse.Namespace) -> dict:
    from engine.optimization.contracts import OptimizationRequest
    from engine.optimization.service import optimize_formula

    request = OptimizationRequest.from_dict(_read_request(args))
    if args.registry:
        request.model_registry_path = args.registry
    result = optimize_formula(request, output_dir=args.output_dir)
    return result.to_dict()


def _run_recommend_experiments(args: argparse.Namespace) -> dict:
    from engine.optimization.contracts import OptimizationRequest
    from engine.optimization.service import optimize_next_experiments

    request = OptimizationRequest.from_dict(_read_request(args))
    if args.registry:
        request.model_registry_path = args.registry
    result = optimize_next_experiments(request, output_dir=args.output_dir)
    return result.to_dict()


def _read_request(args: argparse.Namespace) -> dict[str, Any]:
    return json.loads(Path(args.request).read_text(encoding="utf-8"))


def _run_chart_data(args: argparse.Namespace) -> dict:
    from engine.visualization import build_visualization_bundle

    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    source_kind = args.source_kind
    if source_kind == "auto":
        if "training_run" in report:
            source_kind = "training"
        elif "predictions" in report:
            source_kind = "prediction"
        elif "execution_report" in report:
            source_kind = "preprocessing"
        elif report.get("record_type") == "optimization_result":
            source_kind = "optimization"
        else:
            raise ValueError("cannot infer visualization source kind")
    bundle = build_visualization_bundle(
        report,
        source_kind=source_kind,
        source_uri=str(Path(args.input).resolve()),
    )
    return bundle.to_dict()


def _quality_context(args: argparse.Namespace) -> dict:
    closure_config = ClosureConfig(
        identifier_fields=args.identifier_field or [],
        required_fields=args.closure_required_field or [],
        min_closure_ratio=(
            args.min_closure_ratio if args.min_closure_ratio is not None else 0.95
        ),
    )
    leakage_config = LeakageConfig(
        post_experiment_fields=args.post_experiment_field or [],
        forbidden_fields=args.leakage_field or [],
    )
    consistency_specs: list[TestConsistencySpec] = []
    if args.consistency_config:
        payload = json.loads(Path(args.consistency_config).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("consistency_specs", [])
        consistency_specs = [TestConsistencySpec(**item) for item in payload]
    return {
        "closure_config": closure_config,
        "leakage_config": leakage_config,
        "consistency_specs": consistency_specs,
    }


if __name__ == "__main__":
    raise SystemExit(main())
