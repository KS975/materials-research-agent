from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from app.config import get_settings
from company_data import resolve_company_data_runtime_root


DEMO_ID = "MONDAY_V030_FULL_DEMO"
V013_PROJECT_ID = 9010
V014_PROJECT_ID = 9018
V020_PROJECT_ID = 9026
V030_PROJECT_ID = 9036
REAL_DATA_PROJECT_ID = 930066
MYSQL_DEMO_PROJECT_ID = 115
TARGET = "冲击强度"


class MondayDemoError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _sha256_json(payload: dict[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


class MondayDemoService:
    def __init__(
        self,
        *,
        project_root: str | Path | None = None,
        runtime_root: str | Path | None = None,
    ) -> None:
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else _project_root()
        )
        self.runtime_root = resolve_company_data_runtime_root(
            runtime_root
        )
        self.demo_dir = (
            self.runtime_root
            / "demo"
            / "monday_v030"
        )
        self.report_path = self.demo_dir / "monday_demo_report.json"
        self.command_log_path = self.demo_dir / "prepare_commands.json"

    def _run(
        self,
        args: list[str],
        *,
        timeout: int = 360,
    ) -> dict[str, Any]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root)

        # Windows: redirected Python stdout may inherit cp1252. All Monday
        # Demo child runners must be able to print Chinese metric names.
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8:replace"
        started = datetime.now(timezone.utc)
        cp = subprocess.run(
            [sys.executable, "-X", "utf8", *args],
            cwd=self.project_root,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
        )
        ended = datetime.now(timezone.utc)
        result = {
            "command": [sys.executable, "-X", "utf8", *args],
            "returncode": cp.returncode,
            "stdout_tail": cp.stdout[-12000:],
            "stderr_tail": cp.stderr[-6000:],
            "started_at": started.isoformat(),
            "ended_at": ended.isoformat(),
            "pass": cp.returncode == 0,
        }
        if cp.returncode != 0:
            raise MondayDemoError(
                "Demo prepare command failed:\n"
                + " ".join(result["command"])
                + "\nSTDOUT:\n"
                + cp.stdout[-5000:]
                + "\nSTDERR:\n"
                + cp.stderr[-5000:]
            )
        return result

    def encoding_preflight(self) -> dict[str, Any]:
        """Verify the exact Python child mode used by Monday Demo."""
        result = self._run([
            "-c",
            (
                "import sys; "
                "print('stdout_encoding=' + str(sys.stdout.encoding)); "
                "print('filesystem_encoding=' + str(sys.getfilesystemencoding())); "
                "print('utf8_mode=' + str(sys.flags.utf8_mode)); "
                "print('metric=冲击强度')"
            ),
        ], timeout=20)
        stdout = result.get("stdout_tail") or ""
        ok = (
            "metric=冲击强度" in stdout
            and "utf8_mode=1" in stdout
        )
        return {
            "pass": ok,
            "stdout": stdout,
            "child_command": result.get("command"),
        }

    def _external_readiness(self) -> dict[str, Any]:
        settings = get_settings()

        mysql_configured = bool(
            settings.business_db_host.strip()
            and settings.business_db_user.strip()
            and settings.business_db_password.get_secret_value()
            and settings.business_db_name.strip()
        )
        dev_user = os.getenv("DEV_USER_ID", "").strip()
        dev_company = os.getenv("DEV_COMPANY_ID", "").strip()
        dev_projects = {
            x.strip()
            for x in os.getenv("DEV_PROJECT_IDS", "").split(",")
            if x.strip()
        }
        mysql_scope_ready = (
            bool(dev_user and dev_company)
            and str(MYSQL_DEMO_PROJECT_ID) in dev_projects
        )

        llm_ready = bool(
            settings.llm_enabled
            and settings.llm_base_url.strip()
            and settings.llm_api_key.get_secret_value().strip()
            and settings.llm_model.strip()
        )
        embedding_ready = bool(
            settings.embedding_base_url.strip()
            and settings.embedding_api_key_value()
            and settings.embedding_model.strip()
        )

        qdrant_path = (
            Path(settings.qdrant_local_path)
            if settings.qdrant_mode == "local"
            else None
        )
        qdrant_ready = (
            bool(qdrant_path and qdrant_path.exists())
            if settings.qdrant_mode == "local"
            else bool(settings.qdrant_url.strip())
        )

        company_current = (
            self.runtime_root / "company_data" / "current.json"
        )
        real_company_data_ready = company_current.is_file()

        return {
            "v011": {
                "business_mysql_configured": mysql_configured,
                "project_115_dev_scope_ready": mysql_scope_ready,
                "live_ready": mysql_configured and mysql_scope_ready,
                "note": (
                    "V0.1.1 live demo uses read-only business MySQL. "
                    "No fixture result is substituted."
                ),
            },
            "v012": {
                "current_attachment_parser_ready": True,
                "llm_ready": llm_ready,
                "embedding_ready": embedding_ready,
                "qdrant_ready": qdrant_ready,
                "historical_rag_live_ready": (
                    llm_ready and embedding_ready and qdrant_ready
                ),
                "joint_mysql_rag_live_ready": (
                    mysql_configured
                    and mysql_scope_ready
                    and llm_ready
                    and embedding_ready
                    and qdrant_ready
                ),
            },
            "company_real_data": {
                "ready": real_company_data_ready,
                "current_pointer": (
                    _rel(company_current, self.project_root)
                    if real_company_data_ready
                    else None
                ),
            },
        }

    def _prepare_v013(self) -> list[dict[str, Any]]:
        commands: list[dict[str, Any]] = []
        commands.append(self._run([
            "-m", "scripts.build_v013_t10_training_fixture",
            "--project-id", str(V013_PROJECT_ID),
            "--target", TARGET,
            "--samples", "60",
            "--seed", "42",
        ]))
        reality = ".runtime/v013/fixtures/t10_training_reality.json"
        dataset = ".runtime/v013/fixtures/t10_training_dataset.csv"
        gate = (
            f".runtime/v013/gates/"
            f"project_{V013_PROJECT_ID}_{TARGET}_modeling_gate.json"
        )
        commands.append(self._run([
            "-m", "scripts.run_v013_modeling_gate",
            "--project-id", str(V013_PROJECT_ID),
            "--target", TARGET,
            "--reality-json", reality,
        ]))
        commands.append(self._run([
            "-m", "scripts.run_v013_train",
            "--project-id", str(V013_PROJECT_ID),
            "--target", TARGET,
            "--dataset-csv", dataset,
            "--gate-json", gate,
        ]))
        commands.append(self._run([
            "-m", "scripts.run_v013_model_compare",
            "--project-id", str(V013_PROJECT_ID),
            "--target", TARGET,
            "--dataset-csv", dataset,
            "--gate-json", gate,
        ]))
        commands.append(self._run([
            "-m", "scripts.run_v013_cross_validate",
            "--project-id", str(V013_PROJECT_ID),
            "--target", TARGET,
            "--dataset-csv", dataset,
            "--gate-json", gate,
            "--folds", "5",
            "--random-state", "42",
        ]))
        commands.append(self._run([
            "-m", "scripts.build_v013_t13_ad_fixtures",
            "--dataset-csv", dataset,
        ]))
        best_model = (
            f".runtime/v013/model_comparison/"
            f"project_{V013_PROJECT_ID}_{TARGET}/best_model.joblib"
        )
        for sample in (
            "in_domain_sample.json",
            "borderline_sample.json",
            "out_of_domain_sample.json",
        ):
            commands.append(self._run([
                "-m", "scripts.run_v013_applicability_domain",
                "--project-id", str(V013_PROJECT_ID),
                "--target", TARGET,
                "--dataset-csv", dataset,
                "--gate-json", gate,
                "--sample-json", f".runtime/v013/fixtures/t13/{sample}",
                "--best-model", best_model,
            ]))
        return commands

    def _prepare_v014(self) -> list[dict[str, Any]]:
        commands: list[dict[str, Any]] = []
        commands.append(self._run([
            "-m", "scripts.build_v014_t18_fixture",
        ]))
        commands.append(self._run([
            "-m", "scripts.run_v013_modeling_gate",
            "--project-id", str(V014_PROJECT_ID),
            "--target", TARGET,
            "--reality-json",
            ".runtime/v014/fixtures/t18/reality_冲击强度.json",
        ]))
        commands.append(self._run([
            "-m", "scripts.run_v014_bayesian_optimization",
            "--request-json",
            ".runtime/v014/fixtures/t18/bo_request.json",
            "--observations-csv",
            ".runtime/v014/fixtures/t18/initial_observations.csv",
            "--search-space-json",
            ".runtime/v014/fixtures/t18/search_space.json",
        ]))
        return commands

    def _prepare_v020(self) -> list[dict[str, Any]]:
        return [
            self._run([
                "-m", "scripts.build_v020_t26_fixture",
            ]),
            self._run([
                "-m", "scripts.run_v020_end_to_end_closed_loop",
                "--reset",
            ], timeout=420),
        ]

    def _prepare_v030(self) -> list[dict[str, Any]]:
        return [
            self._run([
                "-m", "scripts.build_v030_t36_fixture",
                "--reset",
            ]),
            self._run([
                "-m", "scripts.run_v030_end_to_end_validation",
                "--reset",
            ], timeout=480),
        ]

    def _find_single_json(
        self,
        pattern: str,
    ) -> dict[str, Any] | None:
        matches = list(self.project_root.glob(pattern))
        if not matches:
            return None
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return _read_json(matches[0])

    def build_status(self) -> dict[str, Any]:
        ext = self._external_readiness()

        gate = _read_json(
            self.project_root
            / ".runtime"
            / "v013"
            / "gates"
            / f"project_{V013_PROJECT_ID}_{TARGET}_modeling_gate.json"
        )
        cv = _read_json(
            self.project_root
            / ".runtime"
            / "v013"
            / "cross_validation"
            / f"project_{V013_PROJECT_ID}_{TARGET}"
            / "cv_report.json"
        )
        ad_reports = []
        ad_dir = (
            self.project_root
            / ".runtime"
            / "v013"
            / "applicability_domain"
            / f"project_{V013_PROJECT_ID}_{TARGET}"
        )
        if ad_dir.exists():
            for path in sorted(ad_dir.glob("*_ad_report.json")):
                data = _read_json(path)
                if data:
                    ad_reports.append(data)

        bo = _read_json(
            self.project_root
            / ".runtime"
            / "v014"
            / "bayesian_optimization"
            / f"project_{V014_PROJECT_ID}"
            / "next_5_impact_experiments"
            / "bo_report.json"
        )

        v020 = self._find_single_json(
            ".runtime/v020/end_to_end/*/end_to_end_report.json"
        )
        v030 = _read_json(
            self.project_root
            / ".runtime"
            / "v030"
            / "final_validation"
            / "V030_T36_FINAL"
            / "v030_final_validation_report.json"
        )

        v013_ready = bool(
            gate
            and gate.get("decision") == "PASS"
            and cv
            and ad_reports
        )
        v014_ready = bool(
            bo
            and len(bo.get("next_experiments") or []) == 5
        )
        v020_ready = bool(
            v020
            and v020.get("decision") == "PASS"
        )
        v030_ready = bool(
            v030
            and v030.get("status") == "PASS"
            and v030.get("component_pass_count") == 9
        )

        cv_best = (cv or {}).get("best_cv_model") or {}
        bo_next = (bo or {}).get("next_experiments") or []
        v020_rows = (
            ((v020 or {}).get("datasets") or {}).get("row_counts")
            or []
        )
        v030_normal = (v030 or {}).get("normal_loop") or {}
        v030_crash = (v030 or {}).get("crash_resume") or {}

        versions = [
            {
                "version": "V0.1.1",
                "title": "Agent + Read-only MySQL",
                "status": (
                    "LIVE_READY"
                    if ext["v011"]["live_ready"]
                    else "CONFIG_REQUIRED"
                ),
                "capabilities": [
                    "样品完整研发上下文",
                    "样品对比",
                    "性能下降原因分析",
                    "权限作用域 + READ ONLY SQL",
                ],
                "project_id": MYSQL_DEMO_PROJECT_ID,
                "demo_prompt": (
                    "为什么 3811 的冲击强度比 3809 低？"
                ),
                "live_external": True,
                "note": ext["v011"]["note"],
            },
            {
                "version": "V0.1.2",
                "title": "Attachment + Knowledge + RAG",
                "status": (
                    "LIVE_READY"
                    if ext["v012"]["joint_mysql_rag_live_ready"]
                    else (
                        "PARTIAL_READY"
                        if ext["v012"]["current_attachment_parser_ready"]
                        else "CONFIG_REQUIRED"
                    )
                ),
                "capabilities": [
                    "当前 Chat PDF/DOCX",
                    "Knowledge Index / Qdrant",
                    "历史 RAG",
                    "MySQL + 历史资料联合分析",
                ],
                "project_id": MYSQL_DEMO_PROJECT_ID,
                "demo_prompt": (
                    "3811 的冲击强度比 3809 低很多，历史上有没有类似问题？"
                    "结合数据库数据和历史报告分析一下。"
                ),
                "live_external": True,
                "readiness": ext["v012"],
            },
            {
                "version": "V0.1.3",
                "title": "Dataset + ML",
                "status": "READY" if v013_ready else "NOT_PREPARED",
                "capabilities": [
                    "Dataset Reality Check",
                    "Modeling Gate",
                    "多模型训练与比较",
                    "Cross Validation",
                    "AD / OOD",
                ],
                "project_id": V013_PROJECT_ID,
                "target_metric": TARGET,
                "demo_prompt": (
                    f"检查 Project {V013_PROJECT_ID} 的{TARGET}建模状态"
                ),
                "summary": {
                    "gate_decision": (gate or {}).get("decision"),
                    "official_model_allowed": (
                        (gate or {}).get("official_model_allowed")
                    ),
                    "best_cv_model": cv_best.get("model_name"),
                    "cv_r2_mean": cv_best.get("r2_mean"),
                    "cv_mae_mean": cv_best.get("mae_mean"),
                    "ad_statuses": sorted({
                        str(
                            (
                                report.get("applicability_domain")
                                or {}
                            ).get("status")
                        )
                        for report in ad_reports
                        if (
                            report.get("applicability_domain")
                            or {}
                        ).get("status")
                    }),
                },
            },
            {
                "version": "V0.1.4",
                "title": "Optimization + Bayesian Optimization",
                "status": "READY" if v014_ready else "NOT_PREPARED",
                "capabilities": [
                    "Search Space + Constraints",
                    "Candidate Generation",
                    "Prediction + AD",
                    "Pareto / Inverse Design",
                    "EI/PI/UCB Bayesian Optimization",
                ],
                "project_id": V014_PROJECT_ID,
                "target_metric": TARGET,
                "demo_prompt": (
                    f"Project {V014_PROJECT_ID}：查看 Bayesian Optimization 下一轮实验"
                ),
                "summary": {
                    "next_experiment_count": len(bo_next),
                    "acquisition": (
                        (bo or {}).get("bayesian_optimization") or {}
                    ).get("acquisition"),
                    "batch_strategy": (
                        (bo or {}).get("bayesian_optimization") or {}
                    ).get("batch_strategy"),
                    "ood_selected": sum(
                        1
                        for item in bo_next
                        if (item.get("applicability_domain") or {}).get("status")
                        == "OUT_OF_DOMAIN"
                    ),
                },
            },
            {
                "version": "V0.2",
                "title": "Experiment Feedback Loop",
                "status": "READY" if v020_ready else "NOT_PREPARED",
                "capabilities": [
                    "Campaign + Round",
                    "Experimental Result Ingestion",
                    "Prediction vs Measurement",
                    "Dataset Versioning",
                    "自动重训 + 模型治理",
                    "Closed-loop BO + Checkpoint/Resume",
                ],
                "project_id": V020_PROJECT_ID,
                "demo_prompt": (
                    f"Project {V020_PROJECT_ID}：查看 V0.2 闭环状态"
                ),
                "summary": {
                    "round_count": (v020 or {}).get("round_count"),
                    "total_experiments": (
                        (v020 or {}).get("experiment_integrity") or {}
                    ).get("experiment_count"),
                    "dataset_row_counts": v020_rows,
                    "net_improvement": (
                        (v020 or {}).get("best_so_far") or {}
                    ).get("net_improvement"),
                    "checkpoint_completed_count": (
                        (v020 or {}).get("checkpoints") or {}
                    ).get("completed_count"),
                    "model_auto_activation": False,
                },
            },
            {
                "version": "V0.3",
                "title": "Autonomous Experiment Orchestration",
                "status": "READY" if v030_ready else "NOT_PREPARED",
                "capabilities": [
                    "Protocol",
                    "Device Adapter / Simulator",
                    "Scheduler",
                    "Telemetry + State Machine",
                    "Safety Interlock",
                    "Automatic Result Capture",
                    "Autonomous Multi-Round Loop",
                    "Crash/Resume + Operator Override",
                ],
                "project_id": V030_PROJECT_ID,
                "demo_prompt": (
                    f"Project {V030_PROJECT_ID}：查看 V0.3 自主实验状态"
                ),
                "summary": {
                    "component_pass": (
                        f"{(v030 or {}).get('component_pass_count', 0)}/"
                        f"{(v030 or {}).get('component_total', 9)}"
                    ),
                    "round_count": v030_normal.get("round_count"),
                    "automatic_captures": (
                        v030_normal.get("automatic_captures")
                    ),
                    "manual_submissions": (
                        v030_normal.get("manual_submissions")
                    ),
                    "dataset_row_counts": (
                        v030_normal.get("dataset_row_counts")
                    ),
                    "crash_progress_percent": (
                        v030_crash.get("crash_progress_percent")
                    ),
                    "recovery_replay_idempotent": (
                        v030_crash.get("replay_idempotent")
                    ),
                    "safety_bypass_forbidden": True,
                    "real_device_connected": False,
                },
            },
        ]

        internal_ready_count = sum(
            item["status"] == "READY"
            for item in versions
            if item["version"] in {
                "V0.1.3", "V0.1.4", "V0.2", "V0.3"
            }
        )

        full_internal_ready = internal_ready_count == 4
        report = {
            "demo_id": DEMO_ID,
            "kind": "monday_demo",
            "status": (
                "READY"
                if full_internal_ready
                else "NOT_PREPARED"
            ),
            "prepared_internal_versions": (
                f"{internal_ready_count}/4"
            ),
            "generated_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "headline": (
                "V0.1.1 → V0.3 全能力演示"
            ),
            "storyline": [
                "数据访问 / 附件 / 历史知识",
                "Reality Check / Modeling Gate",
                "自动训练 / CV / AD-OOD",
                "Inverse Design / Bayesian Optimization",
                "实验结果自动回流",
                "Dataset 自动版本升级",
                "Challenger 自动重训与比较",
                "V0.3 Simulator 自主执行",
                "Telemetry / Safety / Crash-Resume",
                "下一轮 BO",
            ],
            "versions": versions,
            "projects": {
                "mysql_joint_analysis": MYSQL_DEMO_PROJECT_ID,
                "modeling": V013_PROJECT_ID,
                "optimization": V014_PROJECT_ID,
                "feedback_loop": V020_PROJECT_ID,
                "autonomy": V030_PROJECT_ID,
                "company_real_data": REAL_DATA_PROJECT_ID,
            },
            "real_data_bonus": {
                **ext["company_real_data"],
                "project_id": REAL_DATA_PROJECT_ID,
                "product": "PC/ABS FR303",
                "prompt": "FR303 有多少样本？",
                "boundary": (
                    "真实公司数据可用于查询、Reality Check 和探索性建模；"
                    "当前缺少显式测试条件/工艺参数，因此不伪造正式 BO 准入。"
                ),
            },
            "boundaries": {
                "business_mysql_read_only": True,
                "v012_external_rag_is_live_not_fixture": True,
                "v013_v014_demo_data_is_synthetic_fixture": True,
                "v020_v030_measurements_are_synthetic_fixture": True,
                "v030_simulator_only": True,
                "real_device_connected": False,
                "simulator_results_are_not_real_measurements": True,
                "automatic_model_promotion_forbidden": True,
                "operator_override_cannot_bypass_safety": True,
            },
            "recommended_demo_order": [
                "V0.1.1 MySQL 查询/对比",
                "V0.1.2 附件 + 历史联合分析",
                "真实公司数据 FR303 Reality Check（Bonus）",
                "V0.1.3 建模",
                "V0.1.4 BO 推荐下一批实验",
                "V0.2 结果回流 → Dataset → 自动重训",
                "V0.3 自动执行 → Telemetry → Safety → Crash/Resume",
            ],
            "demo_scope_project_ids": [
                MYSQL_DEMO_PROJECT_ID,
                V013_PROJECT_ID,
                V014_PROJECT_ID,
                V020_PROJECT_ID,
                V030_PROJECT_ID,
                REAL_DATA_PROJECT_ID,
            ],
        }
        report["report_sha256"] = _sha256_json(report)
        return report

    def prepare(
        self,
        *,
        reset: bool = False,
    ) -> dict[str, Any]:
        self.demo_dir.mkdir(parents=True, exist_ok=True)

        encoding_check = self.encoding_preflight()
        if not encoding_check["pass"]:
            raise MondayDemoError(
                "Python UTF-8 preflight failed before Demo preparation.\n"
                + encoding_check["stdout"]
            )

        command_log: dict[str, Any] = {
            "demo_id": DEMO_ID,
            "reset": reset,
            "encoding_preflight": encoding_check,
            "versions": {},
        }

        # Reuse already prepared deterministic demo runtime by default.
        # --reset forces a full revalidation and may take several minutes.
        before = self.build_status()
        existing = {
            item["version"]: item.get("status")
            for item in before.get("versions") or []
        }
        steps = [
            ("V0.1.3", "v013", self._prepare_v013),
            ("V0.1.4", "v014", self._prepare_v014),
            ("V0.2", "v020", self._prepare_v020),
            ("V0.3", "v030", self._prepare_v030),
        ]
        for version, key, fn in steps:
            if not reset and existing.get(version) == "READY":
                command_log["versions"][key] = [{
                    "pass": True,
                    "skipped": True,
                    "reason": "ALREADY_READY",
                }]
            else:
                command_log["versions"][key] = fn()

        _atomic_json(self.command_log_path, command_log)
        report = self.build_status()
        report["prepare_command_log"] = _rel(
            self.command_log_path, self.project_root
        )
        _atomic_json(self.report_path, report)
        return report

    def status(self) -> dict[str, Any]:
        saved = _read_json(self.report_path)
        live = self.build_status()
        if saved is not None:
            live["last_prepared_at"] = saved.get("generated_at")
            live["prepare_command_log"] = saved.get(
                "prepare_command_log"
            )
        live["report_path"] = _rel(
            self.report_path, self.project_root
        )
        return live
