from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re
from typing import Any

from .campaign import CampaignStore, find_round, utc_now_iso


class PredictionEvaluationError(RuntimeError):
    pass


class PredictionEvaluationValidationError(PredictionEvaluationError):
    pass


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _prediction_value_and_std(snapshot: Any, metric: str) -> tuple[float | None, float | None, str]:
    if not isinstance(snapshot, dict):
        return None, None, ''
    item = snapshot.get(metric)
    if item is None:
        return None, None, ''
    direct = _finite_number(item)
    if direct is not None:
        return direct, None, ''
    if not isinstance(item, dict):
        return None, None, ''

    value = None
    for key in ('value','prediction','predicted_value','posterior_mean','mean'):
        value = _finite_number(item.get(key))
        if value is not None:
            break

    std = None
    for key in ('std','posterior_std','prediction_std','sigma'):
        std = _finite_number(item.get(key))
        if std is not None:
            break
    if std is not None and std <= 0:
        std = None

    return value, std, str(item.get('source') or '').strip()


def _r2_score(actual: list[float], predicted: list[float]) -> float | None:
    if len(actual) < 2:
        return None
    mean_actual = sum(actual) / len(actual)
    ss_tot = sum((x - mean_actual) ** 2 for x in actual)
    if ss_tot <= 0:
        return None
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    return 1.0 - ss_res / ss_tot


def build_prediction_measurement_report(
    campaign: dict[str, Any],
    *,
    round_id: str,
    metric: str,
) -> dict[str, Any]:
    metric = str(metric or '').strip()
    if not metric:
        raise PredictionEvaluationValidationError('metric 不能为空')
    if metric not in (campaign.get('target_metrics') or []):
        raise PredictionEvaluationValidationError(
            f'metric 不属于 campaign target_metrics: {metric}'
        )

    round_record = find_round(campaign, round_id)
    experiments = round_record.get('experiments')
    if experiments is None:
        raise PredictionEvaluationValidationError('该 Round 尚未注册 experiments')

    rows: list[dict[str, Any]] = []
    excluded_missing_prediction: list[str] = []
    eligible_completed_count = 0

    for experiment in experiments:
        result = experiment.get('result') or {}
        if experiment.get('status') != 'COMPLETED' or result.get('training_eligible') is not True:
            continue
        measurements = result.get('measurements') or {}
        actual = _finite_number(measurements.get(metric))
        if actual is None:
            continue
        eligible_completed_count += 1

        predicted, std, source = _prediction_value_and_std(
            experiment.get('prediction_snapshot') or {}, metric
        )
        if predicted is None:
            excluded_missing_prediction.append(experiment['candidate_id'])
            continue

        residual = actual - predicted
        abs_error = abs(residual)
        relative_abs_error = None if actual == 0 else abs_error / abs(actual)
        signed_relative_error = None if actual == 0 else residual / abs(actual)

        row = {
            'candidate_id': experiment['candidate_id'],
            'metric': metric,
            'predicted': predicted,
            'actual': actual,
            'residual': residual,
            'absolute_error': abs_error,
            'relative_absolute_error': relative_abs_error,
            'signed_relative_error': signed_relative_error,
            'prediction_source': source,
            'prediction_std': std,
            'test_condition_signature': result.get('test_condition_signature'),
            'unit': (result.get('units') or {}).get(metric),
        }
        if std is not None:
            z = residual / std
            row.update({
                'z_score': z,
                'absolute_z_score': abs(z),
                'within_1sigma': abs(residual) <= std,
                'within_2sigma': abs(residual) <= 2.0 * std,
                'overconfident_2sigma_miss': abs(residual) > 2.0 * std,
            })
        else:
            row.update({
                'z_score': None,
                'absolute_z_score': None,
                'within_1sigma': None,
                'within_2sigma': None,
                'overconfident_2sigma_miss': None,
            })
        rows.append(row)

    if not rows:
        raise PredictionEvaluationValidationError(
            f'没有可用于 prediction-vs-measurement 的 {metric} 样本'
        )

    actual = [row['actual'] for row in rows]
    predicted = [row['predicted'] for row in rows]
    residuals = [row['residual'] for row in rows]
    abs_errors = [row['absolute_error'] for row in rows]

    n = len(rows)
    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum(r*r for r in residuals) / n)
    bias = sum(residuals) / n
    rel = [row['relative_absolute_error'] for row in rows if row['relative_absolute_error'] is not None]

    largest = max(rows, key=lambda row: (row['absolute_error'], row['candidate_id']))
    uncertainty_rows = [row for row in rows if row['prediction_std'] is not None]
    n_std = len(uncertainty_rows)

    if n_std:
        coverage_1 = sum(bool(row['within_1sigma']) for row in uncertainty_rows) / n_std
        coverage_2 = sum(bool(row['within_2sigma']) for row in uncertainty_rows) / n_std
        mean_abs_z = sum(float(row['absolute_z_score']) for row in uncertainty_rows) / n_std
        overconfident = [
            row['candidate_id'] for row in uncertainty_rows
            if row['overconfident_2sigma_miss'] is True
        ]
    else:
        coverage_1 = coverage_2 = mean_abs_z = None
        overconfident = []

    report = {
        'stage': 'V0.2-T21_prediction_vs_measurement',
        'schema_version': 1,
        'generated_at': utc_now_iso(),
        'campaign_id': campaign.get('campaign_id'),
        'project_id': campaign.get('project_id'),
        'round_id': round_id,
        'metric': metric,
        'round_status': round_record.get('status'),
        'counts': {
            'planned_experiments': len(experiments),
            'eligible_completed_experiments': eligible_completed_count,
            'evaluated': n,
            'excluded_missing_prediction': len(excluded_missing_prediction),
        },
        'aggregate': {
            'mae': mae,
            'rmse': rmse,
            'bias_actual_minus_predicted': bias,
            'r2': _r2_score(actual, predicted),
            'mean_relative_absolute_error': (sum(rel) / len(rel)) if rel else None,
        },
        'largest_error': {
            'candidate_id': largest['candidate_id'],
            'predicted': largest['predicted'],
            'actual': largest['actual'],
            'residual': largest['residual'],
            'absolute_error': largest['absolute_error'],
        },
        'uncertainty': {
            'samples_with_std': n_std,
            'coverage_1sigma': coverage_1,
            'coverage_2sigma': coverage_2,
            'mean_absolute_z_score': mean_abs_z,
            'overconfident_2sigma_miss_count': len(overconfident),
            'overconfident_candidate_ids': overconfident,
            'interpretation': (
                '2σ miss 表示实际值落在模型给出的 ±2σ 区间之外；它是诊断信号，不等于自动判定模型失效。'
            ),
        },
        'excluded': {
            'missing_prediction_candidate_ids': excluded_missing_prediction,
            'note': 'FAILED / INVALID / NOT_TESTED 不进入预测误差统计。',
        },
        'rows': rows,
        'note': 'residual = actual - predicted；本报告只评估预测表现，不修改模型或数据集。',
    }
    return report


def _safe_metric_filename(metric: str) -> str:
    value = re.sub(r'[^0-9A-Za-z_.\-\u4e00-\u9fff]+', '_', metric.strip())
    return value or 'metric'


class PredictionEvaluationService:
    def __init__(self, runtime_root: str | Path = '.runtime') -> None:
        self.runtime_root = Path(runtime_root)
        self.store = CampaignStore(runtime_root)

    def evaluate(self, campaign_id: str, *, round_id: str, metric: str, persist: bool = True) -> dict[str, Any]:
        campaign = self.store.load(campaign_id)
        report = build_prediction_measurement_report(
            campaign, round_id=round_id, metric=metric
        )
        if persist:
            path = self.report_path(campaign_id, round_id=round_id, metric=metric)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + '.tmp')
            tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            tmp.replace(path)
            report['report_json'] = str(path)
        return deepcopy(report)

    def report_path(self, campaign_id: str, *, round_id: str, metric: str) -> Path:
        return (
            self.runtime_root / 'v020' / 'evaluations' / campaign_id / round_id /
            f'prediction_vs_measurement_{_safe_metric_filename(metric)}.json'
        )
