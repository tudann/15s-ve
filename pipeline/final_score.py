"""Combine the technical block and the judge block onto one 1–5 score."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from .config import FinalScoreConfig

HOLD_DIMENSIONS = ("subject_consistency", "background_consistency")
MEAN_DIMENSIONS = ("imaging_quality", "motion_smoothness", "aesthetic_quality")
EXCLUDED_DIMENSIONS = ("dynamic_degree",)


def attach_final_score(record: dict[str, Any], config: FinalScoreConfig) -> dict[str, Any]:
    """Write record['final'] when both blocks exist. Drop it otherwise."""
    scored = final_score(record, config)
    if scored is None:
        record.pop("final", None)
    else:
        record["final"] = scored
    return record


def final_score(record: dict[str, Any], config: FinalScoreConfig) -> dict[str, Any] | None:
    technical = record.get("technical") or {}
    judge = record.get("judge") or {}
    dimensions = technical.get("dimensions") or {}
    scores = judge.get("scores") or {}
    if technical.get("status") != "ok" or judge.get("status") != "ok":
        return None
    if not _has_dimensions(dimensions, config) or not _has_judge(scores, config):
        return None

    technical_components = {}
    for name, weight in config.technical_weights.items():
        unit = _technical_unit(name, dimensions[name])
        technical_components[name] = {
            "unit": round(unit, 6),
            "score": _round2(_to_five(unit)),
            "weight": weight,
        }
    judge_components = {}
    for name, weight in config.judge_weights.items():
        value = float(scores[name])
        judge_components[name] = {"score": _round2(value), "weight": weight}

    technical_score = _round2(_weighted(technical_components))
    judge_score = _round2(_weighted(judge_components))
    score = _round2(
        config.technical_weight * technical_score + config.judge_weight * judge_score
    )
    return {
        "scale": [1, 5],
        "score": score,
        "technical_weight": config.technical_weight,
        "judge_weight": config.judge_weight,
        "technical_score": technical_score,
        "judge_score": judge_score,
        "technical_components": technical_components,
        "judge_components": judge_components,
        "excluded": list(EXCLUDED_DIMENSIONS),
    }


def _has_dimensions(dimensions: dict[str, Any], config: FinalScoreConfig) -> bool:
    for name in config.technical_weights:
        item = dimensions.get(name)
        if not isinstance(item, dict) or "mean" not in item:
            return False
        if name in HOLD_DIMENSIONS and "worst_window" not in item:
            return False
    return True


def _has_judge(scores: dict[str, Any], config: FinalScoreConfig) -> bool:
    return all(name in scores for name in config.judge_weights)


def _technical_unit(name: str, dimension: dict[str, Any]) -> float:
    if name in HOLD_DIMENSIONS:
        mean = float(dimension["mean"])
        worst = float(dimension["worst_window"]["score"])
        gap = max(0.0, float(dimension.get("first_last_gap") or 0))
        return _clamp(0.5 * mean + 0.5 * worst - 0.5 * gap)
    return _clamp(float(dimension["mean"]))


def _weighted(components: dict[str, dict[str, Any]]) -> float:
    return sum(float(item["score"]) * float(item["weight"]) for item in components.values())


def _to_five(unit: float) -> float:
    return 1 + 4 * _clamp(unit)


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def _round2(value: float) -> float:
    return float(Decimal(str(round(value, 6))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
