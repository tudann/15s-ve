from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "default.yaml"

TECHNICAL_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
)

# 技术分只在这两张卡里默认选择，不默认占用 0 号。
DEFAULT_TECHNICAL_GPUS = ("4", "5")


@dataclass
class JudgeConfig:
    protocol: str
    protocol_model: str
    base_url: str
    model: str
    api_key: str
    temperature: float
    timeout_seconds: int
    sample_fps: float
    max_frames: int
    frame_width: int
    jpeg_quality: int
    reference_max_frames: int = 8
    max_context_images: int = 64
    audio_input_format: str = "disabled"
    audio_max_seconds: float = 60


@dataclass
class TechnicalConfig:
    window_seconds: float
    min_tail_seconds: float
    gpu: str
    gpus: list[str]
    device: str
    dimensions: list[str] = field(default_factory=lambda: list(TECHNICAL_DIMENSIONS))


@dataclass
class FinalScoreConfig:
    technical_weight: float
    judge_weight: float
    technical_weights: dict[str, float]
    judge_weights: dict[str, float]


DEFAULT_TECHNICAL_SCORE_WEIGHTS = {
    "subject_consistency": 0.40,
    "background_consistency": 0.25,
    "imaging_quality": 0.15,
    "motion_smoothness": 0.10,
    "aesthetic_quality": 0.10,
}
DEFAULT_JUDGE_SCORE_WEIGHTS = {
    "text_video_alignment": 0.50,
    "visual_quality": 0.25,
    "physical_consistency": 0.25,
}


@dataclass
class PipelineConfig:
    dimension: str
    judge: JudgeConfig
    technical: TechnicalConfig
    final_score: FinalScoreConfig


def load_config(path: Path | None = None) -> PipelineConfig:
    config_path = path or DEFAULT_CONFIG
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"配置不是映射：{config_path}")
    judge = raw.get("judge") or {}
    technical = raw.get("technical") or {}
    dimensions = list(technical.get("dimensions") or TECHNICAL_DIMENSIONS)
    unknown = [name for name in dimensions if name not in TECHNICAL_DIMENSIONS]
    if unknown:
        raise ValueError(f"技术维不在第一期范围内：{unknown}")
    gpus = [str(gpu) for gpu in (technical.get("gpus") or DEFAULT_TECHNICAL_GPUS)]
    if not gpus:
        raise ValueError("技术分至少要有一张默认显卡")
    final_score = _final_score_config(raw.get("final_score") or {})
    if judge.get("audio_input_format", "disabled") not in {"disabled", "input_audio", "audio_url"}:
        raise ValueError("audio_input_format 只支持 disabled、input_audio 或 audio_url")
    if int(judge.get("reference_max_frames", 8)) < 1 or int(judge.get("max_context_images", 64)) < 1 or float(judge.get("audio_max_seconds", 60)) <= 0:
        raise ValueError("裁判素材限制必须为正数")
    return PipelineConfig(
        dimension=str(raw.get("dimension") or "21. 整段保持与提示遵循"),
        judge=JudgeConfig(
            protocol=str(judge.get("protocol") or "VideoScore2"),
            protocol_model=str(judge.get("protocol_model") or "TIGER-Lab/VideoScore2"),
            base_url=str(judge.get("base_url") or "http://127.0.0.1:8011/v1").rstrip("/"),
            model=str(judge.get("model") or "Qwen3-VL-32B-Instruct"),
            api_key=str(judge.get("api_key") or ""),
            temperature=float(judge.get("temperature", 0)),
            timeout_seconds=int(judge.get("timeout_seconds", 180)),
            sample_fps=float(judge.get("sample_fps", 2)),
            max_frames=int(judge.get("max_frames", 32)),
            frame_width=int(judge.get("frame_width", 640)),
            jpeg_quality=int(judge.get("jpeg_quality", 85)),
            reference_max_frames=int(judge.get("reference_max_frames", 8)),
            max_context_images=int(judge.get("max_context_images", 64)),
            audio_input_format=str(judge.get("audio_input_format", "disabled")),
            audio_max_seconds=float(judge.get("audio_max_seconds", 60)),
        ),
        technical=TechnicalConfig(
            window_seconds=float(technical.get("window_seconds", 2)),
            min_tail_seconds=float(technical.get("min_tail_seconds", 1)),
            gpu=str(technical.get("gpu", "auto")),
            gpus=gpus,
            device=str(technical.get("device", "cuda")),
            dimensions=dimensions,
        ),
        final_score=final_score,
    )


def _final_score_config(raw: dict[str, Any]) -> FinalScoreConfig:
    technical_weight = float(raw.get("technical_weight", 0.5))
    if technical_weight < 0 or technical_weight > 1:
        raise ValueError("technical_weight 必须在 0 到 1 之间")
    technical_weights = _weights(
        raw.get("technical") or DEFAULT_TECHNICAL_SCORE_WEIGHTS,
        set(DEFAULT_TECHNICAL_SCORE_WEIGHTS),
        "技术块",
    )
    judge_weights = _weights(
        raw.get("judge") or DEFAULT_JUDGE_SCORE_WEIGHTS,
        set(DEFAULT_JUDGE_SCORE_WEIGHTS),
        "裁判块",
    )
    return FinalScoreConfig(
        technical_weight=technical_weight,
        judge_weight=round(1 - technical_weight, 6),
        technical_weights=technical_weights,
        judge_weights=judge_weights,
    )


def _weights(raw: Any, allowed: set[str], label: str) -> dict[str, float]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{label}权重不能为空")
    unknown = [name for name in raw if name not in allowed]
    if unknown:
        raise ValueError(f"{label}权重含未接入的项：{unknown}")
    weights = {str(name): float(value) for name, value in raw.items()}
    if any(value < 0 for value in weights.values()):
        raise ValueError(f"{label}权重不能为负")
    total = sum(weights.values())
    if abs(total - 1) > 1e-6:
        raise ValueError(f"{label}权重之和必须为 1，现在是 {total}")
    return weights


def config_as_dict(config: PipelineConfig) -> dict[str, Any]:
    return {
        "dimension": config.dimension,
        "judge": config.judge.__dict__,
        "technical": config.technical.__dict__,
        "final_score": {
            "technical_weight": config.final_score.technical_weight,
            "judge_weight": config.final_score.judge_weight,
            "technical": config.final_score.technical_weights,
            "judge": config.final_score.judge_weights,
        },
    }
