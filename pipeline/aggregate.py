from __future__ import annotations

from pathlib import Path
from typing import Any


def summarize_windows(windows: list[dict[str, Any]]) -> dict[str, Any]:
    """Higher scores are better. Gap is the first window minus the last window."""
    if not windows:
        raise ValueError("没有窗口分")
    ordered = sorted(windows, key=lambda item: int(item["index"]))
    scores = [float(item["score"]) for item in ordered]
    worst = min(ordered, key=lambda item: float(item["score"]))
    return {
        "mean": round(sum(scores) / len(scores), 6),
        "worst_window": {
            "index": int(worst["index"]),
            "start": float(worst["start"]),
            "score": round(float(worst["score"]), 6),
        },
        "first_last_gap": round(scores[0] - scores[-1], 6),
        "windows": [
            {
                "index": int(item["index"]),
                "start": float(item["start"]),
                "score": round(float(item["score"]), 6),
            }
            for item in ordered
        ],
    }


def scores_by_window(dimension_result: Any, clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Read one VBench dimension result and align it to cut windows."""
    per_video = _per_video_scores(dimension_result)
    by_name = {_match_key(path): score for path, score in per_video.items()}
    aligned: list[dict[str, Any]] = []
    missing: list[str] = []
    for clip in clips:
        name = Path(str(clip["path"])).name
        score = by_name.get(name)
        if score is None:
            score = by_name.get(Path(str(clip["path"])).stem)
        if score is None:
            missing.append(name)
            continue
        aligned.append({"index": clip["index"], "start": clip["start"], "score": score})
    if missing:
        raise ValueError(f"这些窗口没有技术分：{missing}")
    return aligned


def _per_video_scores(dimension_result: Any) -> dict[str, float]:
    rows = _video_rows(dimension_result)
    scores: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict) or "video_path" not in row:
            continue
        number = _as_score(row.get("video_results"))
        if number is None:
            continue
        scores[str(row["video_path"])] = number
    if not scores:
        raise ValueError(f"无法从 VBench 结果里读出逐段分：{type(dimension_result).__name__}")
    return scores


def _video_rows(dimension_result: Any) -> list[Any]:
    if isinstance(dimension_result, (list, tuple)) and len(dimension_result) == 2 and isinstance(dimension_result[1], list):
        return list(dimension_result[1])
    if isinstance(dimension_result, list):
        return list(dimension_result)
    raise ValueError("VBench 维度结果不是逐视频列表")


def _as_score(value: Any) -> float | None:
    # dynamic_degree is true/false per window. true means the window has motion.
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        numbers = [float(item) for item in value if isinstance(item, (int, float)) and not isinstance(item, bool)]
        if numbers:
            return sum(numbers) / len(numbers)
    return None


def _match_key(path: str) -> str:
    name = Path(path).name
    return name or path
