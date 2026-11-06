from __future__ import annotations

import json
import pickle
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .final_score import attach_final_score
from .judge import judge_video
from .media import probe_video
from .technical import TechnicalUnavailable, score_technical


def evaluate_sample(
    *,
    sample_id: str,
    prompt: str,
    video: Path,
    output_dir: Path,
    config: PipelineConfig,
    run_judge: bool = True,
    run_technical: bool = True,
    defer_technical: bool = False,
    allow_shared_gpu: bool = False,
    mode: str = "t2va",
    references: list[dict] | None = None,
    persist_result: bool = True,
    existing_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample_dir = output_dir / _safe_id(sample_id)
    sample_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = deepcopy(existing_record) if existing_record is not None else {
        "dimension": config.dimension,
        "status": "incomplete",
        "reasons": [],
        "sample": {"id": sample_id, "prompt": prompt, "video": str(video), "mode": mode, "references": references or []},
    }
    record["dimension"] = config.dimension
    record["sample"] = {"id": sample_id, "prompt": prompt, "video": str(video), "mode": mode, "references": references or []}
    record.setdefault("reasons", [])
    try:
        media = probe_video(video)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        record["reasons"].append(str(exc))
        if persist_result:
            _write(sample_dir / "result.json", record)
        return record
    record["media"] = media.as_dict()

    if run_judge:
        try:
            record["judge"] = judge_video(media, prompt, config.judge, sample_dir, mode=mode, references=references)
            record["reasons"] = [reason for reason in record["reasons"] if not str(reason).startswith("裁判：")]
        except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
            record["judge"] = {"status": "error"}
            record["reasons"].append(f"裁判：{exc}")
    elif not has_judge_scores(record):
        record["reasons"].append("未运行裁判")

    if run_technical:
        try:
            record["technical"] = score_technical(
                media, sample_dir, config.technical, allow_shared_gpu=allow_shared_gpu,
            )
            record["reasons"] = [reason for reason in record["reasons"] if not str(reason).startswith("技术分：")]
        except TechnicalUnavailable as exc:
            record["technical"] = {"status": "unavailable"}
            record["reasons"].append(f"技术分：{exc}")
        except (RuntimeError, ValueError, OSError, pickle.UnpicklingError) as exc:
            record["technical"] = {"status": "error"}
            record["reasons"].append(f"技术分：{exc}")
    elif not defer_technical and not has_technical_scores(record):
        record["reasons"].append("未运行技术分")

    record["status"] = "complete" if _is_complete(record) else "incomplete"
    attach_final_score(record, config.final_score)
    if persist_result:
        _write(sample_dir / "result.json", record)
    return record


def apply_technical(record: dict[str, Any], technical: dict[str, Any], output_dir: Path, config: PipelineConfig | None = None) -> dict[str, Any]:
    """Attach a batched technical result and rewrite the sample record."""
    sample_id = str(record["sample"]["id"])
    sample_dir = output_dir / _safe_id(sample_id)
    status = technical.get("status")
    if status == "ok":
        record["technical"] = technical
        record["reasons"] = [reason for reason in record.get("reasons", []) if not str(reason).startswith("技术分：")]
    elif status == "unavailable":
        record["technical"] = {"status": "unavailable"}
        record["reasons"].append(f"技术分：{technical.get('message', '')}")
    else:
        record["technical"] = {"status": "error"}
        record["reasons"].append(f"技术分：{technical.get('message', '')}")
    record["status"] = "complete" if _is_complete(record) else "incomplete"
    if config is not None:
        attach_final_score(record, config.final_score)
    _write(sample_dir / "result.json", record)
    return record


def summarize_run(records: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    modes = {(record.get("sample") or {}).get("mode", "t2va") for record in records}
    for record in records:
        if record.get("status") != "complete":
            continue
        duration = float((record.get("media") or {}).get("duration_seconds") or 0)
        key = f"{round(duration)}s"
        if len(modes) > 1 or modes != {"t2va"}:
            key = f"{(record.get('sample') or {}).get('mode', 't2va')}/{key}"
        groups.setdefault(key, []).append(record)
    summary = {
        "dimension": records[0]["dimension"] if records else "",
        "samples": len(records),
        "complete": sum(1 for record in records if record.get("status") == "complete"),
        "groups": {key: _group_means(rows) for key, rows in groups.items()},
    }
    _write(output_dir / "summary.json", summary)
    return summary


def read_existing_result(path: Path) -> dict[str, Any] | None:
    """Load an existing case result without changing it."""
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return record if isinstance(record, dict) else None


def _is_complete(record: dict[str, Any]) -> bool:
    return has_judge_scores(record) and has_technical_scores(record)


def has_judge_scores(record: dict[str, Any]) -> bool:
    judge = record.get("judge") or {}
    scores = judge.get("scores") or {}
    return judge.get("status") == "ok" and set(scores) == {
        "visual_quality", "text_video_alignment", "physical_consistency",
    }


def has_technical_scores(record: dict[str, Any]) -> bool:
    technical = record.get("technical") or {}
    dimensions = technical.get("dimensions") or {}
    return technical.get("status") == "ok" and len(dimensions) == 6


def _group_means(records: list[dict[str, Any]]) -> dict[str, Any]:
    technical: dict[str, float] = {}
    names = list((records[0].get("technical") or {}).get("dimensions") or {})
    for name in names:
        values = [float(record["technical"]["dimensions"][name]["mean"]) for record in records]
        technical[name] = round(sum(values) / len(values), 6)
    judge_means: dict[str, float] = {}
    for name in ("visual_quality", "text_video_alignment", "physical_consistency"):
        values = [float(record["judge"]["scores"][name]) for record in records]
        judge_means[name] = round(sum(values) / len(values), 6)
    group: dict[str, Any] = {"count": len(records), "technical_mean": technical, "judge_mean": judge_means}
    finals = [float(record["final"]["score"]) for record in records if (record.get("final") or {}).get("score") is not None]
    if len(finals) == len(records):
        group["final_mean"] = round(sum(finals) / len(finals), 2)
    return group


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_id(sample_id: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in sample_id.strip())
    return cleaned or "sample"
