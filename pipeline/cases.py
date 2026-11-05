"""Resolve evaluation cases from configs/eval_run.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import PACKAGE_ROOT
from .references import MODES, discover_references


@dataclass
class EvalRun:
    prompt_filename: str
    cases_root: Path
    input_modes: list[str]
    video_filenames: list[str]
    case_ids: list[str]
    output_dir: Path
    pipeline_config: Path
    force: bool
    skip_judge: bool
    skip_technical: bool
    technical_gpu: str


def load_eval_run(path: Path, *, root: Path = PACKAGE_ROOT) -> EvalRun:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"评测配置不是映射：{path}")
    prompt_filename = str(raw.get("prompt_filename") or "").strip()
    if not prompt_filename:
        raise ValueError("评测配置需要 prompt_filename")
    videos = _names(raw.get("video_filenames"), "video_filenames")
    modes = _names(raw.get("input_modes") or ["t2va"], "input_modes")
    if set(modes) - MODES:
        raise ValueError(f"不支持的输入模式：{set(modes) - MODES}")
    case_ids = [str(item).strip() for item in (raw.get("case_ids") or []) if str(item).strip()]
    if not str(raw.get("cases_root") or "").strip():
        raise ValueError("评测配置需要 cases_root")
    cases_root = _resolve(str(raw.get("cases_root")).strip(), root)
    output_value = str(raw.get("output_dir") or "./outputs/eval").strip()
    pipeline_value = str(raw.get("pipeline_config") or "./configs/default.yaml").strip()
    return EvalRun(
        prompt_filename=prompt_filename,
        cases_root=cases_root,
        input_modes=modes,
        video_filenames=videos,
        case_ids=case_ids,
        output_dir=_resolve(output_value, root),
        pipeline_config=_resolve(pipeline_value, root),
        force=bool(raw.get("force", False)),
        skip_judge=bool(raw.get("skip_judge", False)),
        skip_technical=bool(raw.get("skip_technical", False)),
        technical_gpu=str(raw.get("technical_gpu") or "").strip(),
    )


def discover_cases(cases_root: Path, modes: list[str], case_ids: list[str]) -> tuple[list[Path], list[str]]:
    """Return case directories in requested order, plus ids that were not found."""
    by_id: dict[tuple[str, str, str], Path] = {}
    for mode in modes:
        if not cases_root.is_dir():
            break
        candidates = set(cases_root.glob(f"*/{mode}/*")) | set(cases_root.glob(f"{mode}/*"))
        for case_dir in sorted(candidates):
            if case_dir.is_dir():
                by_id[(case_dir.parent.parent.name, mode, case_dir.name)] = case_dir
    if not case_ids:
        return [by_id[key] for key in sorted(by_id)], []
    found = [path for case_id in case_ids for key, path in by_id.items() if key[2] == case_id]
    missing = [case_id for case_id in case_ids if not any(key[2] == case_id for key in by_id)]
    return found, missing


def samples_for_video(
    case_dirs: list[Path],
    prompt_filename: str,
    video_filename: str,
) -> tuple[list[dict[str, str]], list[str]]:
    samples: list[dict[str, str]] = []
    skipped: list[str] = []
    for case_dir in case_dirs:
        prompt_path = case_dir / prompt_filename
        video_path = case_dir / video_filename
        if not prompt_path.is_file():
            skipped.append(f"{case_dir.name} 缺少 {prompt_filename}")
            continue
        if not video_path.is_file():
            skipped.append(f"{case_dir.name} 缺少 {video_filename}")
            continue
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not prompt:
            skipped.append(f"{case_dir.name} 的 {prompt_filename} 是空的")
            continue
        mode = case_dir.parent.name
        try:
            references = discover_references(case_dir, mode, prompt)
        except ValueError as exc:
            skipped.append(f"{mode}/{case_dir.name}: {exc}")
            continue
        samples.append({
            "id": case_dir.name,
            "mode": mode,
            "task": case_dir.parent.parent.name,
            "references": references,
            "prompt": prompt,
            "video": str(video_path),
        })
    return samples, skipped


def _names(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} 需要是列表")
    names = [str(item).strip() for item in value if str(item).strip()]
    if not names:
        raise ValueError(f"{field} 至少要有一项")
    return names


def _resolve(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()
