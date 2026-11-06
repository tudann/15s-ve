#!/usr/bin/env python3
"""Score the cases listed in configs/eval_run.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.cases import discover_cases, load_eval_run, samples_for_video
from pipeline.config import load_config, config_as_dict
from pipeline.references import fingerprint
from pipeline.media import MediaInfo
from pipeline.run import (
    _safe_id,
    _write,
    apply_technical,
    evaluate_sample,
    has_judge_scores,
    has_technical_scores,
    read_existing_result,
    summarize_run,
)
from pipeline.technical import score_technical_batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按 configs/eval_run.yaml 跑整段保持与提示遵循。")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "eval_run.yaml")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要评分的样本，不打分")
    parser.add_argument("--force", action="store_true", help="重新评估并覆盖已有结果")
    args = parser.parse_args(argv)

    spec = load_eval_run(args.config)
    spec.force = spec.force or args.force
    case_dirs, missing_ids = discover_cases(spec.cases_root, spec.input_modes, spec.case_ids)
    print(f"prompt: {spec.prompt_filename}")
    print(f"cases_root: {spec.cases_root}")
    print(f"modes: {', '.join(spec.input_modes)}")
    print(f"videos: {', '.join(spec.video_filenames)}")
    if spec.case_ids:
        print(f"case_ids: {', '.join(spec.case_ids)}")
    if missing_ids:
        print(f"missing: {', '.join(missing_ids)}", file=sys.stderr)
    if not case_dirs:
        raise SystemExit("没有找到可评分的 case")

    config = None if args.dry_run else load_config(spec.pipeline_config)
    if config is not None and spec.technical_gpu:
        config.technical.gpu = spec.technical_gpu

    planned = 0
    has_skipped = bool(missing_ids)
    groups: list[tuple[Path, list[dict]]] = []
    jobs: list[dict] = []
    pending: list[tuple[dict, Path, str]] = []
    for video_name in spec.video_filenames:
        samples, skipped = samples_for_video(case_dirs, spec.prompt_filename, video_name)
        output_dir = spec.output_dir / _safe_stem(video_name)
        print(f"video: {video_name}")
        print(f"output: {output_dir}")
        for reason in skipped:
            print(f"  skip {reason}", file=sys.stderr)
        has_skipped = has_skipped or bool(skipped)
        if not samples:
            print(f"  没有同时具备 {spec.prompt_filename} 和 {video_name} 的 case")
            continue
        print(f"  {len(samples)} cases: {', '.join(sample['mode'] + '/' + sample['id'] for sample in samples)}")
        planned += len(samples)
        if args.dry_run:
            continue
        records = _prepare(spec, config, samples, output_dir, _safe_stem(video_name), jobs, pending)
        groups.append((output_dir, records))

    if planned == 0:
        raise SystemExit("没有可评分的成片")
    if args.dry_run:
        return 1 if has_skipped else 0
    if jobs:
        print(f"技术分：{len(jobs)} 条共用一次加载")
        scored = score_technical_batch(jobs, spec.output_dir / "_batch", config.technical)
        for record, output_dir, batch_id in pending:
            apply_technical(record, scored[batch_id], output_dir, config)
            print(f"{output_dir.name}/{record['sample']['id']}\t{record['status']}")
            for reason in record.get("reasons") or []:
                print(f"  {reason}", file=sys.stderr)
    exit_code = 1 if has_skipped else 0
    for output_dir, records in groups:
        summarize_run(records, output_dir)
        if any(record.get("status") != "complete" for record in records):
            exit_code = 1
    return exit_code


def _prepare(spec, config, samples, output_dir: Path, video_key: str, jobs: list[dict], pending: list) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    id_counts = {sample["id"]: sum(other["id"] == sample["id"] for other in samples) for sample in samples}
    for sample in samples:
        # Preserve old t2va paths; namespace other modes to avoid collisions.
        sample_id = sample["id"] if sample["mode"] == "t2va" and id_counts[sample["id"]] == 1 else f"{sample['task']}__{sample['mode']}__{sample['id']}"
        signature = fingerprint(sample, config_as_dict(config))
        result_path = output_dir / _safe_id(sample_id) / "result.json"
        existing = read_existing_result(result_path) if not spec.force else None
        if existing is not None and not _same_sample(existing, sample_id, sample):
            records.append(existing)
            print(f"{sample['id']}\t{existing.get('status', 'unknown')}\tinput changed; preserved; use --force to re-evaluate")
            continue
        if existing is not None and has_judge_scores(existing) and has_technical_scores(existing):
            records.append(existing)
            print(f"{sample['id']}\tcomplete\tcached")
            continue
        if existing is None and result_path.exists() and not spec.force:
            record = {
                "dimension": config.dimension,
                "status": "incomplete",
                "reasons": ["已有结果文件无法解析，已保留原文件；使用 --force 才会覆盖"],
                "sample": {"id": sample_id, "prompt": sample["prompt"], "video": sample["video"], "mode": sample["mode"]},
            }
            records.append(record)
            print(f"{sample['id']}\tincomplete\tpreserved unreadable result; use --force to re-evaluate")
            continue
        record = evaluate_sample(
            sample_id=sample_id,
            prompt=sample["prompt"],
            video=Path(sample["video"]),
            output_dir=output_dir,
            config=config,
            run_judge=not spec.skip_judge and (existing is None or not has_judge_scores(existing)),
            run_technical=False,
            defer_technical=not spec.skip_technical and (existing is None or not has_technical_scores(existing)),
            mode=sample["mode"],
            references=sample["references"],
            persist_result=False,
            existing_record=existing,
        )
        record["input_fingerprint"] = signature
        records.append(record)
        if spec.skip_technical or has_technical_scores(record) or not record.get("media"):
            _write(result_path, record)
            print(f"{record['sample']['id']}\t{record['status']}")
            for reason in record.get("reasons") or []:
                print(f"  {reason}", file=sys.stderr)
            continue
        batch_id = f"{video_key}__{record['sample']['id']}"
        media = record["media"]
        jobs.append({
            "id": batch_id,
            "media": MediaInfo(
                path=Path(media["path"]),
                duration_seconds=float(media["duration_seconds"]),
                fps=float(media["fps"]),
                width=int(media["width"]),
                height=int(media["height"]),
            ),
            "work_dir": output_dir / _safe_id(record["sample"]["id"]),
        })
        pending.append((record, output_dir, batch_id))
    return records


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem.strip()
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
    if not cleaned:
        raise SystemExit(f"无法从文件名得到输出目录：{filename}")
    return cleaned


def _same_sample(record: dict, sample_id: str, sample: dict) -> bool:
    old = record.get("sample") or {}
    expected = {
        "id": sample_id,
        "prompt": sample["prompt"],
        "video": sample["video"],
        "mode": sample["mode"],
        "references": sample["references"],
    }
    return all(old.get(key, [] if key == "references" else None) == value for key, value in expected.items())


if __name__ == "__main__":
    raise SystemExit(main())
