from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG, load_config
from .run import evaluate_sample, summarize_run
from .references import validate_references


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="整段保持与提示遵循：提示词加成片，输出分项得分。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--video", type=Path, help="单条成片路径")
    parser.add_argument("--prompt", default="", help="单条成片对应的原文提示词")
    parser.add_argument("--sample-id", default="", help="单条样本标识，默认用成片文件名")
    parser.add_argument("--mode", choices=["t2va", "f2va", "l2va", "fl2va", "r2va"], default="t2va")
    parser.add_argument("--reference-image", type=Path, action="append", default=[], help="按编号顺序提供图片；可重复")
    parser.add_argument("--reference-video", type=Path, action="append", default=[], help="按编号顺序提供参考视频；可重复")
    parser.add_argument("--reference-audio", type=Path, action="append", default=[], help="按编号顺序提供参考音频；可重复")
    parser.add_argument("--manifest", type=Path, help="JSONL，每行含 id、prompt、video")
    parser.add_argument("--output", type=Path, required=True, help="结果目录")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-technical", action="store_true")
    parser.add_argument("--technical-gpu", default="", help="指定技术分 GPU。默认在 4 号和 5 号中选空闲显存更多的一张")
    parser.add_argument("--allow-shared-gpu", action="store_true", help="指定 4、5 以外的卡时，允许使用已有进程的 GPU")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.technical_gpu:
        config.technical.gpu = args.technical_gpu
    samples = _samples(args)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for sample in samples:
        record = evaluate_sample(
            sample_id=sample["id"],
            prompt=sample["prompt"],
            video=Path(sample["video"]),
            output_dir=args.output,
            config=config,
            run_judge=not args.skip_judge,
            run_technical=not args.skip_technical,
            allow_shared_gpu=args.allow_shared_gpu,
            mode=sample.get("mode", "t2va"),
            references=sample.get("references", []),
        )
        records.append(record)
        print(f"{record['sample']['id']}\t{record['status']}")
        for reason in record.get("reasons") or []:
            print(f"  {reason}", file=sys.stderr)
    summarize_run(records, args.output)
    incomplete = any(record["status"] != "complete" for record in records)
    return 1 if incomplete else 0


def _samples(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.manifest and args.video:
        raise SystemExit("manifest 和单条 video 只使用其中一个")
    if args.manifest:
        rows = []
        seen = set()
        for line_number, line in enumerate(args.manifest.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not {"id", "prompt", "video"} <= set(item):
                raise SystemExit(f"{args.manifest}:{line_number} 需要 id、prompt、video")
            if str(item["id"]) in seen:
                raise SystemExit(f"{args.manifest}:{line_number} 样本 id 重复：{item['id']}")
            seen.add(str(item["id"]))
            mode = item.get("mode", "t2va")
            refs = [dict(ref) for ref in item.get("references", [])]
            base = args.manifest.resolve().parent
            for ref in refs:
                ref["path"] = str((base / ref["path"]).resolve())
            refs = validate_references(mode, refs, str(item["prompt"]))
            rows.append({"id": str(item["id"]), "prompt": str(item["prompt"]), "video": str((base / str(item["video"])).resolve()), "mode": mode, "references": refs})
        if not rows:
            raise SystemExit("manifest 是空的")
        return rows
    if args.video is None:
        raise SystemExit("需要 --video 和 --prompt，或 --manifest")
    if not str(args.prompt).strip():
        raise SystemExit("单条评估需要 --prompt")
    sample_id = args.sample_id or args.video.stem
    refs = [{"type": kind, "index": index, "path": str(path)}
            for kind, paths in (("image", args.reference_image), ("video", args.reference_video), ("audio", args.reference_audio))
            for index, path in enumerate(paths, 1)]
    refs = validate_references(args.mode, refs, args.prompt)
    return [{"id": sample_id, "prompt": args.prompt, "video": str(args.video), "mode": args.mode, "references": refs}]


if __name__ == "__main__":
    raise SystemExit(main())
