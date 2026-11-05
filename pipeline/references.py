"""Discover and validate generation inputs without reading metadata."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

MODES = {"t2va", "f2va", "l2va", "fl2va", "r2va"}
EXTENSIONS = {"image": {".png", ".jpg", ".jpeg", ".webp"},
              "video": {".mp4", ".mov", ".mkv", ".webm", ".avi"},
              "audio": {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}}


def discover_references(directory: Path, mode: str, prompt: str) -> list[dict]:
    if mode not in MODES:
        raise ValueError(f"不支持的输入模式：{mode}")
    refs = []
    patterns = {"image": r"input_(\d+)", "video": r"ref_video_(\d+)", "audio": r"ref_audio_(\d+)"}
    if mode != "t2va":
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            for kind, pattern in patterns.items():
                match = re.fullmatch(pattern, path.stem, re.IGNORECASE)
                if match and path.suffix.lower() in EXTENSIONS[kind]:
                    refs.append({"type": kind, "index": int(match.group(1)), "path": str(path)})
    return validate_references(mode, refs, prompt)


def validate_references(mode: str, refs: list[dict], prompt: str) -> list[dict]:
    if mode not in MODES:
        raise ValueError(f"不支持的输入模式：{mode}")
    keys = set()
    normalized = []
    for ref in refs:
        kind = ref.get("type")
        try:
            index = int(ref.get("index", 0))
        except (TypeError, ValueError):
            raise ValueError(f"无效素材编号：{ref}") from None
        if kind not in EXTENSIONS or index < 1:
            raise ValueError(f"无效素材：{ref}")
        path = Path(ref.get("path", ""))
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS[kind]:
            raise ValueError(f"素材不存在或格式不支持：{path}")
        if (kind, index) in keys:
            raise ValueError(f"素材编号重复：{kind} {index}")
        keys.add((kind, index))
        normalized.append({**ref, "type": kind, "index": index, "path": str(path)})
    required = {"f2va": [1], "l2va": [1], "fl2va": [1, 2]}.get(mode, [])
    if any(("image", i) not in keys for i in required):
        raise ValueError(f"{mode} 缺少必需的端点图片")
    if required and keys != {("image", i) for i in required}:
        raise ValueError(f"{mode} 仅接受模式规定的端点图片；其他参考素材应放入 r2va")
    if mode == "r2va" and not refs:
        raise ValueError("r2va 缺少参考素材")
    if mode == "t2va" and refs:
        raise ValueError("t2va 不接受参考素材")
    aliases = {"image": r"(?:图(?:片)?|(?<![a-z])(?:picture|image|p))\s*(\d+|[一二三四五六七八九十])",
               "video": r"(?:参考视频|reference\s+video)\s*(\d+)",
               "audio": r"(?:参考音频|reference\s+audio)\s*(\d+)"}
    for kind, pattern in aliases.items():
        for number in re.findall(pattern, prompt, re.IGNORECASE):
            index = int(number) if number.isdigit() else "一二三四五六七八九十".index(number) + 1
            if (kind, index) not in keys:
                raise ValueError(f"提示词引用的 {kind} {number} 不存在")
    return sorted(normalized, key=lambda ref: (ref["type"], ref["index"]))


def fingerprint(sample: dict, config: dict) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({"sample": sample, "config": config, "version": 2}, ensure_ascii=False, sort_keys=True).encode())
    for name in [sample["video"], *(r["path"] for r in sample.get("references", []))]:
        with Path(name).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
