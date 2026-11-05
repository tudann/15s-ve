from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .ffmpeg import ffmpeg_executable


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration_seconds: float
    fps: float
    width: int
    height: int

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "path": str(self.path),
            "duration_seconds": round(self.duration_seconds, 3),
            "fps": round(self.fps, 3),
            "width": self.width,
            "height": self.height,
        }


def probe_video(path: Path) -> MediaInfo:
    if not path.is_file():
        raise FileNotFoundError(f"成片不存在：{path}")
    completed = subprocess.run(
        [ffmpeg_executable(), "-hide_banner", "-i", str(path)],
        capture_output=True, text=True, check=False, timeout=30,
    )
    detail = f"{completed.stderr or ''}\n{completed.stdout or ''}"
    duration = _duration_seconds(detail)
    fps = _fps(detail)
    size = _size(detail)
    if duration is None or fps is None or size is None:
        raise RuntimeError(f"成片缺少时长、帧率或分辨率：{path}")
    width, height = size
    return MediaInfo(path=path, duration_seconds=duration, fps=fps, width=width, height=height)


def _positive_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def _duration_seconds(detail: str) -> float | None:
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", detail)
    if match is None:
        return None
    hours, minutes, seconds = (float(part) for part in match.groups())
    duration = hours * 3600 + minutes * 60 + seconds
    return duration if duration > 0 else None


def _fps(detail: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s+fps", detail)
    if match is None:
        return None
    return _positive_float(match.group(1))


def _size(detail: str) -> tuple[int, int] | None:
    match = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", detail)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))
