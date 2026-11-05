from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .ffmpeg import ffmpeg_executable
from .media import MediaInfo


@dataclass(frozen=True)
class WindowClip:
    index: int
    start: float
    end: float
    path: Path

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "path": str(self.path),
        }


def split_windows(media: MediaInfo, directory: Path, *, window_seconds: float, min_tail_seconds: float) -> tuple[list[WindowClip], float]:
    """Cut the clip into fixed windows. Return clips and the unused tail length."""
    if window_seconds <= 0:
        raise ValueError("窗口长度必须大于 0")
    directory.mkdir(parents=True, exist_ok=True)
    clips: list[WindowClip] = []
    start = 0.0
    index = 0
    while media.duration_seconds - start > 1e-3:
        remaining = media.duration_seconds - start
        if remaining < window_seconds and remaining < min_tail_seconds:
            break
        length = min(window_seconds, remaining)
        target = directory / f"win_{index:03d}.mp4"
        _cut(media.path, target, start, length)
        clips.append(WindowClip(index=index, start=start, end=start + length, path=target))
        start += window_seconds
        index += 1
    if not clips:
        raise RuntimeError(f"成片短于最小窗口，无法切窗：{media.path}")
    unused = max(0.0, media.duration_seconds - clips[-1].end)
    return clips, unused


def _cut(source: Path, target: Path, start: float, length: float) -> None:
    completed = subprocess.run(
        [
            ffmpeg_executable(), "-y", "-v", "error",
            "-i", str(source),
            "-ss", f"{start:.3f}",
            "-t", f"{length:.3f}",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(target),
        ],
        capture_output=True, text=True, check=False, timeout=180,
    )
    if completed.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
        detail = (completed.stderr or "").strip()
        raise RuntimeError(f"切窗失败 {target.name}：{detail}")
