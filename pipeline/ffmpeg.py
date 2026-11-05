from __future__ import annotations

import shutil


def ffmpeg_executable() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("未找到 ffmpeg，也未安装 imageio-ffmpeg。") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()
