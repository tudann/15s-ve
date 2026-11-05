"""Prepare labelled conditioning media for the existing three judge axes."""
from __future__ import annotations

import base64
import subprocess
from pathlib import Path

from PIL import Image

from .ffmpeg import ffmpeg_executable
from .media import probe_video


def build_context(media, mode, references, config, work_dir):
    from .judge import _image_block, _save_frame, extract_frames
    content = []
    evidence = []
    directory = work_dir / "reference_media"
    directory.mkdir(parents=True, exist_ok=True)
    has_audio_context = False
    for ref in references:
        kind, index, path = ref["type"], ref["index"], Path(ref["path"])
        role = {"f2va": "first-frame input", "l2va": "last-frame input"}.get(mode, "reference")
        if mode == "fl2va":
            role = "first-frame input" if index == 1 else "last-frame input"
        label = f"Condition {kind} {index}: {role}. Not generated-video evidence."
        content.append({"type": "text", "text": label})
        row = {**ref, "role": role}
        if kind == "image":
            target = directory / f"image-{index:02d}.jpg"
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((config.frame_width, config.frame_width))
                image.save(target, quality=config.jpeg_quality)
            content.append(_image_block(target))
        elif kind == "video":
            info = probe_video(path)
            frames = extract_frames(info, directory / f"video-{index:02d}", fps=1,
                                    limit=config.reference_max_frames, width=config.frame_width,
                                    quality=config.jpeg_quality, long_side=True)
            row["frame_timestamps"] = [t for t, _ in frames]
            row["coverage"] = "visual frames"
            for timestamp, frame in frames:
                content.append({"type": "text", "text": f"Reference video {index}, timestamp={timestamp:.3f}s."})
                content.append(_image_block(frame))
            probe = subprocess.run([ffmpeg_executable(), "-hide_banner", "-i", str(path)], capture_output=True, text=True, timeout=30)
            if "Audio:" in probe.stderr:
                content.append({"type": "text", "text": f"Reference video {index} audio, starting at timestamp 0.000s."})
                content.append(audio_block(path, directory / f"video-{index:02d}-audio.wav", config))
                row["coverage"] = "visual frames and audio"
                has_audio_context = True
        else:
            content.append(audio_block(path, directory / f"audio-{index:02d}.wav", config))
            row["coverage"] = "audio"
            has_audio_context = True
        evidence.append(row)
    if has_audio_context:
        content.append({"type": "text", "text": "Generated-video audio, starting at timestamp 0.000s. Use only to interpret prompt context; do not add an audio-quality or audio-reference score."})
        # A missing generated audio track is evidence, not a decoding substitution.
        content.append(audio_block(media.path, directory / "generated-audio.wav", config, allow_no_audio=True))
    if mode in {"f2va", "fl2va"}:
        target = directory / "generated-first.jpg"
        if not _save_frame(media.path, 0, target, config.frame_width, config.jpeg_quality, long_side=True):
            raise RuntimeError("无法读取成片实际首帧")
        content.extend([{"type": "text", "text": "Generated-video actual first decoded frame."}, _image_block(target)])
    if mode in {"l2va", "fl2va"}:
        # Decode in presentation order and retain the actual last frame, rather than seeking duration-epsilon.
        target = directory / "generated-last.jpg"
        result = subprocess.run([ffmpeg_executable(), "-y", "-v", "error", "-i", str(media.path),
                                 "-an", "-vf", f"scale={config.frame_width}:{config.frame_width}:force_original_aspect_ratio=decrease",
                                 "-update", "1", str(target)], capture_output=True, timeout=180)
        if result.returncode or not target.is_file():
            raise RuntimeError("无法读取成片实际尾帧")
        content.extend([{"type": "text", "text": "Generated-video actual last decoded frame."}, _image_block(target)])
    return content, evidence


def audio_block(source: Path, target: Path, config, *, allow_no_audio=False):
    if config.audio_input_format == "disabled":
        raise RuntimeError("参考音频需要支持音频的裁判；当前 audio_input_format=disabled。技术分仍可计算，三项裁判分和最终分不补默认值")
    result = subprocess.run([ffmpeg_executable(), "-y", "-v", "error", "-i", str(source),
                             "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", str(target)],
                            capture_output=True, timeout=180)
    if result.returncode or not target.is_file():
        if allow_no_audio and b"matches no streams" in result.stderr:
            return {"type": "text", "text": "Generated video has no audio track."}
        raise RuntimeError(f"音频解码失败：{source.name}")
    import wave
    with wave.open(str(target), "rb") as stream:
        duration = stream.getnframes() / stream.getframerate()
    if duration > config.audio_max_seconds:
        raise RuntimeError(f"音频长于裁判输入上限 {config.audio_max_seconds}s，不静默截断")
    encoded = base64.b64encode(target.read_bytes()).decode("ascii")
    if config.audio_input_format == "input_audio":
        return {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}}
    return {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{encoded}"}}
