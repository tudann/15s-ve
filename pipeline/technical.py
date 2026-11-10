from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .aggregate import scores_by_window, summarize_windows
from .config import TechnicalConfig
from .media import MediaInfo
from .windows import split_windows


class TechnicalUnavailable(RuntimeError):
    """VBench is not installed, or the selected GPU should not be used."""


VBENCH_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "vbench"
_DINO_WEIGHT_NAME = "dino_vitbase16_pretrain.pth"


def local_vbench_files() -> dict[str, Path]:
    """Weights copied for the six technical dimensions. Callers load only these files."""
    root = VBENCH_MODEL_DIR
    return {
        "clip_b32": root / "clip_model" / "ViT-B-32.pt",
        "clip_l14": root / "clip_model" / "ViT-L-14.pt",
        "amt": root / "amt_model" / "amt-s.pth",
        "raft": root / "raft_model" / "models" / "raft-things.pth",
        "musiq": root / "pyiqa_model" / "musiq_spaq_ckpt-358bb6af.pth",
        "aesthetic": root / "aesthetic_model" / "emb_reader" / "sa_0_4_vit_l_14_linear.pth",
        "dino": root / "dino_model" / _DINO_WEIGHT_NAME,
        "dino_repo": root / "dino_model" / "facebookresearch_dino_main" / "hubconf.py",
    }


def choose_gpu(readings: list[tuple[str, int | None]]) -> str:
    """Pick the readable GPU with the most free memory. Earlier entries win a tie."""
    available = [(gpu, free) for gpu, free in readings if free is not None]
    if not available:
        names = "、".join(gpu for gpu, _ in readings) or "未配置"
        raise TechnicalUnavailable(f"默认显卡 {names} 都读不到空闲显存")
    return max(available, key=lambda item: (item[1], -readings.index((item[0], item[1]))))[0]


def resolve_gpu(config: TechnicalConfig, *, allow_shared_gpu: bool = False) -> str:
    if config.gpu and config.gpu != "auto":
        _ensure_explicit_gpu(config.gpu, shared_by_default=config.gpu in set(config.gpus), allow_shared_gpu=allow_shared_gpu)
        return config.gpu
    return choose_gpu([(gpu, _free_mib(gpu)) for gpu in config.gpus])


def score_technical(media: MediaInfo, work_dir: Path, config: TechnicalConfig, *, allow_shared_gpu: bool = False) -> dict[str, Any]:
    scored = score_technical_batch(
        [{"id": "sample", "media": media, "work_dir": work_dir}],
        work_dir / "vbench_batch",
        config,
        allow_shared_gpu=allow_shared_gpu,
    )
    result = scored["sample"]
    if result.get("status") == "error":
        raise RuntimeError(result.get("message") or "技术分失败")
    if result.get("status") == "unavailable":
        raise TechnicalUnavailable(result.get("message") or "技术分不可用")
    return result


def score_technical_batch(
    jobs: list[dict[str, Any]],
    batch_dir: Path,
    config: TechnicalConfig,
    *,
    allow_shared_gpu: bool = False,
) -> dict[str, dict[str, Any]]:
    """Score every job in one VBench call so each of the six models loads once."""
    results: dict[str, dict[str, Any]] = {}
    ready: list[dict[str, Any]] = []
    staged_dir = batch_dir / "windows"
    staged_dir.mkdir(parents=True, exist_ok=True)
    for old in staged_dir.glob("*.mp4"):
        old.unlink()
    for job in jobs:
        sample_id = str(job["id"])
        try:
            clips, unused_tail = split_windows(
                job["media"],
                Path(job["work_dir"]) / "windows",
                window_seconds=config.window_seconds,
                min_tail_seconds=config.min_tail_seconds,
            )
            staged_rows = []
            for clip in clips:
                staged = staged_dir / f"{_batch_token(sample_id)}__{clip.path.name}"
                _stage_window(clip.path, staged)
                staged_rows.append({"index": clip.index, "start": clip.start, "path": str(staged)})
            ready.append({
                "id": sample_id,
                "clips": [clip.as_dict() for clip in clips],
                "unused_tail": unused_tail,
                "staged": staged_rows,
            })
        except (RuntimeError, OSError, ValueError) as exc:
            results[sample_id] = {"status": "error", "message": str(exc)}
    if not ready:
        return results
    try:
        gpu = resolve_gpu(config, allow_shared_gpu=allow_shared_gpu)
        results_path = _run_vbench(staged_dir, batch_dir / "vbench", config, gpu)
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        parsed = [(item, _dimensions_from_payload(payload, config, item["staged"])) for item in ready]
    except TechnicalUnavailable as exc:
        for item in ready:
            results[item["id"]] = {"status": "unavailable", "message": str(exc)}
        return results
    except (RuntimeError, OSError, ValueError, pickle.UnpicklingError) as exc:
        for item in ready:
            results[item["id"]] = {"status": "error", "message": str(exc)}
        return results
    for item, dimensions in parsed:
        results[item["id"]] = {
            "status": "ok",
            "gpu": gpu,
            "window_seconds": config.window_seconds,
            "unused_tail_seconds": round(item["unused_tail"], 3),
            "windows": item["clips"],
            "dimensions": dimensions,
            "raw_results": str(results_path),
        }
    return results


def _run_vbench(window_dir: Path, output_dir: Path, config: TechnicalConfig, gpu: str) -> Path:
    _ensure_unzip()
    _load_vbench_checkpoints()
    _use_local_vbench_weights()
    try:
        from vbench import VBench
    except ImportError as exc:
        raise TechnicalUnavailable(
            "未安装 vbench。技术分默认使用 4 号或 5 号，安装 VBench 及其权重后再跑。"
        ) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    placeholder = output_dir / "placeholder_full_info.json"
    placeholder.write_text("[]", encoding="utf-8")
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    try:
        device = "cuda" if config.device.startswith("cuda") else config.device
        bench = VBench(device, str(placeholder), str(output_dir))
        bench.evaluate(
            videos_path=str(window_dir),
            name="windows",
            dimension_list=list(config.dimensions),
            local=True,
            mode="custom_input",
        )
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous
    results_path = output_dir / "windows_eval_results.json"
    if not results_path.is_file():
        raise RuntimeError(f"VBench 没有写出结果：{results_path}")
    return results_path


def _dimensions_from_payload(payload: dict[str, Any], config: TechnicalConfig, clip_rows: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions: dict[str, Any] = {}
    for name in config.dimensions:
        if name not in payload:
            raise RuntimeError(f"VBench 结果缺少 {name}")
        rows = scores_by_window(payload[name], clip_rows)
        if name == "imaging_quality":
            rows = _imaging_unit_interval(rows)
        dimensions[name] = summarize_windows(rows)
    return dimensions


def _stage_window(source: Path, target: Path) -> None:
    if target.is_symlink() or target.exists():
        target.unlink()
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def _batch_token(sample_id: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in sample_id.strip())
    return cleaned or "sample"


def _imaging_unit_interval(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """MUSIQ writes 0–100 on each window. The official mean divides that by 100."""
    if rows and max(float(row["score"]) for row in rows) > 1:
        return [{**row, "score": float(row["score"]) / 100} for row in rows]
    return rows


def _use_local_vbench_weights() -> None:
    """Point VBench at models/vbench and stop DINO from reading the torch hub cache."""
    missing = [str(path) for path in local_vbench_files().values() if not path.is_file()]
    if missing:
        raise TechnicalUnavailable("VBench 本地权重不完整：" + "、".join(missing))
    os.environ["VBENCH_CACHE_DIR"] = str(VBENCH_MODEL_DIR)
    import vbench.utils as vbench_utils
    vbench_utils.CACHE_DIR = str(VBENCH_MODEL_DIR)
    _redirect_dino_weight(local_vbench_files()["dino"])


def _redirect_dino_weight(weight: Path) -> None:
    """Official hubconf ignores the local path and downloads this filename into TORCH_HOME."""
    import torch
    hub = torch.hub
    current = getattr(hub.load_state_dict_from_url, "_long_eval_dino", None)
    if current is not None:
        current["weight"] = weight
        return
    original = hub.load_state_dict_from_url
    state = {"weight": weight}

    def load_state_dict_from_url(url, *args, **kwargs):
        if str(url).endswith(_DINO_WEIGHT_NAME):
            return torch.load(state["weight"], map_location=kwargs.get("map_location", "cpu"), weights_only=False)
        return original(url, *args, **kwargs)

    load_state_dict_from_url._long_eval_dino = state
    hub.load_state_dict_from_url = load_state_dict_from_url


def _load_vbench_checkpoints() -> None:
    """PyTorch 2.6+ refuses VBench's official checkpoints unless weights_only is off."""
    import torch
    if getattr(torch.load, "_long_eval_vbench", False):
        return
    original = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    load._long_eval_vbench = True
    torch.load = load


def _ensure_unzip() -> None:
    """VBench unpacks RAFT with the unzip command. Provide one when the host has none."""
    if shutil.which("unzip"):
        return
    bindir = Path(__file__).resolve().parents[1] / ".bin"
    bindir.mkdir(exist_ok=True)
    shim = bindir / "unzip"
    if not shim.is_file():
        shim.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "import zipfile\n"
            "from pathlib import Path\n"
            "args = sys.argv[1:]\n"
            "dest = Path('.')\n"
            "archive = None\n"
            "index = 0\n"
            "while index < len(args):\n"
            "    item = args[index]\n"
            "    if item == '-d' and index + 1 < len(args):\n"
            "        dest = Path(args[index + 1])\n"
            "        index += 2\n"
            "        continue\n"
            "    if item.startswith('-'):\n"
            "        index += 1\n"
            "        continue\n"
            "    archive = Path(item)\n"
            "    index += 1\n"
            "if archive is None:\n"
            "    raise SystemExit('usage: unzip -d DEST archive.zip')\n"
            "dest.mkdir(parents=True, exist_ok=True)\n"
            "with zipfile.ZipFile(archive) as package:\n"
            "    package.extractall(dest)\n",
            encoding="utf-8",
        )
        shim.chmod(0o755)
    os.environ["PATH"] = str(bindir) + os.pathsep + os.environ.get("PATH", "")


def _ensure_explicit_gpu(gpu: str, *, shared_by_default: bool, allow_shared_gpu: bool) -> None:
    if shared_by_default or allow_shared_gpu:
        return
    free = _free_mib(gpu)
    if free is None:
        raise TechnicalUnavailable(f"读不到 GPU {gpu} 的显存")
    # 4、5 以外的卡若几乎没有空闲显存，不默认挤上去。
    if free < 4096:
        raise TechnicalUnavailable(f"GPU {gpu} 空闲显存只有 {free} MiB。默认请用 4 号或 5 号。")


def _free_mib(gpu: str) -> int | None:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", gpu],
        capture_output=True, text=True, check=False, timeout=20,
    )
    if completed.returncode != 0:
        return None
    try:
        return int((completed.stdout or "").strip().splitlines()[0])
    except (ValueError, IndexError):
        return None
