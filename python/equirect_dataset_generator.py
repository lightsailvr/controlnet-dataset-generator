#!/usr/bin/env python3
"""
Stereoscopic 180 SBS LoRA dataset builder.

Extracts frames from VR180 / 180° SBS (or mono 180/360 equirect) media and writes:
  - frames/<id>.png   — training image (2:1 aspect for SBS)
  - frames/<id>.txt   — caption sidecar
  - depth/<id>.png    — optional disparity (FoundationStereo on CUDA PC, or OpenCV SGBM fallback)
  - dataset_manifest.json

CLI:
    python equirect_dataset_generator.py input.mov -o ./dataset -t 180sbs
    python equirect_dataset_generator.py --job-file /tmp/job.json

Requires: opencv-python-headless, numpy, Pillow; ffmpeg for video.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

try:
    from depth_extractor import depth_from_sbs_bgr
except ImportError:
    depth_from_sbs_bgr = None  # optional if extractDepth false

# One UserWarning per process when auto falls back to SGBM
_AUTO_DEPTH_WARNED = [False]

# ── formats (same as before) ──

VIDEO_EXTS = {
    ".mov", ".mp4", ".mkv", ".avi", ".mxf", ".webm", ".m4v",
    ".mpg", ".mpeg", ".ts", ".mts", ".m2ts", ".wmv", ".flv",
    ".3gp", ".ogv", ".r3d", ".braw",
}

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".exr",
    ".hdr", ".bmp", ".webp", ".dpx",
}

DEFAULTS = {
    "sourceType": "180sbs",
    "frameInterval": 30,
    "trainingRes": 512,
    "seed": 42,
    "captionPrefix": "stereo180sbs, stereoscopic 180 VR side by side half equirectangular",
    "extractDepth": True,
    # Depth: auto | foundation_stereo | sgbm (see depth_extractor.py)
    "depthBackend": "auto",
    "foundationStereoRoot": "",
    "foundationStereoCkpt": "",
    "foundationStereoScale": 1.0,
    "foundationStereoHiera": 0,
    "foundationStereoValidIters": 32,
}


def _depth_kwargs_from_config(cfg: dict) -> dict:
    root = (cfg.get("foundationStereoRoot") or "").strip()
    ckpt = (cfg.get("foundationStereoCkpt") or "").strip()
    return {
        "backend": (cfg.get("depthBackend") or "auto").strip().lower(),
        "fs_root": root or None,
        "fs_ckpt": ckpt or None,
        "fs_scale": float(cfg.get("foundationStereoScale", 1.0)),
        "fs_hiera": int(cfg.get("foundationStereoHiera", 0)),
        "fs_valid_iters": int(cfg.get("foundationStereoValidIters", 32)),
        "_auto_warned": _AUTO_DEPTH_WARNED,
    }


def log(msg: str) -> None:
    print(msg, flush=True)


def get_video_info(video_path: str) -> dict | None:
    import subprocess

    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"ERROR: ffprobe failed on {video_path}")
        return None

    info = json.loads(result.stdout)
    video_stream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    if not video_stream:
        return None

    fps_parts = video_stream.get("r_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])

    nb_frames = video_stream.get("nb_frames")
    if nb_frames and nb_frames != "N/A":
        total_frames = int(nb_frames)
    else:
        duration = float(info.get("format", {}).get("duration", 0))
        total_frames = int(duration * fps)

    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": fps,
        "total_frames": total_frames,
        "codec": video_stream.get("codec_name", "unknown"),
    }


def extract_frames_ffmpeg(video_path: str, output_dir: str, interval: int) -> list[str]:
    import subprocess

    os.makedirs(output_dir, exist_ok=True)
    select_filter = f"select=not(mod(n\\,{interval}))"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", select_filter,
        "-vsync", "vfr",
        "-pix_fmt", "rgb24",
        "-f", "image2",
        os.path.join(output_dir, "frame_%06d.png"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and "Error" in (result.stderr or ""):
        log(f"  WARNING: ffmpeg: {result.stderr[-300:]}")

    return sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith("frame_") and f.endswith(".png")
    )


def load_image(path: str) -> np.ndarray | None:
    ext = Path(path).suffix.lower()
    if ext in (".exr", ".hdr"):
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if img is not None:
            img = np.clip(img * 255, 0, 255).astype(np.uint8) if img.max() <= 1.0 else np.clip(img, 0, 255).astype(np.uint8)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_to_2_1_equirect(rgb: np.ndarray, short_side: int) -> np.ndarray:
    """Resize to width = 2 * height (equirect / SBS canvas)."""
    h_out = int(short_side)
    w_out = h_out * 2
    return cv2.resize(rgb, (w_out, h_out), interpolation=cv2.INTER_LANCZOS4)


def prepare_source_rgb(rgb: np.ndarray, source_type: str) -> np.ndarray:
    """Normalize mono 180/360 inputs to full-width equirect canvas for consistent 2:1 export."""
    if source_type == "180sbs":
        return rgb
    if source_type == "180":
        h, w = rgb.shape[:2]
        full_w = w * 2
        padded = np.zeros((h, full_w, 3), dtype=rgb.dtype)
        padded[:, full_w // 4 : full_w // 4 + w] = rgb
        return padded
    if source_type == "360":
        return rgb
    return rgb


def process_files(file_paths: list[str], config: dict, output_dir: str) -> None:
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    depth_dir = output_dir / "depth"
    source_root = output_dir / "source_equirects"

    frames_dir.mkdir(parents=True, exist_ok=True)
    if config.get("extractDepth", True):
        depth_dir.mkdir(parents=True, exist_ok=True)

    source_type = config.get("sourceType", "180sbs")
    frame_interval = int(config.get("frameInterval", 30))
    training_res = int(config.get("trainingRes", 512))
    caption_prefix = (config.get("captionPrefix") or DEFAULTS["captionPrefix"]).strip()
    extract_depth = bool(config.get("extractDepth", True)) and source_type == "180sbs"

    if extract_depth and depth_from_sbs_bgr is None:
        log("  WARNING: depth_extractor not available; disabling depth export")
        extract_depth = False

    manifest_path = output_dir / "dataset_manifest.json"
    samples: list[dict] = []
    existing_ids: set[str] = set()

    if manifest_path.exists():
        try:
            prev = json.loads(manifest_path.read_text())
            samples = prev.get("samples", [])
            existing_ids = {s["id"] for s in samples}
            log(f"  Found existing manifest with {len(samples)} samples — appending")
        except (json.JSONDecodeError, KeyError):
            samples = []

    log(f"\n{'=' * 60}")
    log("  Stereoscopic 180 SBS LoRA — dataset builder")
    log(f"{'=' * 60}")
    log(f"  Files:        {len(file_paths)}")
    log(f"  Source type:  {source_type}")
    log(f"  Frame step:   every {frame_interval} frames (video)")
    log(f"  Output size:  {training_res * 2}x{training_res} (2:1)")
    depth_be = (config.get("depthBackend") or "auto").strip().lower() if extract_depth else "off"
    log(f"  Depth maps:   {'on' if extract_depth else 'off'}{' (' + depth_be + ')' if extract_depth else ''}")
    log(f"  Output:       {output_dir}")
    log(f"{'=' * 60}\n")

    start = time.time()
    files_done = 0

    for file_idx, file_path in enumerate(file_paths):
        fp = Path(file_path)
        if not fp.exists():
            log(f"  WARNING: missing {fp}, skip")
            continue
        ext = fp.suffix.lower()
        log(f"\n[{file_idx + 1}/{len(file_paths)}] {fp.name}")

        if ext in VIDEO_EXTS:
            info = get_video_info(str(fp))
            if not info:
                continue
            vid_dir = source_root / fp.stem
            vid_dir.mkdir(parents=True, exist_ok=True)
            paths = extract_frames_ffmpeg(str(fp), str(vid_dir), frame_interval)
            log(f"  Extracted {len(paths)} frames")

            for p in paths:
                frame_id = f"{fp.stem}_{Path(p).stem}"
                if frame_id in existing_ids:
                    continue
                bgr = cv2.imread(p, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                _write_sample(
                    rgb, frame_id, fp.name, config, source_type,
                    training_res, caption_prefix, extract_depth,
                    frames_dir, depth_dir, samples,
                )
                existing_ids.add(frame_id)
            files_done += 1

        elif ext in IMAGE_EXTS:
            rgb = load_image(str(fp))
            if rgb is None:
                log("  ERROR: could not read image")
                continue
            dest = source_root / fp.name
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dest), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            frame_id = fp.stem
            if frame_id in existing_ids:
                log("  Skipped (already in dataset)")
                continue
            _write_sample(
                rgb, frame_id, fp.name, config, source_type,
                training_res, caption_prefix, extract_depth,
                frames_dir, depth_dir, samples,
            )
            existing_ids.add(frame_id)
            files_done += 1
        else:
            log(f"  Unsupported: {ext}")

    manifest = {
        "generator": "equirect_dataset_generator.py",
        "format": "stereo_lora_v1",
        "total_samples": len(samples),
        "files_processed": files_done,
        "training_resolution": f"{training_res * 2}x{training_res}",
        "config": config,
        "samples": samples,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Legacy-style summary for Electron
    (output_dir / "generation_results.json").write_text(
        json.dumps({
            "total_frames": len(samples),
            "total_samples": len(samples),
            "files_processed": files_done,
        })
    )

    # Minimal pairs.json for older tooling: one pseudo-entry per sample
    pairs_legacy = []
    for s in samples:
        pairs_legacy.append({
            "conditioning": s.get("depth") or s["image"],
            "target": s["image"],
            "metadata": s["caption_file"],
            "source_frame": s["id"],
            "source_file": s["source_file"],
        })
    (output_dir / "pairs.json").write_text(
        json.dumps({
            "generator": "equirect_dataset_generator.py",
            "format": "stereo_lora_v1",
            "total_pairs": len(pairs_legacy),
            "total_frames": len(samples),
            "pairs": pairs_legacy,
        }, indent=2)
    )

    elapsed = time.time() - start
    log(f"\n{'=' * 60}")
    log(f"  DONE — {elapsed:.1f}s | {len(samples)} samples")
    log(f"{'=' * 60}\n")


def _write_sample(
    rgb: np.ndarray,
    frame_id: str,
    source_name: str,
    config: dict,
    source_type: str,
    training_res: int,
    caption_prefix: str,
    extract_depth: bool,
    frames_dir: Path,
    depth_dir: Path,
    samples: list[dict],
) -> None:
    rgb = prepare_source_rgb(rgb, source_type)
    out_rgb = resize_to_2_1_equirect(rgb, training_res)
    img_path = frames_dir / f"{frame_id}.png"
    cap_path = frames_dir / f"{frame_id}.txt"
    cv2.imwrite(str(img_path), cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))

    caption = caption_prefix
    if config.get("extraCaption"):
        caption = f"{caption}, {config['extraCaption']}"
    cap_path.write_text(caption + "\n")

    depth_rel = None
    if extract_depth and source_type == "180sbs":
        bgr_full = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        try:
            dk = _depth_kwargs_from_config(config)
            d = depth_from_sbs_bgr(bgr_full, **dk)
            depth_rel = f"depth/{frame_id}.png"
            cv2.imwrite(str(depth_dir / f"{frame_id}.png"), d)
        except Exception as e:
            log(f"    WARNING depth for {frame_id}: {e}")

    samples.append({
        "id": frame_id,
        "image": f"frames/{frame_id}.png",
        "caption_file": f"frames/{frame_id}.txt",
        "depth": depth_rel,
        "source_file": source_name,
        "source_type": source_type,
    })


def collect_files_from_path(input_path: str) -> list[str]:
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files = []
        for ext in sorted(VIDEO_EXTS | IMAGE_EXTS):
            files.extend(str(f) for f in p.rglob(f"*{ext}"))
            files.extend(str(f) for f in p.rglob(f"*{ext.upper()}"))
        return sorted(set(files))
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stereo 180 SBS LoRA dataset")
    parser.add_argument("inputs", nargs="*", help="Media files or folders")
    parser.add_argument("-o", "--output", default="./dataset", help="Output directory")
    parser.add_argument("-t", "--source-type", choices=["360", "180", "180sbs"], default="180sbs")
    parser.add_argument("-i", "--frame-interval", type=int, default=30)
    parser.add_argument("-r", "--training-res", type=int, default=512)
    parser.add_argument("--caption-prefix", type=str, default=None)
    parser.add_argument("--no-depth", action="store_true", help="Skip disparity extraction")
    parser.add_argument(
        "--depth-backend",
        choices=["auto", "foundation_stereo", "sgbm"],
        default=None,
        help="Disparity backend: FoundationStereo (CUDA PC), OpenCV SGBM, or auto-detect",
    )
    parser.add_argument("--foundation-stereo-root", type=str, default=None, help="Path to NVlabs/FoundationStereo clone")
    parser.add_argument("--foundation-stereo-ckpt", type=str, default=None, help="Path to model_best_bp2.pth")
    parser.add_argument("--foundation-stereo-scale", type=float, default=None, help="Resize factor (0,1], default 1")
    parser.add_argument("--foundation-stereo-hiera", type=int, choices=[0, 1], default=None, help="1 = hierarchical for >~1K px")
    parser.add_argument("--foundation-stereo-iters", type=int, default=None, help="FoundationStereo valid_iters (default 32)")
    parser.add_argument("--job-file", type=str, default=None)

    args = parser.parse_args()

    if args.job_file:
        job = json.loads(Path(args.job_file).read_text())
        paths = job.get("files", [])
        cfg = {**DEFAULTS, **job.get("config", {})}
        out = job.get("outputDir", "./dataset")
        if args.caption_prefix:
            cfg["captionPrefix"] = args.caption_prefix
        if args.depth_backend is not None:
            cfg["depthBackend"] = args.depth_backend
        if args.foundation_stereo_root is not None:
            cfg["foundationStereoRoot"] = args.foundation_stereo_root
        if args.foundation_stereo_ckpt is not None:
            cfg["foundationStereoCkpt"] = args.foundation_stereo_ckpt
        if args.foundation_stereo_scale is not None:
            cfg["foundationStereoScale"] = args.foundation_stereo_scale
        if args.foundation_stereo_hiera is not None:
            cfg["foundationStereoHiera"] = args.foundation_stereo_hiera
        if args.foundation_stereo_iters is not None:
            cfg["foundationStereoValidIters"] = args.foundation_stereo_iters
        process_files(paths, cfg, out)
        return

    if not args.inputs:
        parser.error("Provide inputs or --job-file")

    paths: list[str] = []
    for inp in args.inputs:
        paths.extend(collect_files_from_path(inp))
    if not paths:
        log("ERROR: No supported media found.")
        sys.exit(1)

    cfg = {
        **DEFAULTS,
        "sourceType": args.source_type,
        "frameInterval": args.frame_interval,
        "trainingRes": args.training_res,
        "captionPrefix": args.caption_prefix or DEFAULTS["captionPrefix"],
        "extractDepth": not args.no_depth,
        "seed": DEFAULTS["seed"],
    }
    if args.depth_backend is not None:
        cfg["depthBackend"] = args.depth_backend
    if args.foundation_stereo_root is not None:
        cfg["foundationStereoRoot"] = args.foundation_stereo_root
    if args.foundation_stereo_ckpt is not None:
        cfg["foundationStereoCkpt"] = args.foundation_stereo_ckpt
    if args.foundation_stereo_scale is not None:
        cfg["foundationStereoScale"] = args.foundation_stereo_scale
    if args.foundation_stereo_hiera is not None:
        cfg["foundationStereoHiera"] = args.foundation_stereo_hiera
    if args.foundation_stereo_iters is not None:
        cfg["foundationStereoValidIters"] = args.foundation_stereo_iters
    process_files(paths, cfg, args.output)


if __name__ == "__main__":
    main()
