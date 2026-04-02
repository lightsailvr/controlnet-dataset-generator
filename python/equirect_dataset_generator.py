#!/usr/bin/env python3
"""
equirect_dataset_generator.py
─────────────────────────────
Generate ControlNet training datasets from 360°/180° equirectangular media.

Supports:
  - Video files (ProRes, H.264, HEVC, R3D, etc. — anything ffmpeg reads)
  - Image files (JPG, PNG, TIFF, EXR, HDR, DPX, BMP, WebP)
  - Full 360° equirect, 180° equirect, and 180° side-by-side stereo
  - CLI mode and Electron job-file mode

CLI Usage:
    python equirect_dataset_generator.py input.mov -o ./dataset
    python equirect_dataset_generator.py /path/to/folder -o ./dataset -t 180sbs

Electron (job-file) mode:
    python equirect_dataset_generator.py --job-file /tmp/job.json

Requirements:
    pip3 install py360convert opencv-python-headless numpy Pillow
    ffmpeg must be on PATH (for video files)
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import py360convert
except ImportError:
    print("ERROR: py360convert not found. Install with: pip3 install py360convert")
    sys.exit(1)


# ─────────────────────────────────────────────
# Supported formats
# ─────────────────────────────────────────────

VIDEO_EXTS = {
    ".mov", ".mp4", ".mkv", ".avi", ".mxf", ".webm", ".m4v",
    ".mpg", ".mpeg", ".ts", ".mts", ".m2ts", ".wmv", ".flv",
    ".3gp", ".ogv", ".r3d", ".braw",
}

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".exr",
    ".hdr", ".bmp", ".webp", ".dpx",
}


# ─────────────────────────────────────────────
# Configuration defaults
# ─────────────────────────────────────────────

DEFAULTS = {
    "sourceType": "360",
    "frameInterval": 30,
    "cropsPerFrame": 10,
    "trainingRes": 512,
    "conditioningRes": 512,
    "seed": 42,
    "fovMin": 60,
    "fovMax": 110,
    "horizonBias": 0.7,
}

# Pitch ranges by source type
PITCH_RANGE = {"360": (-45, 45), "180": (-30, 30), "180sbs": (-30, 30)}
YAW_RANGE = {"360": (0, 360), "180": (90, 270), "180sbs": (90, 270)}
HORIZON_BIAS_PITCH = (-15, 15)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def log(msg: str):
    """Print with flush for real-time output in Electron."""
    print(msg, flush=True)


def get_video_info(video_path: str) -> dict:
    """Use ffprobe to get frame count, resolution, fps."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"ERROR: ffprobe failed on {video_path}")
        return None

    info = json.loads(result.stdout)
    video_stream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "video"),
        None
    )

    if not video_stream:
        log(f"ERROR: No video stream found in {video_path}")
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
    """Extract frames from video at given interval using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)
    select_filter = f"select=not(mod(n\\,{interval}))"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", select_filter,
        "-vsync", "vfr",
        "-pix_fmt", "rgb24",
        "-f", "image2",
        os.path.join(output_dir, "frame_%06d.png")
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and "Error" in result.stderr:
        log(f"  WARNING: ffmpeg error: {result.stderr[-300:]}")

    frames = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith("frame_") and f.endswith(".png")
    ])
    return frames


def load_image(path: str) -> np.ndarray | None:
    """Load an image file, handling HDR/EXR via OpenCV flags."""
    ext = Path(path).suffix.lower()
    if ext in (".exr", ".hdr"):
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if img is not None:
            # Tonemap HDR to 8-bit for training
            img = np.clip(img * 255, 0, 255).astype(np.uint8) if img.max() <= 1.0 else np.clip(img, 0, 255).astype(np.uint8)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def handle_sbs_180(frame: np.ndarray, eye: str = "left") -> np.ndarray:
    """Split SBS 180° frame → pad to full equirect."""
    h, w = frame.shape[:2]
    half_w = w // 2
    eye_frame = frame[:, :half_w] if eye == "left" else frame[:, half_w:]

    full_w = half_w * 2
    padded = np.zeros((h, full_w, 3), dtype=frame.dtype)
    pad_left = full_w // 4
    padded[:, pad_left:pad_left + half_w] = eye_frame
    return padded


def handle_180(frame: np.ndarray) -> np.ndarray:
    """Pad 180° equirect to full 360° equirect."""
    h, w = frame.shape[:2]
    full_w = w * 2
    padded = np.zeros((h, full_w, 3), dtype=frame.dtype)
    padded[:, full_w // 4: full_w // 4 + w] = frame
    return padded


def generate_crop_params(config: dict, num_crops: int, seed: int = None) -> list[dict]:
    """Generate randomised yaw/pitch/FOV crop parameters."""
    if seed is not None:
        random.seed(seed)

    source_type = config["sourceType"]
    fov_min = config.get("fovMin", 60)
    fov_max = config.get("fovMax", 110)
    horizon_bias = config.get("horizonBias", 0.7)
    pitch_range = PITCH_RANGE.get(source_type, (-45, 45))
    yaw_range = YAW_RANGE.get(source_type, (0, 360))

    crops = []
    for i in range(num_crops):
        fov = random.uniform(fov_min, fov_max)
        yaw = random.uniform(*yaw_range)

        if random.random() < horizon_bias:
            pitch = random.uniform(*HORIZON_BIAS_PITCH)
        else:
            pitch = random.uniform(*pitch_range)

        crops.append({
            "index": i,
            "yaw": round(yaw, 2),
            "pitch": round(pitch, 2),
            "fov_deg": round(fov, 2),
        })

    return crops


def extract_rectilinear(equirect: np.ndarray, yaw: float, pitch: float,
                         fov_deg: float, output_size: int) -> np.ndarray:
    """Extract a rectilinear perspective crop from equirectangular image."""
    u_deg = yaw - 180.0
    rectilinear = py360convert.e2p(
        equirect,
        fov_deg=fov_deg,
        u_deg=u_deg,
        v_deg=pitch,
        out_hw=(output_size, output_size),
        mode="bilinear"
    )
    return rectilinear.astype(np.uint8)


def resize_equirect(equirect: np.ndarray, target_size: int) -> np.ndarray:
    """Resize equirect to 2:1 aspect at target height."""
    h_out = target_size
    w_out = target_size * 2
    return cv2.resize(equirect, (w_out, h_out), interpolation=cv2.INTER_LANCZOS4)


def process_equirect_frame(
    equirect_rgb: np.ndarray,
    frame_name: str,
    source_name: str,
    config: dict,
    dirs: dict,
    frame_idx: int,
    pairs_manifest: list,
) -> int:
    """Process a single equirect frame: resize target, generate crops. Returns crop count."""
    source_type = config["sourceType"]
    training_res = config["trainingRes"]
    conditioning_res = config.get("conditioningRes", training_res)
    crops_per_frame = config["cropsPerFrame"]
    seed = config.get("seed", 42)

    # Handle 180 variants
    if source_type == "180sbs":
        equirect_rgb = handle_sbs_180(equirect_rgb, eye="left")
    elif source_type == "180":
        equirect_rgb = handle_180(equirect_rgb)

    # Resize for target
    target_equirect = resize_equirect(equirect_rgb, training_res)
    target_path = dirs["target"] / f"{frame_name}.png"
    cv2.imwrite(str(target_path), cv2.cvtColor(target_equirect, cv2.COLOR_RGB2BGR))

    # Generate crops
    frame_seed = (seed + frame_idx) if seed is not None else None
    crop_params = generate_crop_params(config, crops_per_frame, seed=frame_seed)

    count = 0
    for crop in crop_params:
        crop_id = f"{frame_name}_y{crop['yaw']:.0f}_p{crop['pitch']:.0f}_f{crop['fov_deg']:.0f}"

        rect_crop = extract_rectilinear(
            equirect_rgb,
            yaw=crop["yaw"],
            pitch=crop["pitch"],
            fov_deg=crop["fov_deg"],
            output_size=conditioning_res
        )

        cond_path = dirs["conditioning"] / f"{crop_id}.png"
        cv2.imwrite(str(cond_path), cv2.cvtColor(rect_crop, cv2.COLOR_RGB2BGR))

        meta = {
            "source_frame": frame_name,
            "source_file": source_name,
            "source_type": source_type,
            "crop_params": {
                "yaw_deg": crop["yaw"],
                "pitch_deg": crop["pitch"],
                "fov_deg": crop["fov_deg"],
            },
            "conditioning_resolution": conditioning_res,
            "target_resolution": f"{training_res * 2}x{training_res}",
            "conditioning_file": f"{crop_id}.png",
            "target_file": f"{frame_name}.png",
        }
        meta_path = dirs["metadata"] / f"{crop_id}.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        pairs_manifest.append({
            "conditioning": f"conditioning/{crop_id}.png",
            "target": f"target/{frame_name}.png",
            "metadata": f"metadata/{crop_id}.json",
            "yaw": crop["yaw"],
            "pitch": crop["pitch"],
            "fov_deg": crop["fov_deg"],
            "source_frame": frame_name,
            "source_file": source_name,
        })
        count += 1

    return count


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def process_files(file_paths: list[str], config: dict, output_dir: str):
    """Process a list of files (videos and images) into a training dataset."""
    output_dir = Path(output_dir)
    dirs = {
        "source": output_dir / "source_equirects",
        "conditioning": output_dir / "conditioning",
        "target": output_dir / "target",
        "metadata": output_dir / "metadata",
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    frame_interval = config.get("frameInterval", 30)
    crops_per_frame = config.get("cropsPerFrame", 10)
    training_res = config.get("trainingRes", 512)

    log(f"\n{'=' * 60}")
    log(f"  ControlNet Dataset Generator")
    log(f"{'=' * 60}")
    log(f"  Files:          {len(file_paths)}")
    log(f"  Source type:    {config.get('sourceType', '360')}")
    log(f"  Frame interval: {frame_interval}")
    log(f"  Crops/frame:    {crops_per_frame}")
    log(f"  Training res:   {training_res}px ({training_res*2}x{training_res})")
    log(f"  FOV range:      {config.get('fovMin', 60)}°–{config.get('fovMax', 110)}°")
    log(f"  Horizon bias:   {int(config.get('horizonBias', 0.7) * 100)}%")
    log(f"  Output:         {output_dir}")
    log(f"{'=' * 60}\n")

    # Load existing manifest if appending to a previous dataset
    existing_frames = set()
    pairs_manifest = []
    existing_pairs = 0
    existing_files = 0
    manifest_path = output_dir / "pairs.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                prev = json.load(f)
            pairs_manifest = prev.get("pairs", [])
            existing_pairs = len(pairs_manifest)
            existing_frames = {p["source_frame"] for p in pairs_manifest}
            existing_files = prev.get("files_processed", 0)
            log(f"  Found existing dataset with {existing_pairs} pairs "
                f"({len(existing_frames)} frames) — appending")
        except (json.JSONDecodeError, KeyError) as e:
            log(f"  WARNING: Could not read existing pairs.json ({e}), starting fresh")
            pairs_manifest = []

    total_frames = len(existing_frames)
    total_crops = existing_pairs
    files_processed = existing_files
    global_frame_idx = len(existing_frames)
    start_time = time.time()

    for file_idx, file_path in enumerate(file_paths):
        fp = Path(file_path)
        ext = fp.suffix.lower()

        if not fp.exists():
            log(f"  WARNING: File not found: {fp}, skipping")
            continue

        log(f"\n[{file_idx + 1}/{len(file_paths)}] Processing: {fp.name}")

        if ext in VIDEO_EXTS:
            # ── Video file ──
            video_info = get_video_info(str(fp))
            if not video_info:
                log(f"  ERROR: Could not read video info, skipping")
                continue

            log(f"  Resolution: {video_info['width']}x{video_info['height']} | "
                f"FPS: {video_info['fps']:.2f} | "
                f"Frames: {video_info['total_frames']} | "
                f"Codec: {video_info['codec']}")

            expected = video_info["total_frames"] // frame_interval
            log(f"  Extracting ~{expected} frames (every {frame_interval} frames)...")

            # Extract frames to a temp subdir under source_equirects
            video_source_dir = dirs["source"] / fp.stem
            frame_paths = extract_frames_ffmpeg(str(fp), str(video_source_dir), frame_interval)
            log(f"  Extracted {len(frame_paths)} frames")

            skipped = 0
            for local_idx, frame_path in enumerate(frame_paths):
                frame_name = f"{fp.stem}_{Path(frame_path).stem}"
                if frame_name in existing_frames:
                    skipped += 1
                    continue
                equirect = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                if equirect is None:
                    continue
                equirect_rgb = cv2.cvtColor(equirect, cv2.COLOR_BGR2RGB)

                count = process_equirect_frame(
                    equirect_rgb, frame_name, fp.name,
                    config, dirs, global_frame_idx, pairs_manifest
                )
                existing_frames.add(frame_name)
                total_crops += count
                total_frames += 1
                global_frame_idx += 1

                if (local_idx + 1) % 10 == 0 or local_idx == len(frame_paths) - 1:
                    elapsed = time.time() - start_time
                    log(f"    [{local_idx + 1}/{len(frame_paths)}] {count} crops | "
                        f"Total: {total_crops} pairs | "
                        f"Elapsed: {elapsed:.0f}s")
            if skipped:
                log(f"  Skipped {skipped} frames already in dataset")

            files_processed += 1
            log(f"  ✓ {fp.name} complete")

        elif ext in IMAGE_EXTS:
            # ── Image file ──
            equirect_rgb = load_image(str(fp))
            if equirect_rgb is None:
                log(f"  ERROR: Could not read image, skipping")
                continue

            log(f"  Resolution: {equirect_rgb.shape[1]}x{equirect_rgb.shape[0]}")

            # Copy source
            source_dest = dirs["source"] / fp.name
            if not source_dest.exists():
                cv2.imwrite(str(source_dest), cv2.cvtColor(equirect_rgb, cv2.COLOR_RGB2BGR))

            frame_name = fp.stem
            if frame_name in existing_frames:
                log(f"  Skipped (already in dataset)")
                continue
            count = process_equirect_frame(
                equirect_rgb, frame_name, fp.name,
                config, dirs, global_frame_idx, pairs_manifest
            )
            existing_frames.add(frame_name)
            total_crops += count
            total_frames += 1
            global_frame_idx += 1
            files_processed += 1
            log(f"  ✓ {fp.name} complete — {count} pairs")

        else:
            log(f"  Skipping unsupported format: {ext}")

    # ── Write manifest ──
    new_pairs = len(pairs_manifest) - existing_pairs
    log(f"\nWriting pairs manifest ({new_pairs} new, {len(pairs_manifest)} total)...")
    manifest_data = {
        "generator": "equirect_dataset_generator.py",
        "total_pairs": len(pairs_manifest),
        "total_frames": total_frames,
        "files_processed": files_processed,
        "training_resolution": f"{training_res * 2}x{training_res}",
        "conditioning_resolution": f"{config.get('conditioningRes', training_res)}x{config.get('conditioningRes', training_res)}",
        "config": config,
        "pairs": pairs_manifest,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    # ── Write results for Electron ──
    results = {
        "total_pairs": len(pairs_manifest),
        "total_frames": total_frames,
        "files_processed": files_processed,
    }
    results_path = output_dir / "generation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Summary ──
    elapsed = time.time() - start_time
    log(f"\n{'=' * 60}")
    log(f"  COMPLETE — {elapsed:.1f}s")
    log(f"{'=' * 60}")
    log(f"  Total pairs:    {total_crops}")
    log(f"  Source frames:  {total_frames}")
    log(f"  Files:          {files_processed}")
    log(f"  Output:         {output_dir}")
    log(f"{'=' * 60}\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def collect_files_from_path(input_path: str) -> list[str]:
    """If input is a directory, recursively collect supported files."""
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    elif p.is_dir():
        files = []
        for ext in sorted(VIDEO_EXTS | IMAGE_EXTS):
            files.extend(str(f) for f in p.rglob(f"*{ext}"))
            files.extend(str(f) for f in p.rglob(f"*{ext.upper()}"))
        return sorted(set(files))
    else:
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Generate ControlNet training dataset from 360°/180° media.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("inputs", nargs="*", default=[],
                        help="Input video/image files or directories")
    parser.add_argument("--output", "-o", type=str, default="./dataset",
                        help="Output directory")
    parser.add_argument("--source-type", "-t", type=str,
                        choices=["360", "180", "180sbs"], default="360")
    parser.add_argument("--frame-interval", "-i", type=int, default=30)
    parser.add_argument("--crops-per-frame", "-c", type=int, default=10)
    parser.add_argument("--training-res", "-r", type=int, default=512)
    parser.add_argument("--conditioning-res", type=int, default=None)
    parser.add_argument("--fov-min", type=int, default=60)
    parser.add_argument("--fov-max", type=int, default=110)
    parser.add_argument("--horizon-bias", type=float, default=0.7)
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--job-file", type=str, default=None,
                        help="JSON job file (used by Electron app)")

    args = parser.parse_args()

    if args.job_file:
        # ── Electron mode ──
        with open(args.job_file, "r") as f:
            job = json.load(f)

        file_paths = job.get("files", [])
        config = {**DEFAULTS, **job.get("config", {})}
        output_dir = job.get("outputDir", "./dataset")

        process_files(file_paths, config, output_dir)
    else:
        # ── CLI mode ──
        if not args.inputs:
            parser.error("Provide input files/directories or use --job-file")

        file_paths = []
        for inp in args.inputs:
            file_paths.extend(collect_files_from_path(inp))

        if not file_paths:
            log("ERROR: No supported media files found in the provided inputs.")
            sys.exit(1)

        config = {
            "sourceType": args.source_type,
            "frameInterval": args.frame_interval,
            "cropsPerFrame": args.crops_per_frame,
            "trainingRes": args.training_res,
            "conditioningRes": args.conditioning_res or args.training_res,
            "seed": args.seed,
            "fovMin": args.fov_min,
            "fovMax": args.fov_max,
            "horizonBias": args.horizon_bias,
        }

        process_files(file_paths, config, args.output)


if __name__ == "__main__":
    main()
