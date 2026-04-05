#!/usr/bin/env python3
"""Prepare train_ready/ for FLUX LoRA training from dataset_manifest.json.

Creates symlinked images + caption sidecars and metadata.jsonl for diffusers.

Usage:
    python3 train/prepare_dataset.py /path/to/dataset
    python3 train/prepare_dataset.py /path/to/dataset --caption "override prefix"
    python3 train/prepare_dataset.py --job-file /tmp/prep_job.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path


def _read_caption(dataset_dir: str, caption_file: str | None, default_caption: str) -> str:
    if not caption_file:
        return default_caption
    p = Path(dataset_dir) / caption_file
    if p.exists():
        return p.read_text().strip() or default_caption
    return default_caption


def prepare_dataset(
    dataset_dir: str,
    caption: str = "stereo180sbs, stereoscopic 180 VR side by side half equirectangular",
    validation_split: float = 0.05,
    seed: int = 42,
) -> bool:
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        print(
            json.dumps({"error": f"dataset_manifest.json not found in {dataset_dir}"}),
            flush=True,
        )
        return False

    manifest = json.loads(manifest_path.read_text())
    samples = manifest.get("samples", [])
    if not samples:
        print(json.dumps({"error": "No samples in dataset_manifest.json"}), flush=True)
        return False

    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * validation_split))
    val_samples = shuffled[:val_count]
    train_samples = shuffled[val_count:]

    train_ready = dataset_dir / "train_ready"
    img_dir = train_ready / "images"
    depth_dir = train_ready / "depth"
    cache_vae = train_ready / "cache" / "vae"
    cache_text = train_ready / "cache" / "text"

    for d in (img_dir, depth_dir, cache_vae, cache_text):
        d.mkdir(parents=True, exist_ok=True)

    created = 0
    for s in train_samples:
        sid = s["id"]
        src_img = dataset_dir / s["image"]
        if not src_img.exists():
            continue

        link = img_dir / f"{sid}.png"
        if not link.exists():
            link.symlink_to(src_img.resolve())

        cap_text = _read_caption(str(dataset_dir), s.get("caption_file"), caption)
        txt_path = img_dir / f"{sid}.txt"
        if not txt_path.exists():
            txt_path.write_text(cap_text + "\n")

        depth_rel = s.get("depth")
        if depth_rel:
            src_depth = dataset_dir / depth_rel
            if src_depth.exists():
                dlink = depth_dir / f"{sid}.png"
                if not dlink.exists():
                    dlink.symlink_to(src_depth.resolve())

        created += 1

    # SimpleTuner-style multidatabackend (optional external tools)
    mdb = [
        {
            "id": "stereo-sbs-images",
            "type": "local",
            "dataset_type": "image",
            "instance_data_dir": str(img_dir.resolve()),
            "caption_strategy": "textfile",
            "cache_dir_vae": str(cache_vae.resolve()),
            "resolution": 1024,
            "resolution_type": "pixel_area",
            "minimum_image_size": 256,
            "prepend_instance_prompt": False,
        },
    ]
    (train_ready / "multidatabackend.json").write_text(json.dumps(mdb, indent=2))

    val_path = train_ready / "validation_samples.json"
    val_path.write_text(json.dumps(val_samples, indent=2))

    summary = {
        "total_samples": len(samples),
        "train_samples": created,
        "val_samples": len(val_samples),
        "output_dir": str(train_ready.resolve()),
    }
    print(json.dumps(summary), flush=True)
    return created > 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LoRA training folder from dataset_manifest.json")
    parser.add_argument("dataset_dir", nargs="?", help="Dataset root")
    parser.add_argument("--caption", default="stereo180sbs, stereoscopic 180 VR side by side half equirectangular")
    parser.add_argument("--validation-split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--job-file", type=str, default=None)

    args = parser.parse_args()

    if args.job_file:
        job = json.loads(Path(args.job_file).read_text())
        dataset_dir = job["dataset_dir"]
        caption = job.get("caption", args.caption)
        vs = job.get("validation_split", args.validation_split)
        seed = job.get("seed", args.seed)
    elif args.dataset_dir:
        dataset_dir = args.dataset_dir
        caption = args.caption
        vs = args.validation_split
        seed = args.seed
    else:
        parser.error("dataset_dir or --job-file required")
        return

    ok = prepare_dataset(dataset_dir, caption, vs, seed)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
