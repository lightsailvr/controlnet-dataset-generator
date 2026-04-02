#!/usr/bin/env python3
"""Convert pairs.json dataset to SimpleTuner training format.

Creates symlinked directories with 1:1 filename matching between
target and conditioning images, caption sidecar files, and a
multidatabackend.json config for SimpleTuner.

Usage:
    python3 train/prepare_dataset.py /path/to/dataset
    python3 train/prepare_dataset.py /path/to/dataset --caption "custom prompt"
    python3 train/prepare_dataset.py --job-file /tmp/prep_job.json
"""

import argparse
import json
import os
import random
import sys


def prepare_dataset(dataset_dir, caption="equirectangular panorama",
                    validation_split=0.05, seed=42):
    """Convert pairs.json to SimpleTuner format."""
    pairs_path = os.path.join(dataset_dir, "pairs.json")
    if not os.path.exists(pairs_path):
        print(json.dumps({"error": f"pairs.json not found in {dataset_dir}"}),
              flush=True)
        return False

    with open(pairs_path) as f:
        manifest = json.load(f)

    pairs = manifest.get("pairs", [])
    if not pairs:
        print(json.dumps({"error": "No pairs found in pairs.json"}), flush=True)
        return False

    # Shuffle and split
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * validation_split))
    val_pairs = shuffled[:val_count]
    train_pairs = shuffled[val_count:]

    # Create output directories
    train_ready = os.path.join(dataset_dir, "train_ready")
    target_dir = os.path.join(train_ready, "target")
    cond_dir = os.path.join(train_ready, "conditioning")
    cache_vae = os.path.join(train_ready, "cache", "vae")
    cache_text = os.path.join(train_ready, "cache", "text")

    for d in [target_dir, cond_dir, cache_vae, cache_text]:
        os.makedirs(d, exist_ok=True)

    # Create symlinks and caption files
    created = 0
    for pair in train_pairs:
        crop_id = _crop_id_from_path(pair["conditioning"])
        if not crop_id:
            continue

        # Source paths (relative to dataset_dir)
        target_src = os.path.join(dataset_dir, pair["target"])
        cond_src = os.path.join(dataset_dir, pair["conditioning"])

        if not os.path.exists(target_src) or not os.path.exists(cond_src):
            continue

        # Symlink target image with crop_id filename (flattens many-to-one)
        target_link = os.path.join(target_dir, f"{crop_id}.png")
        if not os.path.exists(target_link):
            os.symlink(os.path.abspath(target_src), target_link)

        # Symlink conditioning image
        cond_link = os.path.join(cond_dir, f"{crop_id}.png")
        if not os.path.exists(cond_link):
            os.symlink(os.path.abspath(cond_src), cond_link)

        # Caption sidecar (only for target images)
        caption_file = os.path.join(target_dir, f"{crop_id}.txt")
        if not os.path.exists(caption_file):
            with open(caption_file, "w") as cf:
                cf.write(caption)

        created += 1

    # Write multidatabackend.json
    mdb_config = [
        {
            "id": "equirect-target",
            "type": "local",
            "dataset_type": "image",
            "instance_data_dir": os.path.abspath(target_dir),
            "caption_strategy": "textfile",
            "cache_dir_vae": os.path.abspath(cache_vae),
            "resolution": 1024,
            "resolution_type": "pixel_area",
            "minimum_image_size": 512,
            "prepend_instance_prompt": False,
        },
        {
            "id": "equirect-conditioning",
            "type": "local",
            "dataset_type": "conditioning",
            "instance_data_dir": os.path.abspath(cond_dir),
            "conditioning_data_dir": os.path.abspath(cond_dir),
            "resolution": 1024,
            "resolution_type": "pixel_area",
            "minimum_image_size": 512,
        },
        {
            "id": "text-embeds",
            "type": "local",
            "dataset_type": "text_embeds",
            "cache_dir": os.path.abspath(cache_text),
            "write_batch_size": 128,
        },
    ]

    mdb_path = os.path.join(train_ready, "multidatabackend.json")
    with open(mdb_path, "w") as f:
        json.dump(mdb_config, f, indent=2)

    # Write validation pairs list (for reference)
    val_path = os.path.join(train_ready, "validation_pairs.json")
    with open(val_path, "w") as f:
        json.dump(val_pairs, f, indent=2)

    summary = {
        "total_pairs": len(pairs),
        "train_pairs": created,
        "val_pairs": len(val_pairs),
        "output_dir": os.path.abspath(train_ready),
    }
    print(json.dumps(summary), flush=True)
    return True


def _crop_id_from_path(conditioning_path):
    """Extract crop ID from a conditioning path like 'conditioning/crop_name.png'."""
    basename = os.path.basename(conditioning_path)
    name, _ = os.path.splitext(basename)
    return name


def main():
    parser = argparse.ArgumentParser(
        description="Convert pairs.json dataset to SimpleTuner training format"
    )
    parser.add_argument("dataset_dir", nargs="?", help="Path to dataset directory")
    parser.add_argument("--caption", default="equirectangular panorama",
                        help="Caption text for all training pairs")
    parser.add_argument("--validation-split", type=float, default=0.05,
                        help="Fraction of pairs to reserve for validation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffle/split")
    parser.add_argument("--job-file", help="JSON job file (Electron mode)")

    args = parser.parse_args()

    if args.job_file:
        with open(args.job_file) as f:
            job = json.load(f)
        dataset_dir = job["dataset_dir"]
        caption = job.get("caption", args.caption)
        validation_split = job.get("validation_split", args.validation_split)
        seed = job.get("seed", args.seed)
    elif args.dataset_dir:
        dataset_dir = args.dataset_dir
        caption = args.caption
        validation_split = args.validation_split
        seed = args.seed
    else:
        parser.error("Either dataset_dir or --job-file is required")
        return

    success = prepare_dataset(dataset_dir, caption, validation_split, seed)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
