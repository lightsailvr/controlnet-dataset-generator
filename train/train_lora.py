#!/usr/bin/env python3
"""FLUX LoRA training wrapper for stereoscopic 180 SBS datasets.

Detects hardware, builds metadata.jsonl from train_ready/images/, then runs
accelerate launch train_lora_flux.py with progress JSON for the Electron UI.

Usage:
    python3 train/train_lora.py --dataset /path/to/dataset
    python3 train/train_lora.py --job-file /tmp/train_job.json
    python3 train/train_lora.py --detect-hardware
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def detect_hardware():
    """Detect hardware platform and return preset info."""
    system = platform.system()
    result = {
        "system": system,
        "platform": "unknown",
        "preset": None,
        "gpu_info": None,
        "memory_gb": 0,
    }

    if system == "Darwin":
        mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        mem_gb = mem_bytes / (1024**3)
        result["memory_gb"] = round(mem_gb)
        result["platform"] = "mps"
        try:
            chip = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            result["gpu_info"] = chip.stdout.strip()
        except Exception:
            result["gpu_info"] = "Apple Silicon"
        result["preset"] = "mac_studio_m3_ultra" if mem_gb > 200 else "macbook_m4_max"

    elif system in ("Linux", "Windows"):
        try:
            smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if smi.returncode == 0:
                gpus = [l.strip() for l in smi.stdout.strip().split("\n") if l.strip()]
                result["platform"] = "cuda"
                result["gpu_info"] = gpus
                result["memory_gb"] = sum(int(g.split(",")[-1].strip()) // 1024 for g in gpus)
                result["preset"] = "wsl2_dual_a6000"
        except FileNotFoundError:
            pass

    return result


def _generate_jsonl(dataset_dir: str) -> str:
    """Write train_ready/metadata.jsonl with image + caption (+ optional depth path)."""
    train_ready = os.path.join(dataset_dir, "train_ready")
    images_dir = os.path.join(train_ready, "images")
    depth_dir = os.path.join(train_ready, "depth")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"train_ready/images not found in {dataset_dir}. Run dataset preparation first."
        )

    jsonl_path = os.path.join(train_ready, "metadata.jsonl")
    count = 0
    with open(jsonl_path, "w") as f:
        for img_file in sorted(glob.glob(os.path.join(images_dir, "*.png"))):
            stem = os.path.splitext(os.path.basename(img_file))[0]
            cap_file = os.path.join(images_dir, f"{stem}.txt")
            caption = "stereo180sbs, stereoscopic 180 VR side by side half equirectangular"
            if os.path.exists(cap_file):
                with open(cap_file) as cf:
                    caption = cf.read().strip() or caption
            row = {
                "image": os.path.abspath(img_file),
                "text": caption,
            }
            dpath = os.path.join(depth_dir, f"{stem}.png")
            if os.path.isfile(dpath):
                row["depth_image"] = os.path.abspath(dpath)
            f.write(json.dumps(row) + "\n")
            count += 1

    print(
        json.dumps({"type": "log", "message": f"Generated metadata.jsonl with {count} samples"}),
        flush=True,
    )
    if count == 0:
        raise ValueError("No training images in train_ready/images/")
    return jsonl_path


def _build_accelerate_cmd(preset: dict, overrides: dict, jsonl_path: str, output_dir: str) -> list:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    training_script = os.path.join(script_dir, "train_lora_flux.py")

    resolution = int(overrides.get("resolution", preset.get("resolution", 512)))
    train_w = int(overrides.get("train_width", preset.get("train_width", resolution * 2)))
    train_h = int(overrides.get("train_height", preset.get("train_height", resolution)))
    val_w = int(overrides.get("validation_width", preset.get("validation_width", train_w)))
    val_h = int(overrides.get("validation_height", preset.get("validation_height", train_h)))

    batch_size = int(overrides.get("train_batch_size", preset.get("train_batch_size", 1)))
    grad_accum = int(
        overrides.get("gradient_accumulation_steps", preset.get("gradient_accumulation_steps", 4))
    )
    lr = float(overrides.get("learning_rate", preset.get("learning_rate", 1e-4)))
    max_steps = int(overrides.get("max_train_steps", preset.get("max_train_steps", 3000)))
    ckpt_steps = int(overrides.get("checkpointing_steps", preset.get("checkpointing_steps", 500)))
    val_steps = int(overrides.get("validation_steps", preset.get("validation_steps", 250)))
    mixed_prec = preset.get("mixed_precision", "fp16")
    lr_warmup = int(overrides.get("lr_warmup_steps", preset.get("lr_warmup_steps", 100)))
    lr_scheduler = overrides.get("lr_scheduler", preset.get("lr_scheduler", "cosine"))
    seed = int(overrides.get("seed", preset.get("seed", 42)))
    model_name = preset.get("pretrained_model", "black-forest-labs/FLUX.1-dev")
    workers = int(preset.get("dataloader_num_workers", 0))
    rank = int(overrides.get("lora_rank", preset.get("lora_rank", 64)))
    lora_alpha = int(overrides.get("lora_alpha", preset.get("lora_alpha", rank)))
    lora_dropout = float(overrides.get("lora_dropout", preset.get("lora_dropout", 0.0)))
    guidance = float(overrides.get("guidance_scale", preset.get("guidance_scale", 3.5)))
    val_prompt = overrides.get("validation_prompt", preset.get("validation_prompt", ""))
    instance_prompt = overrides.get("instance_prompt", preset.get("instance_prompt", "stereo180sbs"))

    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--mixed_precision",
        mixed_prec,
        "--num_processes",
        "1",
        training_script,
        "--pretrained_model_name_or_path",
        model_name,
        "--jsonl_for_train",
        jsonl_path,
        "--instance_prompt",
        instance_prompt,
        "--output_dir",
        output_dir,
        "--resolution",
        str(resolution),
        "--train_width",
        str(train_w),
        "--train_height",
        str(train_h),
        "--validation_width",
        str(val_w),
        "--validation_height",
        str(val_h),
        "--train_batch_size",
        str(batch_size),
        "--gradient_accumulation_steps",
        str(grad_accum),
        "--learning_rate",
        str(lr),
        "--lr_scheduler",
        lr_scheduler,
        "--lr_warmup_steps",
        str(lr_warmup),
        "--max_train_steps",
        str(max_steps),
        "--checkpointing_steps",
        str(ckpt_steps),
        "--rank",
        str(rank),
        "--lora_alpha",
        str(lora_alpha),
        "--lora_dropout",
        str(lora_dropout),
        "--guidance_scale",
        str(guidance),
        "--dataloader_num_workers",
        str(workers),
        "--seed",
        str(seed),
        "--validation_steps",
        str(val_steps),
        "--num_validation_images",
        "1",
    ]

    if preset.get("gradient_checkpointing", True):
        cmd.append("--gradient_checkpointing")

    if val_prompt:
        cmd.extend(["--validation_prompt", val_prompt])

    report_to = overrides.get("report_to", preset.get("report_to"))
    if report_to:
        cmd.extend(["--report_to", report_to])
    else:
        cmd.extend(["--report_to", "tensorboard"])

    return cmd


_TQDM_PATTERN = re.compile(
    r"(\d+)/(\d+).*?loss[=:]\s*([\d.eE+-]+).*?lr[=:]\s*([\d.eE+-]+)",
    re.IGNORECASE,
)
_TQDM_BASIC = re.compile(r"(\d+)/(\d+)")
_LOSS_ONLY = re.compile(r"(?:train_)?loss[=:\s]+([\d.eE+-]+)", re.IGNORECASE)
_CKPT_PATTERN = re.compile(r"saving.*checkpoint|checkpoint.*saved", re.IGNORECASE)
_VALIDATION_SAMPLE = re.compile(r"Saved validation image:\s*(\S+\.png)", re.IGNORECASE)
_VALIDATION_STEP_RE = re.compile(r"step_(\d+)")
_ERROR_PATTERN = re.compile(r"error|traceback|exception|oom|out of memory", re.IGNORECASE)
_STEPS_PER_SEC = re.compile(r"([\d.]+)\s*(?:s/it|it/s)")
_LOAD_HINTS = re.compile(
    r"loading|downloading|fetching|tokenizer|text.encoder|vae|transformer|"
    r"accelerat|convert|shard|safetensor|checkpoint|config\.json|"
    r"model\.safetensors|Running training",
    re.IGNORECASE,
)
_TRAINING_START = re.compile(r"Running training|\*{3,}.*training", re.IGNORECASE)
_MAP_PATTERN = re.compile(
    r"(?:Map|Filter|Tokeniz|Encod|Preprocess)\S*\s+(\d+)%\|.*?\|\s*([\d,]+)/([\d,]+)",
    re.IGNORECASE,
)


def _write_progress(output_dir: str, **kwargs) -> None:
    progress = {
        "status": kwargs.get("status", "unknown"),
        "current_step": kwargs.get("current_step", 0),
        "total_steps": kwargs.get("total_steps", 0),
        "loss": kwargs.get("loss"),
        "loss_history": kwargs.get("loss_history", []),
        "learning_rate": kwargs.get("learning_rate"),
        "elapsed_seconds": kwargs.get("elapsed_seconds", 0),
        "steps_per_second": kwargs.get("steps_per_second", 0),
        "eta_seconds": kwargs.get("eta_seconds", 0),
        "last_checkpoint": kwargs.get("last_checkpoint"),
        "validation_images": kwargs.get("validation_images", []),
        "platform": kwargs.get("platform"),
        "preset": kwargs.get("preset"),
        "error_message": kwargs.get("error_message"),
        "phase": kwargs.get("phase"),
        "phase_detail": kwargs.get("phase_detail"),
        "preprocess_progress": kwargs.get("preprocess_progress"),
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    progress_path = os.path.join(output_dir, "training_progress.json")
    tmp_path = progress_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp_path, progress_path)


def run_training(dataset_dir: str, preset: dict, overrides: dict | None = None, output_dir: str | None = None) -> bool:
    overrides = overrides or {}
    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(dataset_dir, "training_output", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    train_ready = os.path.join(dataset_dir, "train_ready")
    if not os.path.isdir(train_ready):
        print(json.dumps({"type": "log", "message": "Preparing dataset..."}), flush=True)
        prep = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "prepare_dataset.py"), dataset_dir],
            capture_output=True,
            text=True,
        )
        if prep.returncode != 0:
            _write_progress(
                output_dir,
                status="error",
                error_message=f"Dataset preparation failed: {prep.stderr}",
            )
            return False
        print(prep.stdout, flush=True)

    try:
        jsonl_path = _generate_jsonl(dataset_dir)
    except (FileNotFoundError, ValueError) as e:
        _write_progress(output_dir, status="error", error_message=str(e))
        return False

    preset_name = preset.get("display_name", "custom") if isinstance(preset, dict) else str(preset)
    platform_name = preset.get("platform", "unknown") if isinstance(preset, dict) else "unknown"
    cmd = _build_accelerate_cmd(preset if isinstance(preset, dict) else {}, overrides, jsonl_path, output_dir)

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if isinstance(preset, dict):
        for k, v in preset.get("env", {}).items():
            env[k] = str(v)

    total_steps = int(
        overrides.get(
            "max_train_steps",
            preset.get("max_train_steps", 3000) if isinstance(preset, dict) else 3000,
        )
    )
    _write_progress(
        output_dir,
        status="training",
        total_steps=total_steps,
        platform=platform_name,
        preset=preset_name,
    )

    print(json.dumps({"type": "log", "message": f"Launching LoRA training ({len(cmd)} args)"}), flush=True)

    loss_history: list = []
    validation_images: list = []
    last_checkpoint = None
    last_step = 0
    start_time = time.time()
    phase = "loading"
    phase_detail = "Loading model weights..."
    preprocess_progress = None

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=1,
            universal_newlines=True,
        )
        line_queue: queue.Queue = queue.Queue()

        def _reader():
            for ln in proc.stdout:
                line_queue.put(ln)
            line_queue.put(None)

        threading.Thread(target=_reader, daemon=True).start()

        last_heartbeat = 0.0
        HEARTBEAT = 5.0

        while True:
            try:
                line = line_queue.get(timeout=1)
            except queue.Empty:
                now = time.time()
                if now - last_heartbeat >= HEARTBEAT:
                    last_heartbeat = now
                    status_str = "training" if phase == "training" else ("preprocessing" if phase == "preprocessing" else "loading")
                    _write_progress(
                        output_dir,
                        status=status_str,
                        current_step=last_step,
                        total_steps=total_steps,
                        loss_history=loss_history,
                        last_checkpoint=last_checkpoint,
                        validation_images=validation_images,
                        elapsed_seconds=round(now - start_time),
                        platform=platform_name,
                        preset=preset_name,
                        phase=phase,
                        phase_detail=phase_detail,
                        preprocess_progress=preprocess_progress,
                    )
                continue

            if line is None:
                break

            line = line.rstrip()
            if not line:
                continue
            print(json.dumps({"type": "log", "message": line}), flush=True)

            map_m = _MAP_PATTERN.search(line)
            if map_m:
                pct = int(map_m.group(1))
                cur = int(map_m.group(2).replace(",", ""))
                tot = int(map_m.group(3).replace(",", ""))
                phase = "preprocessing"
                phase_detail = f"Preprocessing dataset: {cur:,}/{tot:,} ({pct}%)"
                preprocess_progress = {"current": cur, "total": tot, "pct": pct}
                _write_progress(
                    output_dir,
                    status="preprocessing",
                    current_step=last_step,
                    total_steps=total_steps,
                    loss_history=loss_history,
                    validation_images=validation_images,
                    elapsed_seconds=round(time.time() - start_time),
                    platform=platform_name,
                    preset=preset_name,
                    phase=phase,
                    phase_detail=phase_detail,
                    preprocess_progress=preprocess_progress,
                )
                continue

            if phase in ("loading", "preprocessing"):
                if _LOAD_HINTS.search(line) and phase == "loading":
                    snippet = line.strip()[:77] + ("..." if len(line.strip()) > 80 else "")
                    phase_detail = snippet
                if _TRAINING_START.search(line):
                    phase = "training"
                    phase_detail = None
                    preprocess_progress = None

            m = _TQDM_PATTERN.search(line)
            if m:
                phase = "training"
                step = int(m.group(1))
                total = int(m.group(2))
                loss = float(m.group(3))
                lr = float(m.group(4))
                last_step = step
                loss_history.append([step, loss])
                if len(loss_history) > 1000:
                    loss_history = loss_history[-1000:]

                elapsed = time.time() - start_time
                sps_match = _STEPS_PER_SEC.search(line)
                steps_per_sec = 0.0
                eta = 0.0
                if sps_match:
                    val = float(sps_match.group(1))
                    steps_per_sec = (1.0 / val if val > 0 and "s/it" in line else val)
                    rem = total - step
                    eta = rem / steps_per_sec if steps_per_sec > 0 else 0.0

                if step % 10 == 0 or step == total:
                    _write_progress(
                        output_dir,
                        status="training",
                        current_step=step,
                        total_steps=total,
                        loss=loss,
                        loss_history=loss_history,
                        learning_rate=lr,
                        elapsed_seconds=round(elapsed),
                        steps_per_second=round(steps_per_sec, 3),
                        eta_seconds=round(eta),
                        last_checkpoint=last_checkpoint,
                        validation_images=validation_images,
                        platform=platform_name,
                        preset=preset_name,
                    )
                print(
                    json.dumps(
                        {
                            "type": "progress",
                            "step": step,
                            "total": total,
                            "loss": loss,
                            "lr": lr,
                            "pct": round((step / total) * 100, 2) if total else 0,
                        }
                    ),
                    flush=True,
                )
                continue

            m_basic = _TQDM_BASIC.search(line)
            if m_basic:
                step = int(m_basic.group(1))
                total = int(m_basic.group(2))
                if step > last_step:
                    last_step = step
                lm = _LOSS_ONLY.search(line)
                if lm:
                    loss_history.append([step, float(lm.group(1))])

            if _CKPT_PATTERN.search(line):
                last_checkpoint = line
                _write_progress(
                    output_dir,
                    status="training",
                    current_step=last_step,
                    total_steps=total_steps,
                    loss_history=loss_history,
                    last_checkpoint=last_checkpoint,
                    validation_images=validation_images,
                    platform=platform_name,
                    preset=preset_name,
                )

            vs = _VALIDATION_SAMPLE.search(line)
            if vs:
                path = vs.group(1)
                sm = _VALIDATION_STEP_RE.search(path)
                step_num = int(sm.group(1)) if sm else last_step
                entry = next((e for e in validation_images if e["step"] == step_num), None)
                if not entry:
                    entry = {"step": step_num, "conditioning": None, "samples": []}
                    validation_images.append(entry)
                entry["samples"].append(path)
                _write_progress(
                    output_dir,
                    status="training",
                    current_step=last_step,
                    total_steps=total_steps,
                    loss_history=loss_history,
                    validation_images=validation_images,
                    platform=platform_name,
                    preset=preset_name,
                )

        proc.wait()
        final = "complete" if proc.returncode == 0 else "error"
        err = None if proc.returncode == 0 else f"exit {proc.returncode}"
        _write_progress(
            output_dir,
            status=final,
            current_step=last_step,
            total_steps=total_steps,
            loss=loss_history[-1][1] if loss_history else None,
            loss_history=loss_history,
            validation_images=validation_images,
            last_checkpoint=last_checkpoint,
            platform=platform_name,
            preset=preset_name,
            error_message=err,
        )
        return proc.returncode == 0

    except KeyboardInterrupt:
        proc.terminate()
        _write_progress(
            output_dir,
            status="cancelled",
            current_step=last_step,
            total_steps=total_steps,
            loss_history=loss_history,
            platform=platform_name,
            preset=preset_name,
        )
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="FLUX LoRA training wrapper")
    parser.add_argument("--dataset", help="Dataset directory")
    parser.add_argument("--preset", help="Hardware preset name")
    parser.add_argument("--output-dir", help="Training output directory")
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--job-file", help="JSON job file (Electron)")
    parser.add_argument("--detect-hardware", action="store_true")

    args = parser.parse_args()

    if args.detect_hardware:
        print(json.dumps(detect_hardware(), indent=2, default=str))
        return

    if args.job_file:
        job = json.loads(Path(args.job_file).read_text())
        dataset_dir = job["dataset_dir"]
        preset_name = job.get("preset")
        output_dir = job.get("output_dir")
        overrides = job.get("overrides", {})
    elif args.dataset:
        dataset_dir = args.dataset
        preset_name = args.preset
        output_dir = args.output_dir
        overrides = {}
        if args.max_train_steps:
            overrides["max_train_steps"] = args.max_train_steps
        if args.learning_rate:
            overrides["learning_rate"] = args.learning_rate
        if args.train_batch_size:
            overrides["train_batch_size"] = args.train_batch_size
        if args.resolution:
            overrides["resolution"] = args.resolution
    else:
        parser.error("Need --dataset or --job-file")
        return

    if not preset_name:
        hw = detect_hardware()
        preset_name = hw.get("preset")
        if not preset_name:
            msg = "Could not auto-detect hardware. Use --preset."
            print(json.dumps({"type": "error", "message": msg}), flush=True)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                _write_progress(output_dir, status="error", error_message=msg)
            sys.exit(1)
        print(json.dumps({"type": "log", "message": f"Auto-detected preset: {preset_name}"}), flush=True)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from train.configs.presets import get_preset

    preset = get_preset(preset_name)
    ok = run_training(dataset_dir, preset, overrides, output_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
