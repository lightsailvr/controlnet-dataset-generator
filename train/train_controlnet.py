#!/usr/bin/env python3
"""Flux ControlNet training wrapper.

Detects hardware platform, generates a JSONL manifest from the prepared
dataset, then launches the diffusers train_controlnet_flux.py script
via `accelerate launch` with progress reporting.

Usage:
    python3 train/train_controlnet.py --dataset /path/to/dataset
    python3 train/train_controlnet.py --dataset /path --preset macbook_m4_max
    python3 train/train_controlnet.py --job-file /tmp/train_job.json
    python3 train/train_controlnet.py --detect-hardware
"""

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
        mem_gb = mem_bytes / (1024 ** 3)
        result["memory_gb"] = round(mem_gb)
        result["platform"] = "mps"

        try:
            chip = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            result["gpu_info"] = chip.stdout.strip()
        except Exception:
            result["gpu_info"] = "Apple Silicon"

        if mem_gb > 200:
            result["preset"] = "mac_studio_m3_ultra"
        elif mem_gb > 100:
            result["preset"] = "macbook_m4_max"
        else:
            result["preset"] = "macbook_m4_max"

    elif system == "Linux" or system == "Windows":
        try:
            smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if smi.returncode == 0:
                gpus = [l.strip() for l in smi.stdout.strip().split("\n") if l.strip()]
                result["platform"] = "cuda"
                result["gpu_info"] = gpus
                result["memory_gb"] = sum(
                    int(g.split(",")[-1].strip()) // 1024
                    for g in gpus
                )
                if len(gpus) >= 2:
                    result["preset"] = "wsl2_dual_a6000"
                else:
                    result["preset"] = "wsl2_dual_a6000"
        except FileNotFoundError:
            pass

    return result


def _generate_jsonl(dataset_dir):
    """Generate metadata.jsonl from train_ready/ for diffusers dataset loading.

    Returns the path to the written JSONL file.
    Each line: {"image": "<abs_path>", "text": "<caption>", "conditioning_image": "<abs_path>"}
    """
    train_ready = os.path.join(dataset_dir, "train_ready")
    target_dir = os.path.join(train_ready, "target")
    cond_dir = os.path.join(train_ready, "conditioning")

    if not os.path.isdir(target_dir) or not os.path.isdir(cond_dir):
        raise FileNotFoundError(
            f"train_ready/target or train_ready/conditioning not found in {dataset_dir}. "
            "Run dataset preparation first."
        )

    jsonl_path = os.path.join(train_ready, "metadata.jsonl")
    count = 0

    with open(jsonl_path, "w") as f:
        for target_file in sorted(glob.glob(os.path.join(target_dir, "*.png"))):
            stem = os.path.splitext(os.path.basename(target_file))[0]
            cond_file = os.path.join(cond_dir, f"{stem}.png")
            caption_file = os.path.join(target_dir, f"{stem}.txt")

            if not os.path.exists(cond_file):
                continue

            caption = "equirectangular panorama"
            if os.path.exists(caption_file):
                with open(caption_file) as cf:
                    caption = cf.read().strip() or caption

            entry = {
                "image": os.path.abspath(target_file),
                "text": caption,
                "conditioning_image": os.path.abspath(cond_file),
            }
            f.write(json.dumps(entry) + "\n")
            count += 1

    print(json.dumps({
        "type": "log",
        "message": f"Generated metadata.jsonl with {count} training pairs"
    }), flush=True)

    if count == 0:
        raise ValueError("No valid training pairs found in train_ready/")

    return jsonl_path


def _build_accelerate_cmd(preset, overrides, jsonl_path, output_dir):
    """Build the `accelerate launch` CLI argument list."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    training_script = os.path.join(script_dir, "train_controlnet_flux.py")

    resolution = overrides.get("resolution", preset.get("resolution", 512))
    batch_size = overrides.get("train_batch_size", preset.get("train_batch_size", 1))
    grad_accum = overrides.get("gradient_accumulation_steps",
                               preset.get("gradient_accumulation_steps", 4))
    lr = overrides.get("learning_rate", preset.get("learning_rate", 1e-4))
    max_steps = overrides.get("max_train_steps", preset.get("max_train_steps", 3000))
    ckpt_steps = overrides.get("checkpointing_steps", preset.get("checkpointing_steps", 500))
    val_steps = overrides.get("validation_steps", preset.get("validation_steps", 250))
    mixed_prec = preset.get("mixed_precision", "fp16")
    num_double = overrides.get("num_double_layers", preset.get("num_double_layers", 4))
    num_single = overrides.get("num_single_layers", preset.get("num_single_layers", 0))
    lr_warmup = overrides.get("lr_warmup_steps", preset.get("lr_warmup_steps", 100))
    lr_scheduler = overrides.get("lr_scheduler", preset.get("lr_scheduler", "cosine"))
    seed = overrides.get("seed", preset.get("seed", 42))
    model_name = preset.get("pretrained_model", "black-forest-labs/FLUX.1-dev")
    dataloader_workers = preset.get("dataloader_num_workers", 0)
    validation_prompt = overrides.get("validation_prompt", "equirectangular panorama")

    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--mixed_precision", mixed_prec,
        "--num_processes", "1",
        training_script,
        "--pretrained_model_name_or_path", model_name,
        "--jsonl_for_train", jsonl_path,
        "--output_dir", output_dir,
        "--resolution", str(resolution),
        "--train_batch_size", str(batch_size),
        "--gradient_accumulation_steps", str(grad_accum),
        "--learning_rate", str(lr),
        "--lr_scheduler", lr_scheduler,
        "--lr_warmup_steps", str(lr_warmup),
        "--max_train_steps", str(max_steps),
        "--checkpointing_steps", str(ckpt_steps),
        "--num_double_layers", str(num_double),
        "--num_single_layers", str(num_single),
        "--dataloader_num_workers", str(dataloader_workers),
        "--seed", str(seed),
    ]

    if preset.get("gradient_checkpointing", True):
        cmd.append("--gradient_checkpointing")

    if val_steps and val_steps > 0:
        # Find a conditioning image to use for validation
        cond_dir = os.path.join(os.path.dirname(jsonl_path), "conditioning")
        val_image = None
        if os.path.isdir(cond_dir):
            cond_images = sorted(glob.glob(os.path.join(cond_dir, "*.png")))
            if cond_images:
                val_image = cond_images[0]

        if val_image:
            cmd.extend(["--validation_steps", str(val_steps)])
            cmd.extend(["--validation_prompt", validation_prompt])
            cmd.extend(["--validation_image", val_image])

    report_to = overrides.get("report_to", preset.get("report_to"))
    if report_to:
        cmd.extend(["--report_to", report_to])

    return cmd


# Regex patterns for parsing diffusers training output
_TQDM_PATTERN = re.compile(
    r"(\d+)/(\d+).*?loss[=:]\s*([\d.eE+-]+).*?lr[=:]\s*([\d.eE+-]+)",
    re.IGNORECASE
)
_TQDM_BASIC = re.compile(r"(\d+)/(\d+)")
_LOSS_ONLY = re.compile(r"(?:train_)?loss[=:\s]+([\d.eE+-]+)", re.IGNORECASE)
_LR_ONLY = re.compile(r"(?:learning_rate|lr)[=:\s]+([\d.eE+-]+)", re.IGNORECASE)
_CKPT_PATTERN = re.compile(r"saving.*checkpoint|checkpoint.*saved", re.IGNORECASE)
_VALIDATION_IMG = re.compile(r"saved.*?(\S+\.png)", re.IGNORECASE)
_ERROR_PATTERN = re.compile(r"error|traceback|exception|oom|out of memory", re.IGNORECASE)
_STEPS_PER_SEC = re.compile(r"([\d.]+)\s*(?:s/it|it/s)")


def run_training(dataset_dir, preset, overrides=None, output_dir=None):
    """Run the full training pipeline using diffusers."""
    overrides = overrides or {}

    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(dataset_dir, "training_output", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Prepare dataset if needed
    train_ready = os.path.join(dataset_dir, "train_ready")
    if not os.path.isdir(train_ready):
        print(json.dumps({"type": "log", "message": "Preparing dataset..."}), flush=True)
        prep_result = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "prepare_dataset.py"),
             dataset_dir],
            capture_output=True, text=True
        )
        if prep_result.returncode != 0:
            _write_progress(output_dir, status="error",
                            error_message=f"Dataset preparation failed: {prep_result.stderr}")
            return False
        print(prep_result.stdout, flush=True)

    # Generate JSONL for diffusers
    try:
        jsonl_path = _generate_jsonl(dataset_dir)
    except (FileNotFoundError, ValueError) as e:
        _write_progress(output_dir, status="error", error_message=str(e))
        return False

    # Build and launch training
    preset_name = preset if isinstance(preset, str) else preset.get("display_name", "custom")
    platform_name = preset.get("platform", "unknown") if isinstance(preset, dict) else "unknown"

    cmd = _build_accelerate_cmd(
        preset if isinstance(preset, dict) else {},
        overrides, jsonl_path, output_dir
    )

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if isinstance(preset, dict):
        for key, val in preset.get("env", {}).items():
            env[key] = str(val)

    print(json.dumps({
        "type": "log",
        "message": f"Launching training: {' '.join(cmd[-10:])}"
    }), flush=True)

    total_steps = overrides.get("max_train_steps",
                                preset.get("max_train_steps", 3000) if isinstance(preset, dict) else 3000)
    _write_progress(output_dir, status="training", total_steps=total_steps,
                    platform=platform_name, preset=preset_name)

    loss_history = []
    validation_images = []
    last_checkpoint = None
    last_step = 0
    start_time = time.time()
    phase = "loading"
    phase_detail = "Loading model weights..."
    preprocess_progress = None

    _LOAD_HINTS = re.compile(
        r"loading|downloading|fetching|tokenizer|text.encoder|vae|transformer|"
        r"controlnet|accelerat|convert|shard|safetensor|checkpoint|config\.json|"
        r"model\.safetensors|Running training",
        re.IGNORECASE,
    )
    _TRAINING_START = re.compile(r"Running training|\*{3,}.*training", re.IGNORECASE)
    _MAP_PATTERN = re.compile(
        r"(?:Map|Filter|Tokeniz|Encod|Preprocess)\S*\s+(\d+)%\|.*?\|\s*([\d,]+)/([\d,]+)",
        re.IGNORECASE,
    )

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, bufsize=1, universal_newlines=True
        )

        line_queue = queue.Queue()
        def _reader():
            for ln in proc.stdout:
                line_queue.put(ln)
            line_queue.put(None)
        threading.Thread(target=_reader, daemon=True).start()

        last_heartbeat = 0
        HEARTBEAT_INTERVAL = 5

        while True:
            try:
                line = line_queue.get(timeout=1)
            except queue.Empty:
                now = time.time()
                if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                    last_heartbeat = now
                    elapsed = now - start_time
                    if phase == "training":
                        status_str = "training"
                    elif phase == "preprocessing":
                        status_str = "preprocessing"
                    else:
                        status_str = "loading"
                    _write_progress(
                        output_dir, status=status_str,
                        current_step=last_step, total_steps=total_steps,
                        loss_history=loss_history, last_checkpoint=last_checkpoint,
                        validation_images=validation_images,
                        elapsed_seconds=round(elapsed),
                        platform=platform_name, preset=preset_name,
                        phase=phase, phase_detail=phase_detail,
                        preprocess_progress=preprocess_progress,
                    )
                continue

            if line is None:
                break

            line = line.rstrip()
            if not line:
                continue

            print(json.dumps({"type": "log", "message": line}), flush=True)

            # Detect Map/preprocessing progress (runs before training starts)
            map_m = _MAP_PATTERN.search(line)
            if map_m:
                pct = int(map_m.group(1))
                current = int(map_m.group(2).replace(",", ""))
                total_map = int(map_m.group(3).replace(",", ""))
                phase = "preprocessing"
                phase_detail = f"Preprocessing dataset: {current:,}/{total_map:,} examples ({pct}%)"
                preprocess_progress = {"current": current, "total": total_map, "pct": pct}

                elapsed = time.time() - start_time
                _write_progress(
                    output_dir, status="preprocessing",
                    current_step=last_step, total_steps=total_steps,
                    loss_history=loss_history, last_checkpoint=last_checkpoint,
                    validation_images=validation_images,
                    elapsed_seconds=round(elapsed),
                    platform=platform_name, preset=preset_name,
                    phase=phase, phase_detail=phase_detail,
                    preprocess_progress=preprocess_progress,
                )
                continue

            if phase in ("loading", "preprocessing"):
                hint = _LOAD_HINTS.search(line)
                if hint and phase == "loading":
                    snippet = line.strip()
                    if len(snippet) > 80:
                        snippet = snippet[:77] + "..."
                    phase_detail = snippet
                if _TRAINING_START.search(line):
                    phase = "training"
                    phase_detail = None
                    preprocess_progress = None

            # Parse tqdm-style progress
            m = _TQDM_PATTERN.search(line)
            if m:
                phase = "training"
                phase_detail = None
                preprocess_progress = None
                step = int(m.group(1))
                total = int(m.group(2))
                loss = float(m.group(3))
                lr = float(m.group(4))
                last_step = step

                elapsed = time.time() - start_time
                loss_history.append([step, loss])
                if len(loss_history) > 1000:
                    loss_history = loss_history[-1000:]

                sps_match = _STEPS_PER_SEC.search(line)
                steps_per_sec = 0
                eta = 0
                if sps_match:
                    val = float(sps_match.group(1))
                    if "s/it" in line:
                        steps_per_sec = 1.0 / val if val > 0 else 0
                    else:
                        steps_per_sec = val
                    remaining = total - step
                    eta = remaining / steps_per_sec if steps_per_sec > 0 else 0

                if step % 10 == 0 or step == total:
                    _write_progress(
                        output_dir, status="training",
                        current_step=step, total_steps=total,
                        loss=loss, loss_history=loss_history,
                        learning_rate=lr, elapsed_seconds=round(elapsed),
                        steps_per_second=round(steps_per_sec, 3),
                        eta_seconds=round(eta),
                        last_checkpoint=last_checkpoint,
                        validation_images=validation_images,
                        platform=platform_name, preset=preset_name,
                    )

                print(json.dumps({
                    "type": "progress", "step": step, "total": total,
                    "loss": loss, "lr": lr,
                    "pct": round((step / total) * 100, 2) if total > 0 else 0,
                }), flush=True)
                continue

            # Basic step match without loss/lr
            m_basic = _TQDM_BASIC.search(line)
            if m_basic and not m:
                step = int(m_basic.group(1))
                total = int(m_basic.group(2))
                if step > last_step:
                    last_step = step
                    loss_m = _LOSS_ONLY.search(line)
                    loss_val = float(loss_m.group(1)) if loss_m else None
                    if loss_val is not None:
                        loss_history.append([step, loss_val])

            # Checkpoint detection
            if _CKPT_PATTERN.search(line):
                last_checkpoint = line
                _write_progress(
                    output_dir, status="training",
                    current_step=last_step, total_steps=total_steps,
                    loss_history=loss_history, last_checkpoint=last_checkpoint,
                    validation_images=validation_images,
                    platform=platform_name, preset=preset_name,
                )

            # Validation image detection
            vm = _VALIDATION_IMG.search(line)
            if vm:
                validation_images.append(vm.group(1))

        proc.wait()

        final_status = "complete" if proc.returncode == 0 else "error"
        final_error = None if proc.returncode == 0 else f"Process exited with code {proc.returncode}"

        _write_progress(
            output_dir, status=final_status,
            current_step=last_step, total_steps=total_steps,
            loss=loss_history[-1][1] if loss_history else None,
            loss_history=loss_history,
            validation_images=validation_images,
            last_checkpoint=last_checkpoint,
            platform=platform_name, preset=preset_name,
            error_message=final_error,
        )

        return proc.returncode == 0

    except KeyboardInterrupt:
        proc.terminate()
        _write_progress(output_dir, status="cancelled",
                        current_step=last_step, total_steps=total_steps,
                        loss_history=loss_history, platform=platform_name,
                        preset=preset_name)
        return False
    except Exception as e:
        _write_progress(output_dir, status="error",
                        error_message=str(e), platform=platform_name,
                        preset=preset_name)
        raise


def _write_progress(output_dir, **kwargs):
    """Write training_progress.json atomically."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Flux ControlNet training wrapper (diffusers backend)"
    )
    parser.add_argument("--dataset", help="Path to dataset directory")
    parser.add_argument("--preset", help="Hardware preset name")
    parser.add_argument("--output-dir", help="Training output directory")
    parser.add_argument("--max-train-steps", type=int, help="Override max training steps")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--train-batch-size", type=int, help="Override batch size")
    parser.add_argument("--resolution", type=int, help="Override resolution")
    parser.add_argument("--job-file", help="JSON job file (Electron mode)")
    parser.add_argument("--detect-hardware", action="store_true",
                        help="Detect hardware and print info")

    args = parser.parse_args()

    if args.detect_hardware:
        info = detect_hardware()
        print(json.dumps(info, indent=2, default=str))
        return

    # Resolve config from job file or CLI args
    if args.job_file:
        with open(args.job_file) as f:
            job = json.load(f)
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
        parser.error("Either --dataset or --job-file is required")
        return

    # Auto-detect preset if not specified
    if not preset_name:
        hw = detect_hardware()
        preset_name = hw.get("preset")
        if not preset_name:
            error_msg = "Could not auto-detect hardware. Use --preset to specify."
            print(json.dumps({"type": "error", "message": error_msg}), flush=True)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                _write_progress(output_dir, status="error", error_message=error_msg)
            sys.exit(1)
        print(json.dumps({
            "type": "log",
            "message": f"Auto-detected hardware preset: {preset_name}"
        }), flush=True)

    # Load preset config
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from train.configs.presets import PRESETS, get_preset
    preset = get_preset(preset_name)

    success = run_training(dataset_dir, preset, overrides, output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
