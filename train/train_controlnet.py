#!/usr/bin/env python3
"""Flux ControlNet training wrapper.

Detects hardware platform, configures SimpleTuner, and launches training
with unified progress reporting.

Usage:
    python3 train/train_controlnet.py --dataset /path/to/dataset
    python3 train/train_controlnet.py --dataset /path --preset wsl2_dual_a6000
    python3 train/train_controlnet.py --job-file /tmp/train_job.json
    python3 train/train_controlnet.py --detect-hardware
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

# Add parent dir to path so we can import train modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.configs.presets import PRESETS, get_preset, validate_config
from train.configs.simpletuner_template import generate_config
from train.progress.parser import ProgressParser


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
        # macOS — check memory to distinguish M3 Ultra vs M4 Max
        mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        mem_gb = mem_bytes / (1024 ** 3)
        result["memory_gb"] = round(mem_gb)
        result["platform"] = "mps"

        # Try to get chip name
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
            result["preset"] = "macbook_m4_max"  # Conservative default

    elif system == "Linux" or system == "Windows":
        # Check for NVIDIA GPUs
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
                    result["preset"] = "wsl2_dual_a6000"  # Use as base even for single GPU
        except FileNotFoundError:
            pass

    # Check NVLink topology if CUDA
    if result["platform"] == "cuda":
        result["nvlink"] = _check_nvlink()
        result["tcc_status"] = _check_tcc()

    return result


def _check_nvlink():
    """Check NVLink connectivity between GPUs."""
    try:
        topo = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True, text=True, timeout=10
        )
        if topo.returncode == 0:
            output = topo.stdout
            has_nvlink = "NV" in output and "PIX" not in output.split("\n")[1]
            return {"available": has_nvlink, "topology": output.strip()}
    except Exception:
        pass
    return {"available": False, "topology": None}


def _check_tcc():
    """Check TCC mode status on GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,driver_model.current",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            modes = {}
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2:
                    modes[int(parts[0])] = parts[1]
            return modes
    except Exception:
        pass
    return {}


def check_simpletuner():
    """Verify SimpleTuner is installed."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import simpletuner; print('ok')"],
            capture_output=True, text=True, timeout=15
        )
        return result.stdout.strip() == "ok"
    except Exception:
        return False


def run_training(dataset_dir, preset_name, overrides=None, output_dir=None):
    """Run the full training pipeline."""
    overrides = overrides or {}
    preset = get_preset(preset_name)

    # Validate no forbidden options
    validate_config(preset_name, overrides)

    # Resolve output directory
    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(dataset_dir, "training_output", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Prepare dataset if needed
    train_ready = os.path.join(dataset_dir, "train_ready")
    mdb_path = os.path.join(train_ready, "multidatabackend.json")
    if not os.path.exists(mdb_path):
        print(json.dumps({"type": "log", "message": "Preparing dataset for SimpleTuner..."}),
              flush=True)
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

    # Generate SimpleTuner config
    run_dir = os.path.join(output_dir, "config")
    config_path = generate_config(preset, overrides, run_dir, mdb_path)

    # Set up environment
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    for key, val in preset.get("env", {}).items():
        env[key] = val

    # Build SimpleTuner launch command
    cmd = [sys.executable, "-m", "simpletuner.train", "--config-dir", run_dir]

    print(json.dumps({
        "type": "log",
        "message": f"Launching SimpleTuner with preset '{preset_name}' on {preset['platform']}"
    }), flush=True)

    # Write initial progress
    total_steps = overrides.get("max_train_steps", preset["max_train_steps"])
    _write_progress(output_dir, status="training", total_steps=total_steps,
                    platform=preset["platform"], preset=preset_name)

    # Spawn training process
    parser = ProgressParser()
    loss_history = []
    validation_images = []
    last_checkpoint = None

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, bufsize=1, universal_newlines=True
        )

        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue

            event = parser.parse_line(line)
            if not event:
                continue

            # Forward to stdout for Electron
            print(json.dumps(event), flush=True)

            # Update progress file periodically
            if event["type"] == "progress":
                step = event.get("step", 0)
                loss = event.get("loss")
                if loss is not None:
                    loss_history.append([step, loss])
                    # Keep last 1000 points
                    if len(loss_history) > 1000:
                        loss_history = loss_history[-1000:]

                if step % 10 == 0 or step == total_steps:
                    _write_progress(
                        output_dir,
                        status="training",
                        current_step=step,
                        total_steps=total_steps,
                        loss=loss,
                        loss_history=loss_history,
                        learning_rate=event.get("lr"),
                        elapsed_seconds=event.get("elapsed", 0),
                        steps_per_second=event.get("steps_per_sec", 0),
                        eta_seconds=event.get("eta", 0),
                        last_checkpoint=last_checkpoint,
                        validation_images=validation_images,
                        platform=preset["platform"],
                        preset=preset_name,
                    )

            elif event["type"] == "validation":
                img_path = event.get("image")
                if img_path:
                    validation_images.append(img_path)

            elif event["type"] == "checkpoint":
                last_checkpoint = event.get("message", "")

            elif event["type"] == "error":
                _write_progress(
                    output_dir, status="error",
                    error_message=event.get("message", "Unknown error"),
                    current_step=parser.last_step,
                    total_steps=total_steps,
                    loss_history=loss_history,
                    platform=preset["platform"],
                    preset=preset_name,
                )

        proc.wait()

        final_status = "complete" if proc.returncode == 0 else "error"
        final_error = None if proc.returncode == 0 else f"Process exited with code {proc.returncode}"

        _write_progress(
            output_dir,
            status=final_status,
            current_step=parser.last_step,
            total_steps=total_steps,
            loss=loss_history[-1][1] if loss_history else None,
            loss_history=loss_history,
            validation_images=validation_images,
            last_checkpoint=last_checkpoint,
            platform=preset["platform"],
            preset=preset_name,
            error_message=final_error,
        )

        return proc.returncode == 0

    except KeyboardInterrupt:
        proc.terminate()
        _write_progress(output_dir, status="cancelled",
                        current_step=parser.last_step, total_steps=total_steps,
                        loss_history=loss_history, platform=preset["platform"],
                        preset=preset_name)
        return False
    except Exception as e:
        _write_progress(output_dir, status="error",
                        error_message=str(e), platform=preset["platform"],
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
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }

    progress_path = os.path.join(output_dir, "training_progress.json")
    tmp_path = progress_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp_path, progress_path)


def main():
    parser = argparse.ArgumentParser(
        description="Flux ControlNet training wrapper"
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
            "message": f"Auto-detected hardware: {PRESETS[preset_name]['display_name']}"
        }), flush=True)

        # Print warnings for NVIDIA
        if hw.get("platform") == "cuda":
            nvlink = hw.get("nvlink", {})
            if not nvlink.get("available"):
                print(json.dumps({
                    "type": "log",
                    "message": "WARNING: NVLink not detected. Training will use PCIe for GPU sync (slower)."
                }), flush=True)

            tcc = hw.get("tcc_status", {})
            for gpu_id, mode in tcc.items():
                if gpu_id > 0 and "WDDM" in mode:
                    print(json.dumps({
                        "type": "log",
                        "message": f"WARNING: GPU {gpu_id} is in WDDM mode. "
                                   f"Consider TCC mode: nvidia-smi -i {gpu_id} -dm 1"
                    }), flush=True)

    # Verify SimpleTuner is installed
    if not check_simpletuner():
        preset = get_preset(preset_name)
        req_file = "requirements_mac.txt" if preset["platform"] == "mps" else "requirements_nvidia.txt"
        error_msg = f"SimpleTuner is not installed. Run: pip3 install -r train/{req_file}"
        print(json.dumps({"type": "error", "message": error_msg}), flush=True)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            _write_progress(output_dir, status="error", error_message=error_msg)
        sys.exit(1)

    success = run_training(dataset_dir, preset_name, overrides, output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
