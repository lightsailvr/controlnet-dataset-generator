"""Generate SimpleTuner config files from hardware presets."""

import json
import os
import shutil


def generate_config(preset, overrides, run_dir, multidatabackend_path):
    """Generate SimpleTuner config.json and accelerate config in run_dir.

    Args:
        preset: Dict from presets.py
        overrides: Dict of user config overrides
        run_dir: Output directory for config files
        multidatabackend_path: Path to the multidatabackend.json from prepare_dataset
    """
    os.makedirs(run_dir, exist_ok=True)

    config = _build_simpletuner_config(preset, overrides)
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Copy multidatabackend.json into run dir
    mdb_dest = os.path.join(run_dir, "multidatabackend.json")
    shutil.copy2(multidatabackend_path, mdb_dest)

    # Generate accelerate config
    accel_config = _build_accelerate_config(preset, overrides)
    accel_path = os.path.join(run_dir, "accelerate_config.yaml")
    _write_yaml(accel_path, accel_config)

    # Generate DeepSpeed config if needed
    if "deepspeed" in preset:
        ds_config = _build_deepspeed_config(preset)
        ds_path = os.path.join(run_dir, "deepspeed_config.json")
        with open(ds_path, "w") as f:
            json.dump(ds_config, f, indent=2)

    return config_path


def _build_simpletuner_config(preset, overrides):
    """Build SimpleTuner config.json from preset + overrides."""
    config = {
        "model_type": preset["model_type"],
        "controlnet": preset["controlnet"],
        "model_family": preset["model_family"],
        "pretrained_model_name_or_path": preset["pretrained_model"],
        "resolution": overrides.get("resolution", preset["resolution"]),
        "resolution_type": preset.get("resolution_type", "pixel_area"),
        "train_batch_size": overrides.get("train_batch_size", preset["train_batch_size"]),
        "gradient_accumulation_steps": overrides.get(
            "gradient_accumulation_steps", preset["gradient_accumulation_steps"]
        ),
        "mixed_precision": preset["mixed_precision"],
        "base_model_precision": preset.get("base_model_precision", "bf16"),
        "optimizer": overrides.get("optimizer", preset["optimizer"]),
        "learning_rate": overrides.get("learning_rate", preset["learning_rate"]),
        "lr_scheduler": overrides.get("lr_scheduler", preset["lr_scheduler"]),
        "lr_warmup_steps": overrides.get("lr_warmup_steps", preset["lr_warmup_steps"]),
        "max_train_steps": overrides.get("max_train_steps", preset["max_train_steps"]),
        "checkpointing_steps": overrides.get(
            "checkpointing_steps", preset["checkpointing_steps"]
        ),
        "validation_steps": overrides.get("validation_steps", preset["validation_steps"]),
        "validation_prompt": overrides.get(
            "validation_prompt", "equirectangular panorama"
        ),
        "num_validation_images": overrides.get("num_validation_images", 4),
        "gradient_checkpointing": preset.get("gradient_checkpointing", True),
        "max_grad_norm": preset.get("max_grad_norm", 1.0),
        "adam_weight_decay": preset.get("adam_weight_decay", 0.01),
        "dataloader_num_workers": preset["dataloader_num_workers"],
        "seed": overrides.get("seed", preset["seed"]),
        "report_to": overrides.get("report_to", "wandb"),
        "tracker_project_name": overrides.get(
            "tracker_project_name", "flux-controlnet-equirect"
        ),
    }

    if preset.get("flash_attention"):
        config["enable_flash_attention"] = True

    return config


def _build_accelerate_config(preset, overrides):
    """Build accelerate config based on platform."""
    platform = preset["platform"]

    if platform == "cuda" and preset.get("num_gpus", 1) > 1:
        config = {
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "DEEPSPEED" if "deepspeed" in preset else "MULTI_GPU",
            "mixed_precision": preset["mixed_precision"],
            "num_processes": preset.get("num_gpus", 2),
            "num_machines": 1,
        }
        if "deepspeed" in preset:
            config["deepspeed_config"] = {
                "zero_stage": preset["deepspeed"]["zero_stage"],
                "gradient_accumulation_steps": overrides.get(
                    "gradient_accumulation_steps",
                    preset["gradient_accumulation_steps"],
                ),
                "gradient_clipping": preset["deepspeed"].get("gradient_clipping", 1.0),
                "offload_optimizer_device": preset["deepspeed"].get(
                    "offload_optimizer_device", "none"
                ),
                "offload_param_device": preset["deepspeed"].get(
                    "offload_param_device", "none"
                ),
            }
    else:
        # Single device (Mac or single GPU)
        config = {
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "NO",
            "mixed_precision": preset["mixed_precision"],
            "num_processes": 1,
            "num_machines": 1,
        }

    return config


def _build_deepspeed_config(preset):
    """Build standalone deepspeed_config.json."""
    ds = preset["deepspeed"]
    return {
        "zero_optimization": {
            "stage": ds["zero_stage"],
            "offload_optimizer": {
                "device": ds.get("offload_optimizer_device", "none"),
            },
            "offload_param": {
                "device": ds.get("offload_param_device", "none"),
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "gradient_clipping": ds.get("gradient_clipping", 1.0),
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "bf16": {
            "enabled": True,
        },
    }


def _write_yaml(path, data, indent=0):
    """Write a simple dict as YAML without requiring pyyaml."""
    lines = []
    _yaml_lines(data, lines, indent)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _yaml_lines(data, lines, indent):
    """Recursively convert dict to YAML lines."""
    prefix = "  " * indent
    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, dict):
                lines.append(f"{prefix}{key}:")
                _yaml_lines(val, lines, indent + 1)
            elif isinstance(val, bool):
                lines.append(f"{prefix}{key}: {'true' if val else 'false'}")
            elif isinstance(val, (int, float)):
                lines.append(f"{prefix}{key}: {val}")
            elif isinstance(val, str):
                lines.append(f"{prefix}{key}: {val}")
            else:
                lines.append(f"{prefix}{key}: {val}")
