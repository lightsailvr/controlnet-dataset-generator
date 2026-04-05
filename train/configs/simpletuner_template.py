"""Build accelerate launch CLI args from preset + overrides.

This replaces the old SimpleTuner config.json / accelerate_config.yaml
generation. All configuration is now passed as CLI arguments to the
diffusers train_lora_flux.py script (this repo trains LoRA; SimpleTuner remains optional).
"""

import os
import sys


def build_launch_args(preset, overrides, jsonl_path, output_dir):
    """Build the full `accelerate launch` command list.

    Args:
        preset: Dict from presets.py
        overrides: Dict of user config overrides
        jsonl_path: Path to the metadata.jsonl file
        output_dir: Training output directory

    Returns:
        List of command strings ready for subprocess.Popen
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    training_script = os.path.join(script_dir, "train_lora_flux.py")

    def _get(key, default=None):
        return overrides.get(key, preset.get(key, default))

    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--mixed_precision", _get("mixed_precision", "fp16"),
        "--num_processes", "1",
        training_script,
        "--pretrained_model_name_or_path", _get("pretrained_model", "black-forest-labs/FLUX.1-dev"),
        "--jsonl_for_train", jsonl_path,
        "--output_dir", output_dir,
        "--resolution", str(_get("resolution", 512)),
        "--train_batch_size", str(_get("train_batch_size", 1)),
        "--gradient_accumulation_steps", str(_get("gradient_accumulation_steps", 4)),
        "--learning_rate", str(_get("learning_rate", 1e-4)),
        "--lr_scheduler", _get("lr_scheduler", "cosine"),
        "--lr_warmup_steps", str(_get("lr_warmup_steps", 100)),
        "--max_train_steps", str(_get("max_train_steps", 3000)),
        "--checkpointing_steps", str(_get("checkpointing_steps", 500)),
        "--num_double_layers", str(_get("num_double_layers", 4)),
        "--num_single_layers", str(_get("num_single_layers", 0)),
        "--dataloader_num_workers", str(_get("dataloader_num_workers", 0)),
        "--seed", str(_get("seed", 42)),
    ]

    if _get("gradient_checkpointing", True):
        cmd.append("--gradient_checkpointing")

    val_steps = _get("validation_steps")
    if val_steps and int(val_steps) > 0:
        import glob as _glob
        cond_dir = os.path.join(os.path.dirname(jsonl_path), "conditioning")
        val_image = None
        if os.path.isdir(cond_dir):
            cond_images = sorted(_glob.glob(os.path.join(cond_dir, "*.png")))
            if cond_images:
                val_image = cond_images[0]
        if val_image:
            cmd.extend(["--validation_steps", str(val_steps)])
            cmd.extend(["--validation_prompt", _get("validation_prompt", "equirectangular panorama")])
            cmd.extend(["--validation_image", val_image])

    report_to = _get("report_to")
    if report_to:
        cmd.extend(["--report_to", report_to])

    return cmd
