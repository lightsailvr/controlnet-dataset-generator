"""Hardware presets for Flux ControlNet training via SimpleTuner."""

# Base config shared by all Mac presets
_MAC_BASE = {
    "platform": "mps",
    "model_type": "lora",
    "controlnet": True,
    "model_family": "flux",
    "pretrained_model": "black-forest-labs/FLUX.1-dev",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "mixed_precision": "bf16",
    "base_model_precision": "bf16",
    "optimizer": "adamw",
    "learning_rate": 3e-5,
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 100,
    "max_train_steps": 50000,
    "checkpointing_steps": 2000,
    "validation_steps": 500,
    "gradient_checkpointing": True,
    "max_grad_norm": 1.0,
    "adam_weight_decay": 0.01,
    "dataloader_num_workers": 2,
    "seed": 42,
    "forbidden": [
        "bitsandbytes", "fp8", "int8", "nf4", "quantize",
        "enable_group_offload", "ramtorch", "group_offload_use_stream",
    ],
}

PRESETS = {
    "mac_studio_m3_ultra": {
        **_MAC_BASE,
        "display_name": "Mac Studio M3 Ultra (256GB)",
        "train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "env": {
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.8",
            "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.6",
            "SIMPLETUNER_LOG_LEVEL": "INFO",
        },
    },
    "macbook_m4_max": {
        **_MAC_BASE,
        "display_name": "MacBook M4 Max (128GB)",
        "train_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "env": {
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.7",
            "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.5",
            "SIMPLETUNER_LOG_LEVEL": "INFO",
        },
    },
    "wsl2_dual_a6000": {
        "display_name": "Dual NVIDIA A6000 (2x 48GB, WSL2)",
        "platform": "cuda",
        "model_type": "lora",
        "controlnet": True,
        "model_family": "flux",
        "pretrained_model": "black-forest-labs/FLUX.1-dev",
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "num_gpus": 2,
        "mixed_precision": "bf16",
        "base_model_precision": "bf16",
        "optimizer": "adamw8bit",
        "learning_rate": 1e-5,
        "lr_scheduler": "constant",
        "lr_warmup_steps": 500,
        "max_train_steps": 50000,
        "checkpointing_steps": 2000,
        "validation_steps": 500,
        "gradient_checkpointing": True,
        "max_grad_norm": 1.0,
        "adam_weight_decay": 0.01,
        "flash_attention": True,
        "dataloader_num_workers": 4,
        "seed": 42,
        "deepspeed": {
            "zero_stage": 2,
            "offload_optimizer_device": "none",
            "offload_param_device": "none",
            "gradient_clipping": 1.0,
        },
        "env": {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        "forbidden": [],
    },
}


def get_preset(name):
    """Get a preset by name. Raises KeyError if not found."""
    return PRESETS[name]


def list_presets():
    """Return list of (name, display_name) tuples."""
    return [(k, v["display_name"]) for k, v in PRESETS.items()]


def validate_config(preset_name, overrides):
    """Check that no forbidden options are present in overrides."""
    preset = PRESETS[preset_name]
    forbidden = preset.get("forbidden", [])
    violations = []
    for key, val in overrides.items():
        for f in forbidden:
            if f in str(key).lower() or f in str(val).lower():
                violations.append(f"'{key}={val}' conflicts with forbidden option '{f}'")
    if violations:
        raise ValueError(
            f"Preset '{preset_name}' forbids the following on {preset['platform']}:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )
