"""Hardware presets for Flux ControlNet training via diffusers."""

_MAC_BASE = {
    "platform": "mps",
    "pretrained_model": "black-forest-labs/FLUX.1-dev",
    "resolution": 512,
    "mixed_precision": "fp16",
    "learning_rate": 1e-4,
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 100,
    "max_train_steps": 3000,
    "checkpointing_steps": 500,
    "validation_steps": 250,
    "gradient_checkpointing": True,
    "num_double_layers": 4,
    "num_single_layers": 0,
    "dataloader_num_workers": 0,
    "seed": 42,
    "env": {
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
    },
}

PRESETS = {
    "mac_studio_m3_ultra": {
        **_MAC_BASE,
        "display_name": "Mac Studio M3 Ultra (256GB)",
        "train_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "env": {
            **_MAC_BASE["env"],
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.8",
            "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.6",
        },
    },
    "macbook_m4_max": {
        **_MAC_BASE,
        "display_name": "MacBook M4 Max (128GB)",
        "train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "env": {
            **_MAC_BASE["env"],
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
        },
    },
    "wsl2_dual_a6000": {
        "display_name": "Dual NVIDIA A6000 (2x 48GB, WSL2)",
        "platform": "cuda",
        "pretrained_model": "black-forest-labs/FLUX.1-dev",
        "resolution": 512,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "mixed_precision": "bf16",
        "learning_rate": 1e-5,
        "lr_scheduler": "constant",
        "lr_warmup_steps": 500,
        "max_train_steps": 50000,
        "checkpointing_steps": 2000,
        "validation_steps": 500,
        "gradient_checkpointing": True,
        "num_double_layers": 4,
        "num_single_layers": 0,
        "dataloader_num_workers": 4,
        "seed": 42,
        "env": {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    },
}


def get_preset(name):
    """Get a preset by name. Raises KeyError if not found."""
    return PRESETS[name]


def list_presets():
    """Return list of (name, display_name) tuples."""
    return [(k, v["display_name"]) for k, v in PRESETS.items()]
