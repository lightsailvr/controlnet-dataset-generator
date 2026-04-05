#!/usr/bin/env python3
"""Disparity / depth proxy from stereoscopic side-by-side images.

Backends
--------
- **foundation_stereo** (PC, NVIDIA CUDA): NVlabs FoundationStereo — high quality,
  zero-shot stereo. Requires a clone of https://github.com/NVlabs/FoundationStereo,
  conda env, weights under ``pretrained_models/``, and ``FOUNDATION_STEREO_ROOT`` (or
  ``fs_root`` / ``foundationStereoRoot`` in the job config).
- **sgbm**: OpenCV StereoSGBM — fast CPU preview; noisy on equirect, not recommended
  for training.

Output: single-channel uint8 PNG (lighter = nearer in disparity space), same width
as one SBS half (H × W/2).
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError("depth_extractor requires opencv-python: pip install opencv-python-headless") from e

BackendName = Literal["auto", "foundation_stereo", "sgbm"]

_FS_CACHE: dict[str, Any] | None = None


def split_sbs_halves(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a side-by-side stereo frame into left and right BGR views."""
    h, w = bgr.shape[:2]
    half = w // 2
    left = bgr[:, :half].copy()
    right = bgr[:, half : half * 2].copy()
    return left, right


def compute_disparity_map_sgbm(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    *,
    num_disparities: int = 128,
    block_size: int = 5,
) -> np.ndarray:
    """Compute disparity (left view) using Semi-Global Block Matching."""
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    nd = max(16, (num_disparities // 16) * 16)

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=nd,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disp[disp < 0] = 0
    return disp


def disparity_to_uint8(disp: np.ndarray) -> np.ndarray:
    """Normalize disparity to 0–255 uint8 for saving as depth proxy."""
    valid = np.isfinite(disp) & (disp > 0)
    out = np.zeros(disp.shape[:2], dtype=np.uint8)
    if not np.any(valid):
        return out
    d = disp[valid]
    lo, hi = float(np.min(d)), float(np.max(d))
    if hi <= lo:
        return out
    scaled = np.zeros_like(disp, dtype=np.float64)
    scaled[valid] = (disp[valid] - lo) / (hi - lo + 1e-8)
    scaled = np.clip(scaled, 0, 1)
    out = (scaled * 255.0).astype(np.uint8)
    out[~valid] = 0
    return out


def _resolve_fs_root(fs_root: str | None) -> Path | None:
    for candidate in (
        fs_root,
        os.environ.get("FOUNDATION_STEREO_ROOT"),
        os.environ.get("FOUNDATION_STEREO_HOME"),
    ):
        if candidate and str(candidate).strip():
            p = Path(candidate).expanduser().resolve()
            if p.is_dir():
                return p
    return None


def _resolve_ckpt(fs_root: Path, fs_ckpt: str | None) -> Path:
    if fs_ckpt and str(fs_ckpt).strip():
        p = Path(fs_ckpt).expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"FoundationStereo checkpoint not found: {p}")
    default = fs_root / "pretrained_models" / "23-51-11" / "model_best_bp2.pth"
    if default.is_file():
        return default
    alt = fs_root / "pretrained_models" / "11-33-40" / "model_best_bp2.pth"
    if alt.is_file():
        return alt
    raise FileNotFoundError(
        "No FoundationStereo weights found. Download a model folder (e.g. 23-51-11) into "
        f"{fs_root / 'pretrained_models'}/ — see NVlabs FoundationStereo readme."
    )


def _foundation_stereo_available(fs_root: Path | None, ckpt_path: Path | None) -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
    except ImportError:
        return False
    if fs_root is None or ckpt_path is None or not ckpt_path.is_file():
        return False
    cfg = ckpt_path.parent / "cfg.yaml"
    return cfg.is_file()


def resolve_depth_backend(
    backend: str,
    *,
    fs_root: str | None = None,
    fs_ckpt: str | None = None,
) -> Literal["foundation_stereo", "sgbm"]:
    """Pick concrete backend for ``auto``."""
    b = (backend or "auto").strip().lower()
    if b == "sgbm":
        return "sgbm"
    if b == "foundation_stereo":
        return "foundation_stereo"

    root = _resolve_fs_root(fs_root)
    try:
        ckpt = _resolve_ckpt(root, fs_ckpt) if root else None
    except FileNotFoundError:
        ckpt = None
    if root and _foundation_stereo_available(root, ckpt):
        return "foundation_stereo"
    return "sgbm"


def _load_foundation_stereo_model(fs_root: Path, ckpt_path: Path):
    """Import FoundationStereo and load weights (cached per (root, ckpt))."""
    global _FS_CACHE
    cache_key = (str(fs_root), str(ckpt_path))
    if _FS_CACHE is not None and _FS_CACHE.get("key") == cache_key:
        return _FS_CACHE["model"], _FS_CACHE["args"]

    root_str = str(fs_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    import torch
    from omegaconf import OmegaConf

    from core.foundation_stereo import FoundationStereo  # type: ignore  # noqa: E402

    cfg_path = ckpt_path.parent / "cfg.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing cfg.yaml next to checkpoint: {cfg_path}")

    cfg = OmegaConf.load(str(cfg_path))
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"

    # Defaults expected by scripts/run_demo.py (only fields we need at runtime)
    demo_ns = SimpleNamespace(
        scale=1.0,
        hiera=0,
        valid_iters=32,
        z_far=10.0,
        get_pc=0,
        remove_invisible=1,
        denoise_cloud=1,
        denoise_nb_points=30,
        denoise_radius=0.03,
        left_file="",
        right_file="",
        intrinsic_file="",
        ckpt_dir=str(ckpt_path),
        out_dir="",
    )
    for k in demo_ns.__dict__:
        cfg[k] = getattr(demo_ns, k)

    args = OmegaConf.create(cfg)
    model = FoundationStereo(args)
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cuda", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cuda")
    model.load_state_dict(ckpt["model"])
    model.cuda()
    model.eval()

    _FS_CACHE = {"key": cache_key, "model": model, "args": args}
    return model, args


def _depth_foundation_stereo(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    *,
    fs_root: str | None,
    fs_ckpt: str | None,
    fs_scale: float,
    fs_hiera: int,
    fs_valid_iters: int,
) -> np.ndarray:
    import torch
    from core.utils.utils import InputPadder  # type: ignore  # noqa: E402

    if not torch.cuda.is_available():
        raise RuntimeError(
            "FoundationStereo requires NVIDIA CUDA. Use a PC with a CUDA GPU and the "
            "foundation_stereo conda env, or set depth backend to sgbm / auto on machines without CUDA."
        )

    root = _resolve_fs_root(fs_root)
    if root is None:
        raise RuntimeError(
            "FoundationStereo repo path not found. Set FOUNDATION_STEREO_ROOT or pass foundationStereoRoot "
            "(clone https://github.com/NVlabs/FoundationStereo and run scripts/setup_depth_env.sh)."
        )

    ckpt_path = _resolve_ckpt(root, fs_ckpt)
    model, args = _load_foundation_stereo_model(root, ckpt_path)

    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
    if left_rgb.shape != right_rgb.shape:
        right_rgb = cv2.resize(
            right_rgb,
            (left_rgb.shape[1], left_rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    scale = float(fs_scale)
    if scale <= 0 or scale > 1.0:
        raise ValueError("fs_scale must be in (0, 1]")
    if scale < 1.0:
        left_rgb = cv2.resize(left_rgb, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_LINEAR)
        right_rgb = cv2.resize(right_rgb, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_LINEAR)

    h, w = left_rgb.shape[:2]
    img0 = torch.as_tensor(left_rgb).cuda().float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(right_rgb).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)

    iters = int(fs_valid_iters)
    hiera = int(fs_hiera)

    try:
        autocast_ctx = torch.autocast("cuda", enabled=True)
    except (TypeError, AttributeError):
        autocast_ctx = torch.cuda.amp.autocast(True)

    with autocast_ctx:
        if not hiera:
            disp_t = model.forward(img0, img1, iters=iters, test_mode=True)
        else:
            disp_t = model.run_hierachical(img0, img1, iters=iters, test_mode=True, low_memory=False, small_ratio=0.5)

    disp_t = padder.unpad(disp_t.float())
    disp = disp_t.detach().cpu().numpy().reshape(h, w)
    disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
    return disparity_to_uint8(disp)


def depth_from_sbs_bgr(
    sbs_bgr: np.ndarray,
    *,
    backend: BackendName | str = "auto",
    fs_root: str | None = None,
    fs_ckpt: str | None = None,
    fs_scale: float = 1.0,
    fs_hiera: int = 0,
    fs_valid_iters: int = 32,
    sgbm_num_disparities: int = 128,
    sgbm_block_size: int = 5,
    _auto_warned: list[bool] | None = None,
) -> np.ndarray:
    """SBS BGR full frame → uint8 disparity image (H × W//2, single channel).

    Parameters
    ----------
    backend
        ``auto`` → FoundationStereo if CUDA + repo + weights exist, else SGBM.
    fs_root
        Path to FoundationStereo clone; falls back to ``FOUNDATION_STEREO_ROOT``.
    fs_ckpt
        Path to ``model_best_bp2.pth``; default ``pretrained_models/23-51-11/``.
    """
    left, right = split_sbs_halves(sbs_bgr)
    if left.shape != right.shape:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR)

    root = _resolve_fs_root(fs_root)
    ckpt_path: Path | None = None
    if root:
        try:
            ckpt_path = _resolve_ckpt(root, fs_ckpt)
        except FileNotFoundError:
            ckpt_path = None

    requested = (backend or "auto").strip().lower()
    use_fs = resolve_depth_backend(requested, fs_root=fs_root, fs_ckpt=fs_ckpt) == "foundation_stereo"

    if requested == "auto" and not use_fs and _auto_warned is not None and not _auto_warned[0]:
        warnings.warn(
            "Depth backend auto-selected OpenCV SGBM (CUDA/repo/weights unavailable). "
            "For production depth maps use FoundationStereo on an NVIDIA PC — see README.",
            UserWarning,
            stacklevel=2,
        )
        _auto_warned[0] = True

    if use_fs:
        try:
            return _depth_foundation_stereo(
                left,
                right,
                fs_root=fs_root,
                fs_ckpt=fs_ckpt,
                fs_scale=fs_scale,
                fs_hiera=fs_hiera,
                fs_valid_iters=fs_valid_iters,
            )
        except Exception as e:
            if requested == "auto":
                warnings.warn(
                    f"FoundationStereo failed ({e}); falling back to SGBM.",
                    UserWarning,
                    stacklevel=2,
                )
                disp = compute_disparity_map_sgbm(
                    left, right, num_disparities=sgbm_num_disparities, block_size=sgbm_block_size
                )
                return disparity_to_uint8(disp)
            raise

    disp = compute_disparity_map_sgbm(
        left, right, num_disparities=sgbm_num_disparities, block_size=sgbm_block_size
    )
    return disparity_to_uint8(disp)


def depth_backend_info() -> dict[str, Any]:
    """Diagnostics for UI / CLI."""
    root = _resolve_fs_root(None)
    ckpt_ok = False
    cfg_ok = False
    cuda = False
    torch_ok = False
    try:
        import torch

        torch_ok = True
        cuda = bool(torch.cuda.is_available())
    except ImportError:
        pass
    if root:
        try:
            ck = _resolve_ckpt(root, None)
            ckpt_ok = ck.is_file()
            cfg_ok = (ck.parent / "cfg.yaml").is_file()
        except FileNotFoundError:
            pass
    return {
        "cuda": cuda,
        "torch": torch_ok,
        "foundation_stereo_root": str(root) if root else None,
        "checkpoint_found": ckpt_ok,
        "cfg_found": cfg_ok,
        "foundation_stereo_ready": bool(cuda and root and ckpt_ok and cfg_ok),
    }
