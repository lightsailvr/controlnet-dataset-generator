#!/usr/bin/env python3
"""Extract disparity / depth proxy from stereoscopic side-by-side images.

Uses OpenCV StereoSGBM on left/right half-views. Output is a single-channel
uint8 PNG (lighter = nearer, heuristic) suitable as auxiliary training data.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError("depth_extractor requires opencv-python: pip install opencv-python-headless") from e


def split_sbs_halves(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a side-by-side stereo frame into left and right BGR views."""
    h, w = bgr.shape[:2]
    half = w // 2
    left = bgr[:, :half].copy()
    right = bgr[:, half : half * 2].copy()
    return left, right


def compute_disparity_map(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    *,
    num_disparities: int = 128,
    block_size: int = 5,
) -> np.ndarray:
    """Compute disparity (left view) using Semi-Global Block Matching."""
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    # num_disparities must be divisible by 16
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
    d = disp.copy()
    if d.max() <= d.min():
        return np.zeros_like(d, dtype=np.uint8)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return (d * 255.0).clip(0, 255).astype(np.uint8)


def depth_from_sbs_bgr(sbs_bgr: np.ndarray, **kwargs) -> np.ndarray:
    """Full pipeline: SBS BGR → uint8 disparity image (H x W//2, single channel)."""
    left, right = split_sbs_halves(sbs_bgr)
    if left.shape != right.shape:
        rh, rw = right.shape[:2]
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR)
    disp = compute_disparity_map(left, right, **kwargs)
    return disparity_to_uint8(disp)
