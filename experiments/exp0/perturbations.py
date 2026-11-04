"""Deterministic, local interventions on a *single* auxiliary modality.

Inputs HxWxC float32/uint8 in physical pixel scale [0, 255], BEFORE
model normalization. These functions never mutate the caller's input.
"""
from __future__ import annotations
import numpy as np
from numbers import Integral


def _validate(x: np.ndarray, roi: tuple[int, int, int, int]):
    if not isinstance(x, np.ndarray) or x.ndim not in (2, 3):
        raise ValueError("Expected HxW or HxWxC ndarray")
    if not (np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.floating)):
        raise TypeError("Input must contain real pixel values")
    if x.size == 0 or not np.isfinite(x).all() or np.any((x < 0) | (x > 255)):
        raise ValueError("Input must be nonempty and finite in physical pixel scale [0, 255]")
    if len(roi) != 4 or any(not isinstance(v, Integral) or isinstance(v, bool) for v in roi):
        raise ValueError("ROI must contain four integer coordinates")
    h, w = x.shape[:2]
    y0, x0, y1, x1 = roi
    if not (0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
        raise ValueError("ROI must be within image and nonempty")


def corrupt_thermal(
    thermal: np.ndarray,
    kind: str,
    roi: tuple[int, int, int, int],
    *, seed: int = 3407,
    sigma: float = 20.0,
    fill: float = 0.0,
    dx: int = 3,
    dy: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (corrupted_copy, intervention_support_mask).

    The float32 output and bool mask never alias the input. The mask denotes
    *targeted* pixels, not necessarily all pixels whose value changed.
    `shift_local` translates within the ROI, using `fill` rather than wrapping.
    Gaussian noise is shared over channels, preserving replicated grayscale
    Thermal (as used by Sigma's MFNet loader).
    """
    _validate(thermal, roi)
    if kind not in ("identity", "occlusion", "gaussian", "shift_local"):
        raise ValueError(f"Unknown corruption type: {kind}")
    src = thermal.astype(np.float32, copy=False)
    out = src.copy()
    mask = np.zeros(src.shape[:2], dtype=bool)
    if kind == "identity":
        return out, mask
    y0, x0, y1, x1 = roi
    if kind == "gaussian" and (not np.isfinite(sigma) or sigma < 0):
        raise ValueError("sigma must be finite and nonnegative")
    if kind in ("occlusion", "shift_local") and (not np.isfinite(fill) or not 0 <= fill <= 255):
        raise ValueError("fill must be finite and in [0, 255]")
    if kind == "shift_local" and any(not isinstance(v, Integral) or isinstance(v, bool) for v in (dx, dy)):
        raise ValueError("dx and dy must be integers")
    if kind == "gaussian" and sigma == 0:
        return out, mask
    if kind == "shift_local" and dx == dy == 0:
        return out, mask
    mask[y0:y1, x0:x1] = True
    if kind == "occlusion":
        out[y0:y1, x0:x1] = fill
    elif kind == "gaussian":
        rng = np.random.default_rng(seed)
        patch = out[y0:y1, x0:x1]
        noise = rng.normal(0, sigma, size=patch.shape[:2])
        if patch.ndim == 3:
            noise = noise[..., None]
        out[y0:y1, x0:x1] = np.clip(patch + noise, 0, 255)
    else:
        h, w = y1 - y0, x1 - x0
        target = np.full_like(out[y0:y1, x0:x1], fill)
        yy, xx = np.mgrid[0:h, 0:w]
        sy, sx = yy - dy, xx - dx
        valid = (sy >= 0) & (sy < h) & (sx >= 0) & (sx < w)
        target[valid] = src[y0:y1, x0:x1][sy[valid], sx[valid]]
        out[y0:y1, x0:x1] = target
    return out, mask
