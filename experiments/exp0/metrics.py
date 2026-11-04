"""Paired clean/corrupt segmentation metrics, excluding void GT pixels."""
from __future__ import annotations
import numpy as np


def paired_error_stats(clean: np.ndarray, corrupted: np.ndarray,
                       gt: np.ndarray, intervention_mask: np.ndarray,
                       *, ignore_index: int = 255) -> dict[str, float | int]:
    """Rates use all valid outside pixels, not only initially correct pixels.

    No outside pixels => zero counts and NaN rates. Wrong-to-wrong changes
    contribute to disagreement only. Class 0 is valid; GT 255 is excluded.
    """
    a, b, g, mask = (np.asarray(x) for x in (clean, corrupted, gt, intervention_mask))
    if a.ndim != 2 or a.shape != b.shape or a.shape != g.shape or a.shape != mask.shape:
        raise ValueError("All predictions, GT and mask must be matching HxW arrays")
    if any(not np.issubdtype(x.dtype, np.integer) for x in (a, b, g)):
        raise TypeError("Predictions and GT must be integer label arrays")
    if mask.dtype != np.bool_:
        raise TypeError("Intervention support must be a boolean mask")
    valid = g != ignore_index
    outside = valid & ~mask
    n = int(outside.sum())
    new_error = outside & (a == g) & (b != g)
    corrected = outside & (a != g) & (b == g)
    disagreement = outside & (a != b)
    return {
        "n_valid_outside": n,
        "new_errors_outside": int(new_error.sum()),
        "corrected_errors_outside": int(corrected.sum()),
        "new_error_rate_outside": float(new_error.sum() / n) if n else float('nan'),
        "corrected_error_rate_outside": float(corrected.sum() / n) if n else float('nan'),
        "net_error_delta_outside": float((new_error.sum() - corrected.sum()) / n) if n else float('nan'),
        "disagreement_rate_outside": float(disagreement.sum() / n) if n else float('nan'),
    }
