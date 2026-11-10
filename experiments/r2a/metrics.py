"""Paired counts in pixel coordinates; summaries contain no prediction maps."""
import hashlib

import numpy as np

from experiments.r1d.train import class_metrics

COUNTS = ('pixels', 'new', 'corrected', 'net', 'disagree')


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def mask_record(mask):
    # Row-major runs represent the actual Boolean support without image payloads.
    indices = np.flatnonzero(mask.ravel())
    chunks = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    return {'shape': list(mask.shape), 'true_count': int(mask.sum()),
            'encoding': 'row-major [start, length]; digest of np.packbits(mask, bitorder=big)',
            'runs': [[int(c[0]), len(c)] for c in chunks if len(c)],
            'packed_sha256': digest(np.packbits(mask, bitorder='big'))}


def distance_bands(mask, bounds):
    from scipy.ndimage import distance_transform_edt

    if not mask.any():
        return {}
    distance = distance_transform_edt(~mask)
    return {f'{a}-{b}' if np.isfinite(b) else f'>{a}': (distance > a) & (distance <= b)
            for a, b in zip([0] + bounds, bounds + [np.inf])}


def confusion(gt, pred, mask=None):
    valid = (gt >= 0) & (gt < 9)
    if mask is not None:
        valid &= mask
    return np.bincount(9 * gt[valid].astype(np.int64) + pred[valid], minlength=81).reshape(9, 9)


def scores(matrix):
    out = class_metrics(matrix)
    return {'miou': out['miou_including_0'], 'foreground_miou': out['foreground_miou'],
            'iou_0_to_8': out['iou_0_to_8']}


def pair(clean, changed, gt, support, bands, clean_scores):
    if not (clean.shape == changed.shape == gt.shape == support.shape) or support.dtype != np.bool_:
        raise ValueError('Invalid paired arrays or Boolean support')
    valid = (gt >= 0) & (gt < 9)
    outside = valid & ~support
    wrong_clean, wrong_changed = clean != gt, changed != gt
    new, corrected = ~wrong_clean & wrong_changed, wrong_clean & ~wrong_changed
    disagree = clean != changed

    def counts(region):
        n, a, b, d = [int(x.sum()) for x in (region, region & new, region & corrected, region & disagree)]
        return dict(zip(COUNTS, (n, a, b, a - b, d)))

    matrix = confusion(gt, changed)
    return {'clean': clean_scores, 'corrupt': scores(matrix), 'corrupt_matrix': matrix.tolist(),
            'outside': counts(outside), 'by_distance': {name: counts(mask & outside) for name, mask in bands.items()},
            'by_true_class_outside': [counts(outside & (gt == i)) for i in range(9)],
            'clean_matrix_outside': confusion(gt, clean, outside).tolist(),
            'corrupt_matrix_outside': confusion(gt, changed, outside).tolist(),
            'pred_transition_outside': np.bincount(9 * clean[outside] + changed[outside], minlength=81).reshape(9, 9).tolist()}
