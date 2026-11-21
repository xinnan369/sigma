"""Pure NumPy metrics for frozen R2d MFNet official-test evaluation."""

from __future__ import annotations

import statistics

import numpy as np

CLASSES = 9


def confusion(gt, pred):
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    if gt.shape != pred.shape or gt.ndim != 2:
        raise ValueError('GT and prediction must be same-size two-dimensional maps')
    if not np.issubdtype(gt.dtype, np.integer) or not np.issubdtype(pred.dtype, np.integer):
        raise ValueError('GT and prediction must be integer class IDs')
    valid = (gt >= 0) & (gt < CLASSES)
    if np.any(~valid & (gt != 255)):
        raise ValueError('Unexpected MFNet ground-truth label')
    if np.any((pred < 0) | (pred >= CLASSES)):
        raise ValueError('Unexpected prediction class ID')
    return np.bincount(CLASSES * gt[valid].astype(np.int64) + pred[valid],
                       minlength=CLASSES**2).reshape(CLASSES, CLASSES)


def metrics(hist):
    hist = np.asarray(hist)
    if hist.shape != (CLASSES, CLASSES) or not np.issubdtype(hist.dtype, np.integer) or np.any(hist < 0):
        raise ValueError('Expected nonnegative 9x9 integer confusion matrix')
    if int(hist.sum()) == 0:
        raise ValueError('Cannot score an empty test set')
    inter = np.diag(hist)
    union = hist.sum(axis=0) + hist.sum(axis=1) - inter
    ious = [float(inter[i] / union[i]) if union[i] else None for i in range(CLASSES)]
    all_present = [x for x in ious if x is not None]
    fg_present = [x for x in ious[1:] if x is not None]
    return {
        'miou': float(np.mean(all_present)),
        'foreground_miou': float(np.mean(fg_present)) if fg_present else None,
        'iou_0_to_8': ious,
        'class_present_count': len(all_present),
        'foreground_class_present_count': len(fg_present),
        'valid_pixel_count': int(hist.sum()),
        'confusion_matrix': hist.tolist(),
    }


def compare_runs(rows):
    """Compute seed-level paired comparisons; do not pretend 393 frames are 393 seeds."""
    seeds = (3407, 3408, 3409, 3410, 3411)
    arms = ('ssm', 'cnn', 'attention')
    expected = {(seed, arm) for seed in seeds for arm in arms}
    keyed = {(int(r['seed']), r['arm']): r for r in rows}
    if len(rows) != len(expected) or set(keyed) != expected:
        raise ValueError('All 15 distinct frozen runs are mandatory')
    scalar = ('miou', 'foreground_miou')
    # Foreground mIoU is always available on the full official MFNet test set.
    def report(xs):
        if any(x is None or not np.isfinite(x) for x in xs):
            raise ValueError('Missing or nonfinite test score')
        return {'values': [float(x) for x in xs], 'mean': float(statistics.mean(xs)),
                'sample_std': float(statistics.stdev(xs))}
    per_arm = {arm: {metric: report([keyed[(seed, arm)]['metrics'][metric] for seed in seeds])
                     for metric in scalar} for arm in arms}
    for arm in arms:
        per_arm[arm]['iou_0_to_8'] = [
            report([keyed[(seed, arm)]['metrics']['iou_0_to_8'][k] for seed in seeds])
            for k in range(CLASSES)
        ]
    paired = {}
    for left, right in (('ssm', 'cnn'), ('ssm', 'attention'), ('cnn', 'attention')):
        paired[f'{left}_minus_{right}'] = {
            metric: report([keyed[(seed, left)]['metrics'][metric] -
                            keyed[(seed, right)]['metrics'][metric] for seed in seeds])
            for metric in scalar
        }
    return {'status': 'COMPLETE', 'seeds': list(seeds), 'arms': list(arms),
            'arm_statistics': per_arm, 'paired_statistics': paired,
            'statistical_unit': 'one independently trained seed, evaluated on same frozen 393 images',
            'interpretation': 'Descriptive sample standard deviation; n=5, no significance claims'}
