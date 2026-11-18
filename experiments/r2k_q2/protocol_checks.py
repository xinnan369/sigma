"""Pure protocol checks for synthetic fixtures; no model fitting."""
from collections import Counter
import hashlib
import math


def group_oof(fit_groups, calibration_groups=(), test_groups=(), seed=3407):
    """Assign each Fit group to exactly one deterministic held-out fold."""
    fit, calibration, test = map(set, (fit_groups, calibration_groups, test_groups))
    if fit & calibration or fit & test or calibration & test:
        raise ValueError('Partitions share acquisition groups')
    if len(fit) < 5:
        raise ValueError('Five nonempty folds require five Fit groups')
    ordered = sorted(fit, key=lambda g: (hashlib.sha256(f'{seed}/{g}'.encode()).digest(), g))
    return [(fit - held, held) for k in range(5) if (held := set(ordered[k::5]))]


def supervision(o, device_state, predicted_d, arm):
    """Return D mask and U target; None means no valid device label."""
    if arm not in ('candidate', 'o_matched', 'full_support'):
        raise ValueError('Unknown arm')
    if o not in (0, 1) or device_state not in (0, 1, None):
        raise ValueError('Invalid supervision')
    if not math.isfinite(predicted_d) or not 0 <= predicted_d <= 1:
        raise ValueError('Invalid out-of-fold probability')
    if device_state is None:
        return False, None
    return arm == 'full_support' or o == 1, int((predicted_d >= .5) != device_state)


def frame_input(record):
    """Allowlist: acquisition and supervision stay outside the predictor."""
    return record['current_frame']


def support_weights(groups, scenes, eligible):
    """Use the same hierarchy on every objective/metric's declared support."""
    if not len(groups) == len(scenes) == len(eligible):
        raise ValueError('Unaligned weights')
    rows = [(g, s) for g, s, include in zip(groups, scenes, eligible) if include]
    n_points = Counter(rows)
    n_scenes = Counter(g for g, _ in n_points)
    return [1 / (len(n_scenes)*n_scenes[g]*n_points[g, s]) if include else 0.
            for g, s, include in zip(groups, scenes, eligible)]


def head_budget(widths, grid_points):
    return dict(parameters=sum((a+1)*b for a, b in zip(widths, widths[1:])),
                macs_per_frame=grid_points*sum(a*b for a, b in zip(widths, widths[1:])))


def matched_budget(reference, other):
    limits = {'parameters': .01, 'macs_per_frame': .05, 'median_ms': .10}
    return all(math.isfinite(reference[k]) and math.isfinite(other[k])
               and min(reference[k], other[k]) > 0
               and abs(other[k]/reference[k]-1) <= limit+1e-12
               for k, limit in limits.items())


def selective_risk(labels, predictions, acceptance, weights):
    if not len(labels) == len(predictions) == len(acceptance) == len(weights):
        raise ValueError('Unaligned decisions')
    risks = []
    for state in (0, 1):
        rows = [(pred != y, w) for y, pred, accept, w in
                zip(labels, predictions, acceptance, weights) if y == state and accept]
        mass = sum(w for _, w in rows)
        if mass == 0:
            return None
        risks.append(sum(w for err, w in rows if err)/mass)
    return sum(risks)/2
