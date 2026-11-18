"""Deterministic CPU counterexamples, not a fitted quality estimator."""
import json
import platform

import numpy as np


def blur(x):
    """Periodic separable binomial blur, normalized and DC preserving."""
    for axis in (0, 1):
        x = (np.roll(x, 1, axis) + 2*x + np.roll(x, -1, axis))/4
    return x


def gradient_rms(x):
    return float(np.sqrt(np.mean(sum((np.roll(x, 1, a)-x)**2 for a in (0, 1)))))


def cases():
    _, col = np.indices((64, 64))
    # Integer cosine samples at pi/2; exactly representable under binomial blur.
    pattern = np.array([1., 0., -1., 0.])[col % 4]
    high = .5 + .25*pattern
    flat = np.full_like(high, .5)
    low = .5 + .125*pattern
    degraded = blur(high)
    # Same global histogram and gradient RMS, different phase layout/content.
    shifted = np.roll(low, 1, axis=1)
    noise = high + np.random.default_rng(3407).normal(0, .05, high.shape)
    occluded = high.copy()
    occluded[16:48,16:48] = 0
    return {
        'healthy_uniform': (flat, flat),
        'healthy_low_contrast': (low, low),
        'healthy_high_contrast': (high, high),
        'blurred_high_contrast': (high, degraded),
        'additive_noise': (high, noise),
        'occlusion': (high, occluded),
        'different_content_matched_statistics': (shifted, shifted),
    }


def summary():
    data = cases()
    rows = {}
    for name, (reference, observation) in data.items():
        d = observation-reference
        rows[name] = dict(mean=float(observation.mean()), std=float(observation.std()),
                          gradient_rms=gradient_rms(observation),
                          paired_rmse=float(np.sqrt(np.mean(d*d))),
                          altered_pixels=int(np.count_nonzero(d)))
    low = data['healthy_low_contrast'][1]
    observed_blur = data['blurred_high_contrast'][1]
    shifted = data['different_content_matched_statistics'][1]
    return dict(scope='SYNTHETIC_COUNTEREXAMPLES_ONLY', python=platform.python_version(),
                numpy=np.__version__, seed=3407, cases=rows,
                opposite_label_identical_observation=bool(np.array_equal(low, observed_blur)),
                different_pixels_equal_histogram=bool(np.array_equal(
                    np.sort(shifted.ravel()), np.sort(observed_blur.ravel()))
                    and not np.array_equal(shifted, observed_blur)),
                equal_prior_collision=dict(forced_error_lower_bound=.5,
                                           abstention_example_coverage=0,
                                           abstention_example_selective_risk=None))


if __name__ == '__main__':
    print(json.dumps(summary(), indent=2, allow_nan=False))
