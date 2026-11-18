import numpy as np

from experiments.r2k_q2.synthetic_cases import blur, cases, gradient_rms, summary


def test_exact_nonidentifiability_but_pair_can_disambiguate():
    c = cases()
    a, x = c['healthy_low_contrast']
    b, y = c['blurred_high_contrast']
    assert np.array_equal(x, y)
    assert np.array_equal(a, x)
    assert not np.array_equal(b, y)
    assert not np.any((blur(c['healthy_uniform'][1])-.5))


def test_different_arrays_collision_for_simple_statistics():
    c = cases()
    x = c['different_content_matched_statistics'][1]
    y = c['blurred_high_contrast'][1]
    assert not np.array_equal(x, y)
    assert np.array_equal(np.sort(x.ravel()), np.sort(y.ravel()))
    assert gradient_rms(x) == gradient_rms(y)


def test_noise_gradient_increase_and_occlusion_region():
    c = cases()
    ref, noisy = c['additive_noise']
    assert gradient_rms(noisy) > gradient_rms(ref)
    assert np.array_equal(noisy, cases()['additive_noise'][1])
    ref, changed = c['occlusion']
    mask = np.zeros(ref.shape, bool)
    mask[16:48, 16:48] = True
    assert (changed[mask] == 0).all()
    assert np.array_equal(ref[~mask], changed[~mask])


def test_zero_coverage_does_not_mean_zero_risk():
    r = summary()
    assert r['opposite_label_identical_observation']
    assert r['different_pixels_equal_histogram']
    assert r['equal_prior_collision']['abstention_example_coverage'] == 0
    assert r['equal_prior_collision']['abstention_example_selective_risk'] is None
