import numpy as np


def test_repair_counts_use_corrupt_baseline_and_clean_induced_errors():
    from experiments.r2h.metrics import repair

    gt = np.array([[0, 0, 0, 0, 0, 255]])
    clean = np.array([[0, 1, 0, 1, 0, 0]])
    corrupt = np.array([[1, 0, 0, 1, 1, 0]])
    restored = np.array([[0, 1, 1, 0, 0, 1]])
    roi = np.array([[False, False, False, False, True, False]])
    result = repair(clean, corrupt, restored, gt, roi, {'outside': ~roi})
    assert result['outside'] == {'pixels': 4, 'induced': 1, 'recovered': 1, 'introduced': 2,
                                 'corrected': 2, 'new_change': 0, 'net_change': 0, 'disagree': 4}
    assert result['bands']['outside'] == result['outside']
    assert result['classes'][0] == result['outside']
    assert result['classes'][1]['pixels'] == 0


def test_no_intervention_has_zero_change_but_retains_induced_denominator():
    from experiments.r2h.metrics import repair

    clean, corrupt = np.array([[0]]), np.array([[1]])
    result = repair(clean, corrupt, corrupt, clean, np.zeros_like(clean, dtype=bool), {})['outside']
    assert result['induced'] == 1
    assert result['recovered'] == result['net_change'] == result['new_change'] == 0
