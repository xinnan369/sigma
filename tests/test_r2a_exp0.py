import numpy as np
import pytest
import torch

from experiments.r1g.partition import ROOT, read
from experiments.r2a.analyze import iou_intervals, rate_intervals
from experiments.r2a.hooks import observe, replace, response
from experiments.r2a.metrics import confusion, digest, distance_bands, mask_record, pair, scores
from experiments.r2a.run import cases, masks, perturbed


def test_frozen_cases_support_seed_and_replicated_raw_thermal():
    cfg = read(ROOT / 'configs/R2a_exp0.json')
    assert len(list(cases(cfg))) == 14
    supports = masks(cfg)
    raw = np.full((480, 640), 80, dtype=np.uint8)
    for spec in cases(cfg):
        changed, support, seed = perturbed(raw, spec, '00057D', cfg)
        expected = supports['identity' if spec['kind'] == 'identity' else spec['position']]
        assert np.array_equal(support, expected)
        assert np.array_equal(changed[~support], raw[~support])
        rec = mask_record(support)
        recovered = np.zeros(raw.size, dtype=bool)
        for start, count in rec['runs']:
            recovered[start:start + count] = True
        assert np.array_equal(recovered.reshape(raw.shape), support)
        assert rec['packed_sha256'] == digest(np.packbits(support))
        assert np.all(raw == 80)
        if spec['kind'] == 'gaussian':
            again, _, same_seed = perturbed(raw, spec, '00057D', cfg)
            other, _, other_seed = perturbed(raw, spec, '00058D', cfg)
            assert seed == same_seed and seed != other_seed and np.array_equal(again, changed)
            assert not np.array_equal(other, changed)
        if spec['kind'] == 'shift_local':
            y, x, _, _ = cfg['positions_y0_x0_y1_x1'][spec['position']]
            assert np.all(changed[y:y + 64, x:x + 8] == 0)
            assert np.all(changed[y:y + 64, x + 8:x + 64] == 80)


def test_outside_new_and_correction_cancel_but_disagreement_survives():
    gt = np.array([[0, 1, 2, 255], [0, 1, 2, 3]])
    a = np.array([[0, 0, 1, 1], [0, 1, 2, 3]])
    b = np.array([[1, 1, 0, 2], [1, 1, 2, 3]])
    roi = np.zeros(gt.shape, dtype=bool)
    roi[1, 0] = True
    r = pair(a, b, gt, roi, {'all_out': ~roi}, scores(confusion(gt, a)))
    assert r['outside'] == {'pixels': 6, 'new': 1, 'corrected': 1, 'net': 0, 'disagree': 3}
    assert r['by_distance']['all_out'] == r['outside']
    assert r['by_true_class_outside'][0]['new'] == 1
    assert r['by_true_class_outside'][1]['corrected'] == 1
    assert np.trace(r['pred_transition_outside']) == 3


def test_distances_partition_the_roi_exterior():
    pytest.importorskip('scipy')
    roi = np.zeros((5, 5), dtype=bool)
    roi[2, 2] = True
    bands = distance_bands(roi, [1, 2])
    assert bands['0-1'].sum() == 4
    assert bands['1-2'][1, 1]
    assert np.array_equal(np.sum(list(bands.values()), axis=0), ~roi)


def test_cluster_bootstrap_retains_pairing_and_masks_absent_classes():
    weights = np.array([[2, 0], [0, 2], [1, 1]])
    counts = np.array([[100, 10, 2, 8, 13], [20, 2, 3, -1, 7]])
    rates = rate_intervals(counts, weights)['rates']
    assert rates['new']['point'] == pytest.approx(0.1)
    assert rates['corrected']['point'] == pytest.approx(5 / 120)
    assert rates['net']['point'] == pytest.approx(7 / 120)
    assert rates['net']['ci95'] == pytest.approx(np.percentile([.08, -.05, 7 / 120], [2.5, 97.5]))
    clean = np.tile(np.eye(9, dtype=int), (2, 1, 1))
    clean[1, 1, 1] = 0
    corrupt = clean.copy()
    corrupt[0, 1, 1] = 0
    corrupt[0, 1, 0] = 1
    result = iou_intervals(clean, corrupt, weights)
    assert result['class_iou'][1]['paired_delta'] == {'point': -1, 'ci95': [-1, -1], 'valid_draws': 2}


class Cross(torch.nn.Module):
    def forward(self, a, b):
        return a + b, a - b


class Concat(torch.nn.Module):
    def forward(self, a, b):
        return a - 2 * b


def test_bhwc_two_branch_replacement_and_hook_cleanup():
    cross, concat = Cross(), Concat()
    mapping = {'cross.0': cross, 'concat.0': concat}
    a = torch.ones(1, 4, 4, 2)
    b = 2 * a

    def forward(x, y):
        return concat(*cross(x, y))

    with observe(mapping, 'same') as snap:
        clean = forward(a, b)
    assert torch.equal(snap.values['cross.0', 'in'][1], b)
    assert len(snap.values['cross.0', 'out']) == 2
    with replace(cross, 'cross.0', snap, 'same'):
        assert torch.equal(forward(a, 3 * b), clean)
    assert not torch.equal(forward(a, 3 * b), clean)
    with pytest.raises(ValueError, match='different sample'):
        with replace(cross, 'cross.0', snap, 'different'):
            forward(a, b)
    with pytest.raises(ValueError, match='signature mismatch'):
        with replace(cross, 'cross.0', snap, 'same'):
            forward(a[:, :2], b[:, :2])
    with pytest.raises(RuntimeError, match='expected failure'):
        with observe(mapping, 'same'):
            raise RuntimeError('expected failure')
    for m in mapping.values():
        assert not m._forward_hooks and not m._forward_pre_hooks


def test_roi_max_mapping_preserves_edge_cell_and_zero_response():
    clean = torch.ones(1, 2, 4, 3)
    roi = torch.zeros(1, 1, 8, 16)
    roi[0, 0, 3, 7] = 1
    changed = clean.clone()
    changed[0, 0, 1] *= 2
    changed[0, 1, 3] *= 3
    r = response(clean, changed, roi)
    assert r['inside_cells'] == 1
    assert r['outside_area_fraction'] == pytest.approx(7 / 8)
    assert r['outside_response_ratio'] == pytest.approx(2 / 3)
    assert response(clean, clean, roi)['outside_response_ratio'] is None
