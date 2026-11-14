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

def test_frozen_donors(tmp_path):
    import pytest
    from experiments.r1g.partition import EVIDENCE, read, references
    from experiments.r2h.donors import FILE, build, load
    from experiments.r2h.provenance import CONFIG

    split = read(EVIDENCE / 'split.json')
    graph, _ = references()
    cfg = read(CONFIG)
    document = load(cfg, split, graph)
    mapping = document['mapping']
    groups = {i: g[0] for g in graph['groups'] for i in g}
    assert document == build(split, graph)
    assert len(mapping) == len(set(mapping.values())) == 157
    assert all(i != d and i[-1] == d[-1] and groups[i] != groups[d] for i, d in mapping.items())
    with pytest.raises(ValueError, match='Split digest'):
        build(dict(split, split_sha256='bad'), graph)
    with pytest.raises(ValueError, match='Graph digest'):
        build(split, dict(graph, invalid=True))
    tampered = tmp_path / 'donors.json'
    tampered.write_bytes(FILE.read_bytes() + b' ')
    with pytest.raises(ValueError, match='Frozen donor file digest'):
        load(cfg, split, graph, tampered)
    with pytest.raises(ValueError, match='Frozen donor file digest'):
        load(dict(cfg, mismatch='next ID'), split, graph)


def test_full_gate_rejects_old_implementation(tmp_path):
    import json
    import pytest
    from experiments.r2h.provenance import validate_gate

    (tmp_path / 'certificate.json').write_text(json.dumps(
        {'phase': 'r2h-pilot', 'status': 'PASS', 'fingerprints': {'r2h_sources': {'analysis.py': 'old'}}}))
    with pytest.raises(ValueError, match='pilot gate failed'):
        validate_gate(tmp_path, {'r2h_sources': {'analysis.py': 'current'}}, [], 3407, ())
