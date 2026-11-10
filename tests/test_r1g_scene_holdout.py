import copy

import numpy as np
import pytest
import torch

from experiments.r1d.state import EpochSampler
from experiments.r1g.partition import EVIDENCE, ROOT, TRAIN, UPDATES, read, references, verify
from experiments.r1g.statistics import aggregate, bootstrap, cohorts, grouped, iou_draws


def inputs():
    graph, prior = references()
    stored = read(ROOT / 'reports/evidence/r1c/lists.json')['lists']
    return read(EVIDENCE / 'split.json'), graph, prior, {k: v['ids'] for k, v in stored.items()}


def test_strict_frozen_split_and_partition_certificate():
    split, graph, prior, lists = inputs()
    stats = verify(split, graph, prior, lists)
    cert = read(EVIDENCE / 'isolation.json')
    cfg = read(ROOT / 'configs/R1g_scene.json')
    assert cert['status'] == 'PASS' and stats == cert['class_support']
    assert cfg['split_sha256'] == split['split_sha256'] == cert['split_sha256']
    assert cfg['selected_assets_sha256'] == cert['selected_assets_sha256']
    assert len(split['retained_ids']) == 79 and len(split['added_ids']) == 78
    assert stats['train']['visual_group_count'] + stats['validation']['visual_group_count'] == 396
    assert min(stats['validation']['groups']) >= 6 and min(stats['train']['groups']) >= 6
    assert min(stats['validation']['day'], stats['validation']['night']) >= 50
    regions = cohorts(split)
    assert regions['retained_79'] | regions['added_78'] == regions['day'] | regions['night'] == regions['all']
    assert not regions['retained_79'] & regions['added_78']


def test_refuse_flipped_validation_image_and_cross_partition_edge():
    split, graph, prior, lists = inputs()
    changed = copy.deepcopy(split)
    changed['train_ids'][0] = split['validation_ids'][0] + '_flip'
    with pytest.raises(ValueError, match='Augmentation'):
        verify(changed, graph, prior, lists)
    changed_graph = copy.deepcopy(graph)
    changed_graph['edges'].append({'a': split['validation_ids'][0], 'b': split['train_originals'][0]})
    with pytest.raises(ValueError, match='visual group'):
        verify(split, changed_graph, prior, lists)
    lists['test.txt'][0] = split['train_originals'][0]
    with pytest.raises(ValueError, match='official test'):
        verify(split, graph, prior, lists)


def test_five_epochs_each_have_an_unpadded_six_sample_tail():
    sampler = EpochSampler(TRAIN, 3407)
    generator = torch.Generator().manual_seed(3407)
    for epoch in range(1, 6):
        windows = [sampler.next_window(8) for _ in range(UPDATES)]
        assert [len(x) for x in windows] == [8] * 156 + [6]
        assert [i for x in windows for i in x] == torch.randperm(TRAIN, generator=generator).tolist()
        assert sampler.epoch == epoch and sampler.cursor == TRAIN
        assert sampler.next_window(8) == []
        if epoch < 5:
            sampler.advance_epoch()


def fixture():
    matrices = np.zeros((4, 9, 9), dtype=int)
    matrices[0, 0, 0] = 20
    matrices[1, 1, 1] = 4
    matrices[2, 0, 1] = 8
    matrices[3, 0, 0] = 20
    return [{'id': str(i), 'group': ['A', 'A', 'B', 'C'][i], 'loss_sum': 1.,
             'confusion_matrix': m.tolist()} for i, m in enumerate(matrices)]


def test_cluster_draw_keeps_related_images_together_and_handles_absence():
    records = fixture()
    clusters = grouped(records)
    assert [r['ids'] for r in clusters] == [['0', '1'], ['2'], ['3']]
    observed = aggregate(records)
    assert observed['class_image_support'][1] == observed['class_group_support'][1] == 1
    assert observed['iou_0_to_8'][1] == pytest.approx(4 / 12)
    matrices = np.array([r['confusion_matrix'] for r in clusters])
    ious, truths = iou_draws(matrices, np.array([[2, 1, 0], [0, 2, 1], [0, 0, 3]]))
    assert ious[0, 1] == pytest.approx(8 / 16)
    assert ious[1, 1] == 0 and truths[1, 1] == 0
    assert np.isnan(ious[2, 1]) and truths[2, 1] == 0
    assert np.isnan(ious[:, 2:]).all()


def test_cluster_bootstrap_seed_and_undefined_intervals():
    clusters = grouped(fixture())
    result = bootstrap(clusters, replicates=1000)
    assert result == bootstrap(clusters, replicates=1000)
    assert result['classes'][1]['statistically_unstable'] and result['classes'][1]['truth_groups'] == 1
    assert result['classes'][2]['ci95'] is None and result['classes'][2]['valid_replicates'] == 0
    assert result['classes'][1]['no_ground_truth_replicates'] > result['classes'][1]['undefined_replicates'] > 0
    assert result['foreground']['valid_replicates'] < 1000
