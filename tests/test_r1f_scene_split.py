import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.r1c.data import canonical_sha
from experiments.r1f.split import components, verify


def make_fixture():
    ids = [f'{i:05d}{"D" if i % 2 else "N"}' for i in range(784)]
    train_list = sorted(i + suffix for i in ids for suffix in ('', '_flip'))
    labels = {i: [1] * 9 for i in ids}
    val, train = ids[:79], ids[79:]
    split = {'train_originals': train, 'validation_ids': val,
             'train_ids': sorted(i + suffix for i in train for suffix in ('', '_flip'))}
    split['split_sha256'] = canonical_sha(split)
    return ids, train_list, labels, split


def test_component_chain_and_isolation():
    ids, train_list, labels, split = make_fixture()
    edges = [(ids[0], ids[1]), (ids[1], ids[2]), (ids[90], ids[91])]
    groups = components(ids, edges)
    assert groups[0] == ids[:3]
    out = verify(split, groups, edges, train_list, ['99999N'], labels)
    assert out['validation']['images'] == [79] * 9
    cross = edges + [(ids[2], ids[91])]
    with pytest.raises(ValueError, match='crosses split'):
        verify(split, components(ids, cross), cross, train_list, ['99999N'], labels)
    with pytest.raises(ValueError, match='Visual edge'):
        verify(split, groups, cross, train_list, ['99999N'], labels)


def test_flip_test_id_and_digest_leakage_rejected():
    ids, train_list, labels, split = make_fixture()
    groups = components(ids, [])
    leaked = copy.deepcopy(split)
    leaked['train_ids'][-1] = ids[0] + '_flip'
    with pytest.raises(ValueError, match='membership'):
        verify(leaked, groups, [], train_list, [], labels)
    with pytest.raises(ValueError, match='membership'):
        verify(split, groups, [], train_list, [ids[100]], labels)
    split['split_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='digest'):
        verify(split, groups, [], train_list, [], labels)


def test_rotating_label_support_cannot_replace_group_support():
    ids, train_list, labels, split = make_fixture()
    for i in ids[3:79]:
        labels[i][8] = 0
    edges = [(ids[0], ids[1]), (ids[1], ids[2])]
    with pytest.raises(ValueError, match='class support'):
        verify(split, components(ids, edges), edges, train_list, [], labels)


def test_mirrored_dual_images_match_even_without_close_ids():
    pytest.importorskip('cv2')
    from experiments.r1f.scene import global_scores, near_id

    pair = np.random.default_rng(3407).integers(0, 256, (2, 240, 320), dtype=np.uint8)
    normal, mirror = global_scores([pair, pair[:, :, ::-1]])
    assert np.min(mirror[:, 0, 1]) > .9999
    assert np.max(normal[:, 0, 1]) < .2
    assert not near_id('00001D', '00900D')


def test_frozen_scene_split_in_repository():
    root = Path(__file__).resolve().parents[1]
    evidence = root / 'reports/evidence/r1f'
    if not (evidence / 'isolation.json').exists():
        pytest.skip('Frozen split requires a passed real asset audit')
    def read(path):
        return json.loads(path.read_text(encoding='utf-8'))
    split = read(evidence / 'split.json')
    scene = read(evidence / 'scene-audit.json')
    cert = read(evidence / 'isolation.json')
    lists = read(root / 'reports/evidence/r1c/lists.json')['lists']
    assert scene['all_pairs'] == 306936 and scene['originals'] == 784
    assert scene['exact_flips'] == {'RGB': 784, 'Modal': 784, 'Label': 784}
    assert not scene['flip_mismatches'] and scene['test_pixels_read'] == 0
    assert canonical_sha(scene) == split['graph_sha256']
    assert read(root / 'configs/R1f_scene.json')['split_sha256'] == split['split_sha256'] == cert['split_sha256']
    edges = [(e['a'], e['b']) for e in scene['edges']]
    assert components(scene['label_pixels'], edges) == scene['groups']
    assert verify(split, scene['groups'], edges, lists['train.txt']['ids'], lists['test.txt']['ids'],
                  scene['label_pixels']) == cert['class_support']
