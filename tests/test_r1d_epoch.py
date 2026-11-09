import copy
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from experiments.r1.assets import sha256
from experiments.r1c.pilot import construct_model
from experiments.r1c.state import accumulate_loss, complete_groups, normalize_gradients
from experiments.r1d.state import EpochSampler, load, save
from experiments.r1d.train import BRIDGE, class_metrics, verify_coverage


def test_full_epoch_permutation_and_tail():
    sampler = EpochSampler(1410, 3407)
    windows = [sampler.next_window(8) for _ in range(177)]
    assert [len(w) for w in windows] == [8] * 176 + [2]
    assert sorted(i for w in windows for i in w) == list(range(1410))
    assert sampler.cursor == 1410 and sampler.epoch == 1
    assert sampler.next_window(8) == []
    sampler.advance_epoch()
    assert sampler.epoch == 2 and sampler.cursor == 0
    assert sorted(sampler.order) == list(range(1410))
    assert sampler.order != [i for w in windows for i in w]


def test_epoch_cannot_advance_with_pending_samples():
    sampler = EpochSampler(10, 3407)
    sampler.next_window(8)
    with pytest.raises(ValueError, match='Finish'):
        sampler.advance_epoch()
    assert len(sampler.next_window(8)) == 2
    assert sampler.next_window(8) == []


def test_two_sample_tail_matches_combined_valid_pixel_ce():
    torch.manual_seed(1)
    model = torch.nn.Conv2d(2, 3, 1).double()
    reference = copy.deepcopy(model)
    x = torch.randn(2, 2, 4, 4, dtype=torch.double)
    y = torch.randint(0, 3, (2, 4, 4))
    y[0, :3] = 255
    y[1, :1] = 255
    total = 0
    for i in range(2):
        count = int((y[i] != 255).sum())
        accumulate_loss(torch.nn.functional.cross_entropy(model(x[i:i+1]), y[i:i+1], ignore_index=255), count)
        total += count
    normalize_gradients(model, total)
    torch.nn.functional.cross_entropy(reference(x), y, ignore_index=255).backward()
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-12, rtol=1e-12)


def test_epoch_resume_restores_rng_scheduler_and_adamw(tmp_path):
    def setup():
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .99 ** step)
        return model, optimizer, scheduler, EpochSampler(18, 3407)

    def step(model, optimizer, scheduler, sampler):
        indices = sampler.next_window(8)
        x = torch.tensor(indices, dtype=torch.float32).view(-1, 1)
        x = x + random.random() + np.random.random() + torch.rand_like(x)
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        return indices, loss.item()

    first = setup()
    step(*first)
    path = tmp_path / 'last.pt'
    save(path, *first, {'updates': 1}, {'protocol': 'fixture'})
    expected = [step(*first), step(*first)]
    resumed = setup()
    progress, keys = load(path, *resumed, {'protocol': 'fixture'})
    assert progress == {'updates': 1} and 'All keys matched' in keys
    assert (resumed[3].epoch, resumed[3].cursor) == (1, 8)
    assert [step(*resumed), step(*resumed)] == expected
    assert len(expected[-1][0]) == 2 and resumed[3].next_window(8) == []
    assert first[2].state_dict() == resumed[2].state_dict()
    for left, right in zip(first[0].parameters(), resumed[0].parameters()):
        assert torch.equal(left, right)
        for key, tensor in first[1].state[left].items():
            assert torch.equal(tensor, resumed[1].state[right][key])
    first[3].advance_epoch()
    resumed[3].advance_epoch()
    assert first[3].order == resumed[3].order


def test_foreground_metrics_and_undefined_precision():
    hist = np.zeros((9, 9), dtype=np.int64)
    hist[:3, :3] = [[80, 3, 0], [2, 7, 0], [8, 0, 0]]
    out = class_metrics(hist)
    assert out['classes'][1]['precision'] == .7
    assert out['classes'][1]['recall'] == pytest.approx(7 / 9)
    assert out['classes'][2]['precision'] is None
    assert out['classes'][2]['recall'] == 0
    assert out['foreground_miou'] == pytest.approx((7 / 12 + 0) / 2)
    assert out['classes'][0]['predicted_fraction'] == .9


def test_real_sigma_optimizer_coverage():
    upstream = os.getenv('SIGMA_R1D_TEST_UPSTREAM')
    init = os.getenv('SIGMA_R1D_TEST_INIT')
    if not upstream or not init:
        pytest.skip('Pinned Sigma checkout and VMamba initializer required')
    cfg = json.loads((BRIDGE / 'configs/R1d_epoch.json').read_text())
    assert sha256(Path(init)) == cfg['init_sha256']
    import subprocess
    command = ['git', '-C', upstream]
    assert subprocess.check_output(command + ['rev-parse', 'HEAD'], text=True).strip() == cfg['upstream_commit']
    subprocess.run(command + ['diff', '--quiet', 'HEAD', '--'], check=True)
    reference = json.loads((BRIDGE / 'reports/evidence/r1c/pilot/parameter-coverage.json').read_text())
    torch.set_num_threads(4)
    with construct_model(Path(upstream), Path(init)) as (model, _, _):
        from utils.init_func import group_weight
        groups, report = complete_groups(model, group_weight([], model, torch.nn.BatchNorm2d, cfg['lr']), cfg['lr'], .01)
        verify_coverage(model, groups, report, reference)
        assert report['tensor_count'] == 668 and report['total_numel'] == 48287360
        broken = [dict(g, params=g['params'].copy()) for g in groups]
        broken[0]['params'].pop()
        with pytest.raises(ValueError, match='coverage'):
            verify_coverage(model, broken, report, reference)
        print('Real Sigma-Tiny optimizer verified: 668 tensors, 48287360 elements')
