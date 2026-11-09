import copy
import random

import numpy as np
import pytest
import torch

from experiments.r1c.data import base_id, internal_split
from experiments.r1c.pilot import metrics
from experiments.r1c.state import (Sampler, accumulate_loss, complete_groups, load_checkpoint,
                                   normalize_gradients, save_checkpoint)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.raw = torch.nn.Parameter(torch.ones(2, 2))
        self.bias = torch.nn.Parameter(torch.zeros(2))
        self.A_logs = torch.nn.Parameter(torch.ones(2, 2))
        self.A_logs._no_weight_decay = True

    def forward(self, x):
        return (self.linear(x) @ self.raw + self.bias) * self.A_logs.mean()


def optimizer(model):
    legacy = [{'params': [model.linear.weight]}, {'params': [model.linear.bias], 'weight_decay': 0.}]
    groups, report = complete_groups(model, legacy, 1e-3, .01)
    return torch.optim.AdamW(groups), report


def test_optimizer_covers_raw_ssm_parameters_once_and_preserves_legacy():
    model = Toy()
    opt, report = optimizer(model)
    records = {x['name']: x for x in report['parameters']}
    assert report['legacy_missing_numel'] == 10
    assert records['raw']['weight_decay'] == .01
    assert records['bias']['weight_decay'] == records['A_logs']['weight_decay'] == 0
    assert records['linear.weight']['origin'] == 'legacy'
    assert report['total_numel'] == sum(p.numel() for p in model.parameters())
    with pytest.raises(ValueError, match='duplicate'):
        complete_groups(model, [{'params': [model.raw, model.raw]}], .1, .01)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    model(torch.ones(1, 3)).square().mean().backward()
    assert all(p.grad is not None for p in model.parameters())
    opt.step()
    assert all(not torch.equal(p, before[n]) for n, p in model.named_parameters())


def test_accumulated_gradient_matches_average_microbatch_loss():
    torch.manual_seed(12)
    model, xs = Toy().double(), torch.randn(8, 3, dtype=torch.double)
    combined = copy.deepcopy(model)
    for x in xs:
        (model(x[None]).square().mean() / 8).backward()
    combined(xs).square().mean().backward()
    for p, q in zip(model.parameters(), combined.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-12, atol=1e-12)


def test_valid_pixel_weighted_accumulation_matches_padded_batch():
    torch.manual_seed(14)
    a = torch.nn.Conv2d(2, 3, 1).double()
    b = copy.deepcopy(a)
    inputs = torch.randn(3, 2, 4, 5, dtype=torch.double)
    target = torch.randint(0, 3, (3, 4, 5))
    target[0, :3] = 255
    target[1, :1] = 255
    total = 0
    for x, y in zip(inputs, target):
        count = int((y != 255).sum())
        accumulate_loss(torch.nn.functional.cross_entropy(a(x[None]), y[None], ignore_index=255), count)
        total += count
    normalize_gradients(a, total)
    torch.nn.functional.cross_entropy(b(inputs), target, ignore_index=255).backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match='no supervised'):
        normalize_gradients(a, 0)


def test_holdout_keeps_flip_identity_together_and_excludes_test():
    ids = [f'{i:05d}{"D" if i % 2 else "N"}' for i in range(30)]
    lists = {'train.txt': ids + [x + '_flip' for x in ids], 'test.txt': ['99999N']}
    split = internal_split(lists)
    assert split == internal_split(lists)
    assert len(split['validation_ids']) == 4
    train = {base_id(x) for x in split['train_ids']}
    assert not train & set(split['validation_ids'])
    assert train | set(split['validation_ids']) == set(ids)
    lists['test.txt'] = [ids[0]]
    with pytest.raises(ValueError, match='leakage'):
        internal_split(lists)


def test_metrics_include_background_and_ignore_absent_unions():
    hist = np.zeros((9, 9), dtype=int)
    hist[:2, :2] = [[8, 2], [1, 9]]
    out = metrics(hist)
    assert out['miou_including_0'] == pytest.approx((8 / 11 + 9 / 12) / 2)
    assert out['absent_union_classes'] == list(range(2, 9))
    assert out['pixel_count'] == 20


def test_atomic_resume_restores_optimizer_rng_and_sampler(tmp_path):
    torch.manual_seed(10)
    random.seed(11)
    np.random.seed(12)
    model = Toy()
    opt, _ = optimizer(model)
    sampler = Sampler(3, 9)
    signature = {'config': 'fixture', 'split': 'train-only'}

    def step():
        index = sampler.next()
        x = torch.randn(1, 3) + random.random() + np.random.rand() + index
        loss = model(x).square().mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss.item(), index

    step()
    checkpoint = tmp_path / 'last.pt'
    save_checkpoint(checkpoint, model, opt, sampler, 1, {'miou': .1}, signature)
    assert not (tmp_path / 'last.pt.tmp').exists()
    expected = [step() for _ in range(5)]
    expected_model = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(opt.state_dict())
    model = Toy()
    opt, _ = optimizer(model)
    sampler = Sampler(3, 90)
    update, best, keys = load_checkpoint(checkpoint, model, opt, sampler, signature)
    assert update == 1 and best == {'miou': .1} and 'All keys matched' in keys
    assert [step() for _ in range(5)] == expected
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, expected_model[k], rtol=0, atol=0)
    for key, value in opt.state_dict()['state'].items():
        for name, tensor in value.items():
            torch.testing.assert_close(tensor, expected_optimizer['state'][key][name], rtol=0, atol=0)
    with pytest.raises(ValueError, match='mismatch'):
        load_checkpoint(checkpoint, model, opt, sampler, {'different': 1})
    model(torch.ones(1, 3)).sum().backward()
    with pytest.raises(ValueError, match='boundary'):
        save_checkpoint(checkpoint, model, opt, sampler, 2, best, signature)


def test_sampler_rejects_corrupt_checkpoint():
    sampler = Sampler(3, 1)
    state = sampler.state_dict()
    state['order'] = [0, 0, 2]
    with pytest.raises(ValueError, match='sampler'):
        sampler.load_state_dict(state)
