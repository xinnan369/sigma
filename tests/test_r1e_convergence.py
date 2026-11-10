import copy
import json
import random

import numpy as np
import pytest
import torch

from experiments.r1.assets import sha256
from experiments.r1c.state import capture_rng
from experiments.r1d.state import EpochSampler
from experiments.r1d.train import BRIDGE
from experiments.r1e.state import N, STEPS, gate, read_parent, replace_best, restore, save


def setup():
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
    return model, opt, sched, EpochSampler(N, 3407)


def step(model, opt, sched):
    x = torch.tensor([[1. + random.random() + np.random.rand()]]) + torch.rand(1, 1)
    model(x).square().mean().backward()
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)


def test_parent_requires_hash_version_and_exact_signature(tmp_path):
    p = tmp_path / 'parent.pt'
    state = {'format': 'r1d-epoch-v1', 'signature': {'protocol': 'accepted'},
             'sampler': {'epoch': 1, 'cursor': N}, 'progress': {'updates': STEPS}}
    torch.save(state, p)
    with pytest.raises(ValueError, match='SHA256'):
        read_parent(p, '0' * 64, state['signature'])
    with pytest.raises(ValueError, match='signature'):
        read_parent(p, sha256(p), {'protocol': 'different'})
    assert read_parent(p, sha256(p), state['signature']) == state
    state['sampler']['cursor'] -= 2
    torch.save(state, p)
    with pytest.raises(ValueError, match='boundary'):
        read_parent(p, sha256(p), state['signature'])


def test_native_restore_preserves_adamw_rng_and_next_epoch(tmp_path):
    original = setup()
    for _ in range(STEPS):
        original[3].next_window(8)
        step(*original[:3])
    path = tmp_path / 'last.pt'
    save(path, *original, {'updates': STEPS}, [{'epoch': 1}], {'protocol': 'fixture'})
    original[3].advance_epoch()
    step(*original[:3])
    state = torch.load(path, map_location='cpu', weights_only=False)
    resumed = setup()
    restore(state, *resumed, 3407)
    assert resumed[2].last_epoch == STEPS
    resumed[3].advance_epoch()
    assert resumed[3].order == original[3].order
    step(*resumed[:3])
    for p, q in zip(resumed[0].parameters(), original[0].parameters()):
        assert torch.equal(p, q)
        for key in resumed[1].state[p]:
            assert torch.equal(resumed[1].state[p][key], original[1].state[q][key])
    best = tmp_path / 'best.pt'
    replace_best(path, best)
    digest = sha256(best)
    path.write_bytes(b'new last')
    assert sha256(best) == digest != sha256(path)


def test_restore_rejects_bad_optimizer_sampler_and_rng():
    model, opt, sched, sampler = setup()
    step(model, opt, sched)
    sampler.next_window(8)
    base = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(),
            'sampler': sampler.state_dict(), 'progress': {'updates': 1}, 'rng': capture_rng()}
    broken = copy.deepcopy(base)
    next(iter(broken['optimizer']['state'].values()))['step'].zero_()
    with pytest.raises(ValueError, match='AdamW'):
        restore(broken, *setup(), 3407)
    broken = copy.deepcopy(base)
    broken['sampler']['order'][:2] = reversed(broken['sampler']['order'][:2])
    with pytest.raises(ValueError, match='Sampler'):
        restore(broken, *setup(), 3407)
    broken = copy.deepcopy(base)
    del broken['rng']['numpy']
    with pytest.raises(ValueError, match='RNG'):
        restore(broken, *setup(), 3407)


def test_gate_checks_foreground_loss_and_background():
    cfg = json.loads((BRIDGE / 'configs/R1e_convergence.json').read_text())
    history = [{'epoch': n, 'validation': {'miou_including_0': .4, 'foreground_miou': .3,
                                          'mean_pixel_loss': .1, 'classes': [{'predicted_fraction': .93}]}}
               for n in (1, 2, 3)]
    assert all(gate(history, cfg).values())
    history[-1]['validation'].update(foreground_miou=.26, mean_pixel_loss=.13,
                                     classes=[{'predicted_fraction': .99}])
    assert gate(history, cfg) == {'overall_miou': True, 'foreground_miou': False,
                                  'validation_loss': False, 'background_share': False}
