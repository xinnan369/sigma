"""Strict R1d import and versioned R1e continuation state."""
import math
import os
import shutil

import torch

from experiments.r1.assets import sha256
from experiments.r1c.state import capture_rng, restore_rng
from experiments.r1d.state import EpochSampler

N = 1410
WINDOW = 8
STEPS = math.ceil(N / WINDOW)


def require(value, message):
    if not value:
        raise ValueError(message)


def read_parent(path, digest, signature):
    require(sha256(path) == digest, 'R1d final checkpoint SHA256 mismatch')
    state = torch.load(path, map_location='cpu', weights_only=False)
    require(state['format'] == 'r1d-epoch-v1' and state['signature'] == signature,
            'R1d format or exact accepted signature mismatch')
    require(state['sampler']['epoch'] == 1 and state['sampler']['cursor'] == N
            and state['progress']['updates'] == STEPS, 'R1d checkpoint is not the final epoch boundary')
    return state


def restore(state, model, optimizer, scheduler, sampler, seed):
    saved_sampler = state['sampler']
    epoch, cursor = saved_sampler['epoch'], saved_sampler['cursor']
    require(1 <= epoch <= 5 and 0 <= cursor <= N and (cursor % WINDOW == 0 or cursor == N),
            'Invalid epoch/update boundary')
    updates = (epoch - 1) * STEPS + math.ceil(cursor / WINDOW)
    require(state['progress']['updates'] == updates, 'Optimizer update count disagrees with cursor')
    expected = EpochSampler(N, seed)
    for _ in range(1, epoch):
        expected.next_window(N)
        expected.advance_epoch()
    require(saved_sampler['order'] == expected.order
            and torch.equal(saved_sampler['generator'], expected.generator.get_state()),
            'Sampler permutation or RNG disagrees with the seeded epoch')

    target = model.state_dict()
    saved_model = state['model']
    require(target.keys() == saved_model.keys(), 'Model keys differ')
    for name, tensor in saved_model.items():
        require(tensor.shape == target[name].shape and tensor.dtype == target[name].dtype
                and torch.isfinite(tensor).all().item(), f'Invalid model tensor: {name}')

    saved_opt = state['optimizer']
    fresh_groups = optimizer.state_dict()['param_groups']
    require(saved_opt['param_groups'] == fresh_groups, 'AdamW options or parameter group ordering differ')
    parameters = {idx: p for spec, group in zip(fresh_groups, optimizer.param_groups)
                  for idx, p in zip(spec['params'], group['params'])}
    require(saved_opt['state'].keys() == parameters.keys(), 'Incomplete AdamW moments')
    for idx, entry in saved_opt['state'].items():
        require(entry.keys() == {'step', 'exp_avg', 'exp_avg_sq'}
                and entry['step'].numel() == 1 and entry['step'].item() == updates,
                f'Incorrect AdamW state or step for parameter {idx}')
        for name in ('exp_avg', 'exp_avg_sq'):
            moment = entry[name]
            require(moment.shape == parameters[idx].shape and moment.dtype == parameters[idx].dtype
                    and torch.isfinite(moment).all().item(), f'Invalid AdamW moment {idx}/{name}')
    expected_schedule = scheduler.state_dict()
    expected_schedule.update(last_epoch=updates, _step_count=updates + 1)
    require(state['scheduler'] == expected_schedule, 'Constant scheduler state mismatch')

    rng = state['rng']
    require(rng.keys() == {'python', 'numpy', 'torch', 'cuda'}
            and len(rng['cuda']) == torch.cuda.device_count(), 'RNG streams missing or CUDA device count differs')
    before = capture_rng()
    try:
        restore_rng(rng)
    finally:
        restore_rng(before)
    keys = model.load_state_dict(saved_model, strict=True)
    optimizer.load_state_dict(saved_opt)
    scheduler.load_state_dict(state['scheduler'])
    sampler.load_state_dict(saved_sampler)
    restore_rng(rng)
    return str(keys)


def save(path, model, optimizer, scheduler, sampler, progress, history, signature):
    require(all(p.grad is None for p in model.parameters()), 'Checkpoint requires a completed update')
    state = {'format': 'r1e-convergence-v1', 'signature': signature, 'model': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'sampler': sampler.state_dict(), 'progress': progress, 'history': history, 'rng': capture_rng()}
    temp = path.with_suffix('.pt.tmp')
    with temp.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def replace_best(last, best):
    temp = best.with_suffix('.pt.tmp')
    with last.open('rb') as source, temp.open('wb') as target:
        shutil.copyfileobj(source, target)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temp, best)


def gate(history, cfg):
    require([item['epoch'] for item in history] == [1, 2, 3], 'Gate needs epochs 1–3')
    previous = [item['validation'] for item in history[:-1]]
    current = history[-1]['validation']
    best_loss = min(x['mean_pixel_loss'] for x in previous)
    return {
        'overall_miou': current['miou_including_0'] >= max(x['miou_including_0'] for x in previous) - cfg['gate_max_miou_drop'],
        'foreground_miou': current['foreground_miou'] >= max(x['foreground_miou'] for x in previous) - cfg['gate_max_miou_drop'],
        'validation_loss': current['mean_pixel_loss'] <= best_loss + max(cfg['gate_loss_absolute_slack'],
                                                                      cfg['gate_loss_relative_slack'] * best_loss),
        'background_share': current['classes'][0]['predicted_fraction'] < cfg['gate_max_background_prediction'],
    }
