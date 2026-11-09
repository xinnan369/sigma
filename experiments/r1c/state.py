"""Complete optimizer coverage and update-boundary checkpoint state."""
from __future__ import annotations

import os
from pathlib import Path
import random
import numpy as np
import torch


def accumulate_loss(mean_loss, valid_pixels):
    if valid_pixels <= 0:
        raise ValueError('Microbatch has no supervised pixels')
    (mean_loss * valid_pixels).backward()


def normalize_gradients(model, valid_pixels):
    if valid_pixels <= 0:
        raise ValueError('Accumulation window has no supervised pixels')
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(valid_pixels)


def complete_groups(model, legacy_groups, lr, decay):
    """Preserve every legacy assignment; add missed raw Parameters explicitly."""
    named = dict(model.named_parameters())
    by_id = {id(p): n for n, p in named.items() if p.requires_grad}
    assignments, groups = {}, []
    for old in legacy_groups:
        parameters = []
        wd = old.get('weight_decay', decay)
        for p in old['params']:
            if not p.requires_grad:
                continue
            if id(p) not in by_id or id(p) in assignments:
                raise ValueError('Foreign or duplicate optimizer parameter')
            assignments[id(p)] = ('legacy', wd)
            parameters.append(p)
        groups.append({'params': parameters, 'lr': lr, 'weight_decay': wd})
    missing = [(n, p) for n, p in named.items() if p.requires_grad and id(p) not in assignments]
    added = {0.0: [], decay: []}
    for n, p in missing:
        wd = 0.0 if getattr(p, '_no_weight_decay', False) or p.ndim <= 1 or n.endswith('bias') else decay
        added[wd].append(p)
        assignments[id(p)] = ('added', wd)
    groups.extend({'params': ps, 'lr': lr, 'weight_decay': wd} for wd, ps in added.items() if ps)
    flat = [id(p) for g in groups for p in g['params']]
    if len(flat) != len(set(flat)) or set(flat) != set(by_id):
        raise ValueError('Optimizer must cover every trainable tensor exactly once')
    records = [{'name': n, 'numel': p.numel(), 'shape': list(p.shape),
                'origin': assignments[id(p)][0], 'weight_decay': assignments[id(p)][1]}
               for n, p in named.items() if p.requires_grad]
    return groups, {'total_numel': sum(p.numel() for p in named.values() if p.requires_grad),
                    'legacy_missing_numel': sum(p.numel() for _, p in missing),
                    'tensor_count': len(records), 'duplicate_count': 0, 'missing_count': 0,
                    'parameters': records}


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


class Sampler:
    """No worker prefetch; the permutation and cursor describe the next microbatch."""
    def __init__(self, size, seed):
        if size <= 0:
            raise ValueError('Nonempty dataset required')
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator).tolist()
        self.cursor, self.epoch = 0, 0

    def next(self):
        if self.cursor == self.size:
            self.epoch += 1
            self.order = torch.randperm(self.size, generator=self.generator).tolist()
            self.cursor = 0
        value = self.order[self.cursor]
        self.cursor += 1
        return value

    def state_dict(self):
        return {'size': self.size, 'order': self.order, 'cursor': self.cursor,
                'epoch': self.epoch, 'generator': self.generator.get_state()}

    def load_state_dict(self, state):
        if state['size'] != self.size or sorted(state['order']) != list(range(self.size)) or not 0 <= state['cursor'] <= self.size:
            raise ValueError('Invalid sampler checkpoint')
        self.order, self.cursor, self.epoch = state['order'], state['cursor'], state['epoch']
        self.generator.set_state(state['generator'])


def save_checkpoint(path, model, optimizer, sampler, update, best, signature):
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Checkpoint only at a completed update boundary after zero_grad')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    state = {'format_version': 1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'sampler': sampler.state_dict(), 'update': update, 'best': best, 'rng': capture_rng(),
             'signature': signature, 'accumulation_pending': 0, 'scaler': None,
             'scheduler': {'kind': 'constant-pilot-lr', 'completed_updates': update}}
    with temporary.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_checkpoint(path, model, optimizer, sampler, signature):
    # Only load trusted, locally produced pilot checkpoints. Full RNG needs pickle.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['format_version'] != 1 or state['signature'] != signature or state['accumulation_pending'] != 0:
        raise ValueError('Checkpoint provenance/configuration/state mismatch')
    if state['scheduler'] != {'kind': 'constant-pilot-lr', 'completed_updates': state['update']}:
        raise ValueError('Scheduler/update mismatch')
    keys = model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    sampler.load_state_dict(state['sampler'])
    restore_rng(state['rng'])
    return state['update'], state['best'], str(keys)
