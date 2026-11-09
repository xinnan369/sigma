"""Explicit epoch boundaries and durable optimizer-boundary state."""
import os
from pathlib import Path

import torch

from experiments.r1c.state import capture_rng, restore_rng


class EpochSampler:
    def __init__(self, size, seed):
        if size < 1:
            raise ValueError('Training split is empty')
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch = 0
        self.cursor = size
        self.advance_epoch()

    def advance_epoch(self):
        if self.cursor != self.size:
            raise ValueError('Finish the current epoch before starting another')
        self.order = torch.randperm(self.size, generator=self.generator).tolist()
        self.cursor = 0
        self.epoch += 1

    def next_window(self, accumulation):
        if accumulation < 1:
            raise ValueError('Accumulation must be positive')
        end = min(self.size, self.cursor + accumulation)
        indices = self.order[self.cursor:end]
        self.cursor = end
        return indices

    def state_dict(self):
        return {'size': self.size, 'epoch': self.epoch, 'cursor': self.cursor,
                'order': self.order.copy(), 'generator': self.generator.get_state()}

    def load_state_dict(self, state):
        if (state['size'] != self.size or state['epoch'] < 1
                or not 0 <= state['cursor'] <= self.size
                or sorted(state['order']) != list(range(self.size))):
            raise ValueError('Invalid epoch sampler state')
        self.generator.set_state(state['generator'])
        self.epoch, self.cursor = state['epoch'], state['cursor']
        self.order = state['order'].copy()


def save(path, model, optimizer, scheduler, sampler, progress, signature):
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Save only at a completed optimizer boundary after zero_grad')
    state = {'format': 'r1d-epoch-v1', 'signature': signature, 'model': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'sampler': sampler.state_dict(), 'progress': progress, 'rng': capture_rng()}
    path = Path(path)
    temporary = path.with_suffix('.pt.tmp')
    with temporary.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load(path, model, optimizer, scheduler, sampler, signature):
    # This format contains pickle RNG state. Load only a locally generated checkpoint.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['format'] != 'r1d-epoch-v1' or state['signature'] != signature:
        raise ValueError('R1d checkpoint signature mismatch')
    progress = state['progress']
    if state['scheduler']['last_epoch'] != progress['updates']:
        raise ValueError('Checkpoint scheduler/update mismatch')
    keys = model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    sampler.load_state_dict(state['sampler'])
    restore_rng(state['rng'])
    return progress, str(keys)
