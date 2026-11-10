"""Scope CUDA calls to named fusion modules while leaving the native model intact."""
from contextlib import contextmanager
from unittest.mock import patch

import torch

from experiments.r2a.hooks import modules
from experiments.r2b.audit import compare
from experiments.r2b.state import Scan, check_close, coordinates
from experiments.r2b.triton_state import trajectory


@contextmanager
def intercept(model, extension, callback):
    original = extension.fwd
    active, seen, handles = [], {}, []

    def before(name):
        def hook(module, inputs):
            if active or name in seen or len(inputs) != 2 or inputs[0].ndim != 4:
                raise ValueError('Unexpected nested or repeated fusion call')
            seen[name] = 0
            active.append((name, inputs[0].shape[1:3]))
        return hook

    def after(module, inputs, outputs):
        active.pop()

    def forward(*args):
        out, prefix = original(*args)
        if active:
            name, grid = active[0]
            index = seen[name]
            seen[name] += 1
            out = callback(name, grid, index, args, out, prefix)
        return out, prefix

    try:
        for name, module in modules(model).items():
            handles += [module.register_forward_pre_hook(before(name)), module.register_forward_hook(after)]
        with patch.object(extension, 'fwd', forward):
            yield
        if active or seen != {name: 2 if name.startswith('cross') else 1 for name in modules(model)}:
            raise ValueError(f'Fusion CUDA call counts changed: {seen}')
    finally:
        for handle in reversed(handles):
            handle.remove()


def unpack(name, grid, index, args):
    scan = Scan(*args)
    length = grid[0] * grid[1]
    kind = name.split('.')[0]
    shape = (scan.b.shape[1], scan.u.shape[-1], scan.nrows)
    if kind == 'cross':
        if shape != (1, length, 1) or index not in (0, 1):
            raise ValueError(f'Unexpected CrossMamba scan: {shape}/{index}')
    elif shape != (2, 2 * length, 4) or index != 0:
        raise ValueError(f'Unexpected ConcatMamba scan: {shape}/{index}')
    return scan


def reset_readout(scan, h, native, direction, t, target, *, verify):
    """Replace h[t] and propagate the exact homogeneous correction downstream."""
    directions = scan.b.shape[1]
    width = scan.u.shape[1] // directions
    start, end = direction * width, (direction + 1) * width
    if target.shape != h[..., t].shape:
        raise ValueError('Clean state dimension differs')
    offset = target - h[..., t]
    correction = trajectory(scan, reset=(t, start, end, offset), homogeneous=True)
    if torch.count_nonzero(correction[..., :t]).item():
        raise ValueError('State correction appeared before the selected time')
    if torch.count_nonzero(correction[:, :start]).item() or torch.count_nonzero(correction[:, end:]).item():
        raise ValueError('State correction leaked into another direction')
    if not torch.equal(correction[:, start:end, :, t], offset[:, start:end]):
        raise ValueError('Incorrect correction at reset')
    delta = scan.readout(correction)
    patched = native + delta
    metrics = {'time': t, 'direction': direction, 'changed_reset_state': int(torch.count_nonzero(
                   offset[:, start:end]).item()),
               'changed_future_state_steps': int(torch.count_nonzero(
                   correction[:, start:end, :, t+1:].abs().amax(dim=(0, 1, 2))).item()),
               'changed_readout': int(torch.count_nonzero(patched - native).item()),
               'readout_delta_max': float(delta.abs().amax().item())}
    if verify:
        explicit = trajectory(scan, reset=(t, start, end, target))
        if not torch.equal(explicit[:, start:end, :, t], target[:, start:end]):
            raise ValueError('Explicit replacement did not set the clean h[t]')
        metrics['explicit_vs_correction'] = check_close(h + correction, explicit, 'Reset recurrence')
    return patched, metrics


def summaries(kind, grid, index, scan, h, clean, clean_c, block):
    height, width = grid
    groups = scan.b.shape[1]
    rows = []
    for k, (current, ref) in enumerate(zip(h.chunk(groups, dim=1), clean.chunk(groups, dim=1), strict=True)):
        direction = index if kind == 'cross' else k
        per_t = (current - ref).square().mean(dim=(0, 1, 2)).sqrt()
        coefficients = (scan.c[:, k] - clean_c[:, k]).square().mean().sqrt()
        length = per_t.numel()
        zero = int(torch.count_nonzero(per_t).item()) == 0
        if kind == 'cross' and index == 0 and not zero:
            raise ValueError('Thermal changed CrossMamba RGB transition state')
        rows.append({'direction': direction, 'length': length,
                     'coordinate_start': coordinates(kind, direction, 0, height, width),
                     'coordinate_end': coordinates(kind, direction, length - 1, height, width),
                     'rms_state_difference': float(per_t.square().mean().sqrt().item()),
                     'max_per_t_rms': float(per_t.amax().item()),
                     'max_time': int(per_t.argmax().item()),
                     'readout_c_rms_difference': float(coefficients.item()),
                     'nonzero_state_steps': int(torch.count_nonzero(per_t).item()),
                     'bin_mean_step_rms': [float(x.mean().item()) for x in per_t.split(block)]})
    return rows
