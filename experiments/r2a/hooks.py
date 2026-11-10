"""Per-image BHWC snapshots, online responses and strict output replacement."""
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def modules(model):
    result = {}
    for alias, group in [('cross', model.backbone.cross_mamba), ('concat', model.backbone.channel_attn_mamba)]:
        if len(group) != 4:
            raise ValueError('Expected four actual fusion scales')
        result.update({f'{alias}.{i}': module for i, module in enumerate(group)})
    return result


def tensors(value, size):
    entries = value if isinstance(value, tuple) else (value,)
    if len(entries) != size or any(not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[0] != 1
                                   for x in entries):
        raise ValueError('Unexpected BHWC tuple structure')
    return entries


def match(clean, current):
    if clean.shape != current.shape or clean.dtype != current.dtype or clean.device != current.device:
        raise ValueError('Replacement or response tensor signature mismatch')


def response(clean, current, mask):
    match(clean, current)
    # With the fixed image dimensions, max pooling conservatively marks any
    # grid cell whose geometric bin intersects the image-space ROI.
    mapped = F.adaptive_max_pool2d(mask, clean.shape[1:3]).bool()[0, 0]
    delta = (current.float() - clean.float()).norm(dim=-1)[0]
    clean_rms = clean.float().square().sum(-1).mean().sqrt()
    normalized = delta / clean_rms.clamp_min(1e-8)
    total = normalized.sum()
    inside = (normalized * mapped).sum()
    outside = (normalized * ~mapped).sum()
    values = torch.stack([normalized.square().mean().sqrt(), inside, outside, total,
                          mapped.sum().float(), clean_rms]).tolist()
    if not all(float('-inf') < x < float('inf') for x in values):
        raise FloatingPointError('Nonfinite fusion response')
    rms, inside, outside, total, n_inside, clean_rms = values
    n_out = delta.numel() - n_inside
    return {'shape_bhwc': list(clean.shape), 'normalized_l2': rms,
            'outside_response_ratio': outside / total if total else None,
            'outside_mean': outside / n_out if n_out else None, 'inside_mean': inside / n_inside if n_inside else None,
            'outside_area_fraction': n_out / delta.numel(), 'inside_cells': int(n_inside), 'clean_rms': clean_rms}


@dataclass
class Snapshot:
    sample: str
    values: dict


@contextmanager
def observe(mapping, sample, *, reference=None, mask=None):
    if reference is not None and (reference.sample != sample or mask is None):
        raise ValueError('Response reference sample or mask differs')
    seen, handles, records = {}, [], []
    snap = Snapshot(sample, {})

    def visit(name, side, value, expected):
        key = (name, side)
        if key in seen:
            raise ValueError(f'Unexpected repeated module call {key}')
        seen[key] = True
        values = tensors(value, expected)
        if reference is None:
            snap.values[key] = tuple(x.detach().clone() for x in values)
        else:
            for index, (a, b) in enumerate(zip(reference.values[key], values, strict=True)):
                records.append({'module': name, 'slot': f'{side}{index}', **response(a, b, mask)})

    def pre(name):
        def fn(module, args):
            visit(name, 'in', args, 2)
        return fn

    def post(name):
        def fn(module, args, out):
            if name.startswith('cross') and not isinstance(out, tuple):
                raise ValueError('CrossMamba must return both branches')
            visit(name, 'out', out, 2 if name.startswith('cross') else 1)
        return fn

    try:
        for name, module in mapping.items():
            handles.append(module.register_forward_pre_hook(pre(name)))
            handles.append(module.register_forward_hook(post(name)))
        yield snap if reference is None else records
        if len(seen) != 2 * len(mapping):
            raise ValueError('Each fusion module must be called exactly once')
    finally:
        for h in reversed(handles):
            h.remove()


@contextmanager
def replace(module, name, snapshot, sample):
    if snapshot.sample != sample:
        raise ValueError('Replacement comes from a different sample')
    saved = snapshot.values[(name, 'out')]
    calls = 0

    def hook(mod, args, output):
        nonlocal calls
        calls += 1
        if calls != 1:
            raise ValueError('Unexpected repeated replacement call')
        cross = name.startswith('cross')
        if cross != isinstance(output, tuple):
            raise ValueError('Incorrect replacement output structure')
        values = tensors(output, 2 if cross else 1)
        for a, b in zip(saved, values, strict=True):
            match(a, b)
        result = tuple(x.detach().clone() for x in saved)
        return result if cross else result[0]

    handle = module.register_forward_hook(hook)
    try:
        yield
        if calls != 1:
            raise ValueError('Replacement module was not called')
    finally:
        handle.remove()
