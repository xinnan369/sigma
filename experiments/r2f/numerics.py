"""Independent FP32 recurrence and native basis readout probes."""
import math

import torch

from experiments.r2b.state import error
from experiments.r2e.paths import direct


def native_scan(native, scan, *, coeff=None, skip=True):
    return native(scan.u, scan.delta, scan.a, scan.b, scan.c if coeff is None else coeff,
                  scan.d if skip else None, scan.bias, scan.softplus, scan.nrows)


def basis_states(native, scan):
    """Select one native state coordinate at each token with C=e_n, D absent."""
    h = scan.u.new_empty((*scan.u.shape[:2], scan.a.shape[1], scan.u.shape[-1]))
    c = torch.zeros_like(scan.c)
    for n in range(h.shape[2]):
        c[:, :, n].fill_(1)
        h[:, :, n] = native_scan(native, scan, coeff=c, skip=False)[0]
        c[:, :, n].zero_()
    return h


def readout(scan, states, *, fused=True):
    value = direct(scan)
    group_width = scan.u.shape[1] // scan.c.shape[1]
    for n in range(states.shape[2]):
        c = scan.c[:, :, n].repeat_interleave(group_width, dim=1)
        term = states[:, :, n]
        value = torch.addcmul(value, term, c) if fused else value + term * c
    return value


def sequential(scan, *, steps=None, exp2=False, fused=False, alternate_injection=False):
    """Direct t loop with independent FP32 operations and fixed group semantics."""
    dt = scan.step_sizes() if steps is None else steps
    group_width = scan.u.shape[1] // scan.b.shape[1]
    b = scan.b.repeat_interleave(group_width, dim=1)
    states = scan.u.new_empty((*scan.u.shape[:2], scan.a.shape[1], scan.u.shape[-1]))
    h = torch.zeros_like(states[..., 0])
    a = scan.a * math.log2(math.e) if exp2 else scan.a
    for t in range(scan.u.shape[-1]):
        d = dt[..., t, None]
        x = scan.u[..., t, None]
        bt = b[..., t]
        alpha = torch.exp2(d * a) if exp2 else torch.exp(d * a)
        injection = (d * bt) * x if alternate_injection else (d * x) * bt
        h = torch.addcmul(injection, alpha, h) if fused else alpha * h + injection
        states[..., t] = h
    return states


def summarize(got, ref):
    diff = (got - ref).abs()
    scaled = diff / (2e-5 + 2e-4 * ref.abs())

    def at(values):
        flat = values.flatten().argmax()
        index = tuple(int(i) for i in torch.unravel_index(flat, values.shape))
        return {'index': index, 'observed': float(got[index].item()), 'reference': float(ref[index].item()),
                'absolute': float(diff[index].item()), 'mixed_ratio': float(scaled[index].item())}

    return {**error(got, ref), 'exceedances': int((scaled > 1).sum().item()),
            'elements': ref.numel(), 'worst_absolute': at(diff), 'worst_mixed': at(scaled)}
