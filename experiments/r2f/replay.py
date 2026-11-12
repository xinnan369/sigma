"""Reconstruct hybrids from CUDA state probes using explicit FP32 FMA."""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from experiments.r2b.state import check_close
from experiments.r2e.paths import direct
from experiments.r2f.numerics import basis_states, native_scan, summarize


@triton.jit
def accumulate(H, C, Skip, Out, L: tl.constexpr, D: tl.constexpr, N: tl.constexpr,
               G: tl.constexpr, Size: tl.constexpr, Block: tl.constexpr):
    j = tl.program_id(0) * Block + tl.arange(0, Block)
    t = j % L
    d = j // L % D
    b = j // (D * L)
    g = d // (D // G)
    valid = j < Size
    value = tl.load(Skip + j, valid, other=0.)
    for n in tl.static_range(N):
        h = tl.load(H + ((b * D + d) * N + n) * L + t, valid, other=0.)
        c = tl.load(C + ((b * G + g) * N + n) * L + t, valid, other=0.)
        value = tl.math.fma(h, c, value)
    tl.store(Out + j, value, valid)


def reconstruct(h, c, skip):
    b, d, n, length = h.shape
    out = torch.empty_like(skip)
    accumulate[(triton.cdiv(out.numel(), 256),)](
        h.contiguous(), c.contiguous(), skip.contiguous(), out, length, d, n, c.shape[1], out.numel(), 256)
    return out


def rounded_reference(h, c, skip):
    """Evaluate each FMA in FP64 and round to FP32 after each state term."""
    result = skip
    width = h.shape[1] // c.shape[1]
    for n in range(h.shape[2]):
        state = h[:, :, n].reshape(h.shape[0], c.shape[1], width, h.shape[-1])
        product = state.double() * c[:, :, n, None].double()
        result = (product.reshape_as(result) + result.double()).float()
    return result


@dataclass
class Validated:
    outputs: tuple
    audit: dict

    def output(self, mask):
        return self.outputs[mask]


def validated_factor(native, clean, corrupt, clean_y, corrupt_y):
    scans = (clean, corrupt)
    states = [basis_states(native, scan) for scan in scans]
    skips = [direct(scan) for scan in scans]
    endpoints = [clean_y, corrupt_y]
    native_metrics = []
    for scan, h, skip, actual in zip(scans, states, skips, endpoints, strict=True):
        _, prefix = native_scan(native, scan)
        ends = [min(i + 2048, h.shape[-1]) - 1 for i in range(0, h.shape[-1], 2048)]
        native_metrics.append({
            'chunk_end': check_close(h[..., ends].transpose(-1, -2), prefix[..., 1::2], 'CUDA state prefix'),
            'readout': check_close(reconstruct(h, scan.c, skip), actual, 'CUDA native readout'),
        })
    outputs, tests = [], []
    for mask in range(8):
        h = states[mask & 1]
        c = scans[(mask >> 1) & 1].c
        skip = skips[(mask >> 2) & 1]
        mixed = reconstruct(h, c, skip)
        reference = rounded_reference(h, c, skip)
        metric = check_close(mixed, reference, f'Independent rounded FMA, mask {mask}')
        if mask in (0, 7):
            actual = endpoints[mask == 7]
            check_close(actual, reference, f'Native endpoint, mask {mask}')
            outputs.append(actual)
        else:
            outputs.append(mixed)
        tests.append(metric)
    return Validated(tuple(outputs), {'native': native_metrics, 'hybrid': tests})


def verify_validated(clean, corrupt, clean_y, corrupt_y, clean_prefix, corrupt_prefix, factorial):
    return factorial.audit
