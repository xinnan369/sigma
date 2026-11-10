"""Materialize h[t] using a tiled associative prefix of FP32 affine steps."""
import torch
import triton
import triton.language as tl


@triton.jit
def pack(a, b):
    return (a.to(tl.uint32, bitcast=True).to(tl.uint64) << 32) | b.to(tl.uint32, bitcast=True).to(tl.uint64)


@triton.jit
def unpack(v):
    return (v >> 32).to(tl.uint32).to(tl.float32, bitcast=True), v.to(tl.uint32).to(tl.float32, bitcast=True)


@triton.jit
def compose(left, right):
    a, b = unpack(left)
    c, d = unpack(right)
    return pack(c * a, c * b + d)


@triton.jit
def recurrence(u, dt, a, b, h, reset, length: tl.constexpr, dim: tl.constexpr,
               groups: tl.constexpr, states: tl.constexpr, block: tl.constexpr,
               reset_t: tl.constexpr, first_channel: tl.constexpr, end_channel: tl.constexpr,
               homogeneous: tl.constexpr):
    p = tl.program_id(0)
    n = p % states
    channel = p // states % dim
    batch = p // (states * dim)
    group = channel // (dim // groups)
    decay = tl.load(a + channel * states + n)
    t = tl.arange(0, block)
    carry = 0.0
    for start in range(0, length, block):
        step = start + t
        offset = (batch * dim + channel) * length + step
        valid = step < length
        delta = tl.load(dt + offset, valid, other=0.0)
        alpha = tl.exp(delta * decay)
        if homogeneous:
            forcing = tl.full((block,), 0, tl.float32)
        else:
            drive = tl.load(u + offset, valid, other=0.0)
            coeff = tl.load(b + ((batch * groups + group) * states + n) * length + step, valid, other=0.0)
            forcing = (delta * drive) * coeff
        if reset_t >= 0:
            hit = (step == reset_t) & (channel >= first_channel) & (channel < end_channel)
            value = tl.load(reset + p)
            alpha = tl.where(hit, 0.0, alpha)
            forcing = tl.where(hit, value, forcing)
        # Triton 2.2 accepts one tensor per scan. Pack two FP32 bit patterns losslessly.
        mul, add = unpack(tl.associative_scan(pack(alpha, forcing), 0, compose))
        current = mul * carry + add
        tl.store(h + p * length + step, current, valid)
        carry = tl.sum(tl.where(t == block - 1, current, 0.0), 0)


def trajectory(scan, *, reset=None, homogeneous=False):
    batch, dim, length = scan.u.shape
    states = scan.a.shape[1]
    out = scan.u.new_empty((batch, dim, states, length))
    if reset is None:
        t, start, end, values = -1, 0, 0, scan.u
    else:
        t, start, end, values = reset
        if not (0 <= t < length and 0 <= start < end <= dim and values.shape == out.shape[:-1]
                and values.device == out.device and values.dtype == out.dtype):
            raise ValueError('Invalid state reset')
        values = values.contiguous()
    recurrence[(batch * dim * states,)](
        scan.u, scan.step_sizes().contiguous(), scan.a, scan.b, out, values,
        length, dim, scan.b.shape[1], states, 128, t, start, end, homogeneous, num_warps=4,
        enable_fp_fusion=False)
    return out
