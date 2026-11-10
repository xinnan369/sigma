"""FP32 diagonal state recurrence and causal reset of the post-update state."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Scan:
    u: torch.Tensor
    delta: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    d: torch.Tensor | None
    bias: torch.Tensor | None
    softplus: bool
    nrows: int

    def __post_init__(self):
        self.u, self.delta, self.a = (x.contiguous() for x in (self.u, self.delta, self.a))
        self.b, self.c = ((x[:, None] if x.ndim == 3 else x).contiguous() for x in (self.b, self.c))
        batch, channels, length = self.u.shape
        if (self.delta.shape != self.u.shape or self.a.shape[0] != channels
                or self.b.shape != self.c.shape or self.b.shape[0] != batch
                or self.b.shape[2:] != (self.a.shape[1], length)
                or channels % (self.b.shape[1] * self.nrows)
                or any(x.dtype != torch.float32 or x.device != self.u.device
                       for x in (self.u, self.delta, self.a, self.b, self.c, self.d, self.bias) if x is not None)
                or any(x.shape != (channels,) for x in (self.d, self.bias) if x is not None)):
            raise ValueError('Unexpected grouped FP32 selective scan signature')

    def step_sizes(self):
        delta = self.delta if self.bias is None else self.delta + self.bias[None, :, None]
        return F.softplus(delta) if self.softplus else delta

    def readout(self, states):
        batch, groups, n, length = self.c.shape
        y = (states.reshape(batch, groups, -1, n, length) * self.c[:, :, None]).sum(dim=-2)
        return y.reshape_as(self.u)


def literal(scan, *, reset=None):
    """Small, independent sequential implementation, including actual h[t]."""
    delta = scan.step_sizes()
    group_width = scan.u.shape[1] // scan.b.shape[1]
    b = scan.b.repeat_interleave(group_width, dim=1)
    h = scan.u.new_zeros((*scan.u.shape[:2], scan.a.shape[1]))
    states = []
    for t in range(scan.u.shape[-1]):
        dt = delta[:, :, t, None]
        h = torch.exp(dt * scan.a) * h + (dt * scan.u[:, :, t, None]) * b[..., t]
        if reset is not None and t == reset[0]:
            h = reset[1]
        states.append(h)
    return torch.stack(states, dim=-1)


def outputs(scan, h):
    y = scan.readout(h)
    return y if scan.d is None else y + scan.u * scan.d[None, :, None]


def error(got, expected):
    diff = (got - expected).float()
    peak = expected.abs().amax().clamp_min(1e-8)
    rms = expected.square().mean().sqrt().clamp_min(1e-8)
    metrics = torch.stack((diff.abs().amax(), diff.abs().amax() / peak,
                           diff.square().mean().sqrt() / rms,
                           (diff.abs() / (2e-5 + 2e-4 * expected.abs())).amax())).tolist()
    return dict(zip(('max_abs', 'peak_relative', 'rms_relative', 'max_mixed_ratio'), metrics))


def check_close(got, expected, label, *, mixed=1., rms=2e-5, peak=3e-5):
    if got.shape != expected.shape:
        raise ValueError(f'{label}: shape differs')
    m = error(got, expected)
    if not (m['max_mixed_ratio'] <= mixed and m['rms_relative'] <= rms and m['peak_relative'] <= peak):
        raise ValueError(f'{label}: {m}')
    return m


def coordinates(kind, direction, t, height, width):
    """Map zero-based scan t to modality and geometric feature-grid (row, col)."""
    n = height * width
    if kind == 'concat':
        if direction not in (0, 1) or not 0 <= t < 2 * n:
            raise ValueError('Invalid Concat scan index')
        linear = t if direction == 0 else 2 * n - 1 - t
        modality, index = divmod(linear, n)
        return ('rgb', 'thermal')[modality], *divmod(index, width)
    if kind == 'cross':
        if direction not in (0, 1) or not 0 <= t < n:
            raise ValueError('Invalid Cross scan index')
        return ('rgb', 'thermal')[direction], *divmod(t, width)
    if kind != 'four' or direction not in range(4) or not 0 <= t < n:
        raise ValueError('Invalid four-direction scan index')
    index = n - 1 - t if direction >= 2 else t
    if direction % 2:
        col, row = divmod(index, height)
        return 'feature', row, col
    return 'feature', *divmod(index, width)
