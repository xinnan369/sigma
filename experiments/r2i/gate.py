"""Parameter- and operation-matched residual weighting before CrossMamba."""
from contextlib import contextmanager

import torch
from torch import nn


class Gate(nn.Module):
    def __init__(self, reliability):
        super().__init__()
        self.reliability = reliability
        self.score = nn.Linear(3, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, rgb, thermal):
        other = thermal if self.reliability else rgb
        r, t = rgb.abs().mean(-1), other.square().mean(-1)
        inputs = (r, t, (rgb[..., :-1] * other[..., 1:]).mean(-1))
        delta = torch.tanh(self.score(torch.stack(inputs, dim=-1)))
        return rgb * (1 - delta), thermal * (1 + delta)


@contextmanager
def attach(model, arm):
    if arm == 'baseline':
        yield
        return
    if arm not in ('gated', 'control') or len(model.backbone.cross_mamba) != 4:
        raise ValueError('Unexpected fusion arm or scale count')
    model.backbone.r2i_gate = nn.ModuleList(Gate(arm == 'gated') for _ in range(4))

    def prehook(gate):
        def apply(_, inputs):
            return gate(*inputs)
        return apply

    handles = [module.register_forward_pre_hook(prehook(gate))
               for module, gate in zip(model.backbone.cross_mamba, model.backbone.r2i_gate, strict=True)]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        del model.backbone.r2i_gate
