"""Width-only variants of the accepted R2b fusion blocks."""
from contextlib import contextmanager

from torch import nn

from experiments.r2b.controls import Attention, CNN, WIDTHS


class WideCNN(CNN):
    def __init__(self, channels, inner, *, cross):
        nn.Module.__init__(self)
        self.cross, self.inner = cross, inner
        self.mix = nn.Sequential(nn.Conv2d(2 * channels, inner, 1),
                                 nn.Conv2d(inner, inner, 3, padding=1, groups=inner),
                                 nn.GELU(), nn.Conv2d(inner, (2 if cross else 1) * channels, 1))

    def macs(self, h, w, c):
        return h * w * self.inner * ((4 if self.cross else 3) * c + 9)


class WideAttention(Attention):
    def __init__(self, channels, inner, *, cross):
        nn.Module.__init__(self)
        self.cross, self.inner = cross, inner
        if inner % 4:
            raise ValueError('Four heads require width divisible by four')
        self.query = nn.Linear(channels, inner)
        self.key_value = nn.Linear(channels, 2 * inner)
        self.output = nn.Linear(inner, channels)

    def macs(self, h, w, c):
        directions, tokens = (2, 64) if self.cross else (1, 128)
        return directions * self.inner * (2 * h * w * c + 2 * tokens * c + 2 * h * w * tokens)


def fusion_parameters(family, widths):
    if family == 'cnn':
        return sum((7 * c + 22) * m + 3 * c for c, m in zip(WIDTHS, widths, strict=True))
    if family == 'attention':
        return sum((8 * c + 6) * m + 2 * c for c, m in zip(WIDTHS, widths, strict=True))
    raise ValueError(family)


def search(family, config, target=14_371_200):
    s = config['width_search']
    align, denom = s['alignment'], s['denominator']
    candidates = []
    for q in range(s['min_numerator'], s['max_numerator'] + 1):
        widths = [align * ((2 * c * q + align * denom) // (2 * align * denom)) for c in WIDTHS]
        if min(widths) >= align:
            count = fusion_parameters(family, widths)
            candidates.append((abs(count - target), q, widths, count))
    error, multiplier, widths, count = min(candidates)
    return {'numerator': multiplier, 'widths': widths, 'fusion_parameters': count,
            'absolute_fusion_parameter_difference': error, 'candidate_count': len(candidates)}


@contextmanager
def fusion(model, family, widths):
    if family == 'ssm':
        yield
        return
    cls = {'cnn': WideCNN, 'attention': WideAttention}[family]
    backbone = model.backbone
    previous = backbone.cross_mamba, backbone.channel_attn_mamba
    backbone.cross_mamba, backbone.channel_attn_mamba = (
        nn.ModuleList(cls(c, m, cross=cross) for c, m in zip(WIDTHS, widths[family], strict=True))
        for cross in (True, False))
    try:
        yield
    finally:
        backbone.cross_mamba, backbone.channel_attn_mamba = previous
