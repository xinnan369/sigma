"""Output capture/replacement helper for initial module-level interventions.

Not a selective-scan hidden-state hook. Never claim direct observation of h_t.
"""
from __future__ import annotations
import torch


def _clone_structure(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    if isinstance(x, (tuple, list)):
        return type(x)(_clone_structure(y) for y in x)
    raise TypeError(f'Unsupported module output type: {type(x)}')


def _match_structure(x, y):
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        if x.shape != y.shape:
            raise ValueError(f'Tensor shape mismatch: {x.shape} vs {y.shape}')
        return y.to(device=x.device, dtype=x.dtype).detach().clone()
    if isinstance(x, (tuple, list)) and type(x) is type(y) and len(x)==len(y):
        return type(x)(_match_structure(a,b) for a,b in zip(x,y))
    raise TypeError('Incompatible replacement output structure')


class _HookContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.hook.remove()


class CaptureOutput(_HookContext):
    """Use per sample, then release outputs; no full-dataset feature cache."""
    def __init__(self, module):
        self.outputs = []
        self.hook = module.register_forward_hook(self._fn)

    def _fn(self, module, args, output):
        self.outputs.append(_clone_structure(output))


class ReplaceOutput(_HookContext):
    def __init__(self, module, replacement):
        self.replacement = _clone_structure(replacement)
        self.calls = 0
        self.hook = module.register_forward_hook(self._fn)

    def _fn(self, module, args, output):
        self.calls += 1
        return _match_structure(output, self.replacement)
