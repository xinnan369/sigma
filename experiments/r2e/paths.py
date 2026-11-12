"""Factorial intervention at the selective scan readout boundary."""
from contextlib import contextmanager
from dataclasses import dataclass
from unittest.mock import patch

import torch

from experiments.r1g.partition import require
from experiments.r2a.hooks import modules
from experiments.r2b.state import Scan, check_close, coordinates


def direct(scan):
    return torch.zeros_like(scan.u) if scan.d is None else scan.u * scan.d[None, :, None]


@dataclass
class Readouts:
    clean: torch.Tensor
    corrupt: torch.Tensor
    h_only: torch.Tensor
    c_only: torch.Tensor
    direct_delta: torch.Tensor

    def output(self, mask):
        require(mask in range(8), 'Invalid h/C/D mask')
        if mask == 7:
            return self.corrupt
        readout = (self.clean, self.h_only, self.c_only, self.corrupt)[mask & 3]
        # At h=1 the original native call included the corrupt direct term.
        if bool(mask & 1) == bool(mask & 4):
            return readout
        return readout + self.direct_delta if mask & 4 else readout - self.direct_delta


def factor(native, clean, corrupt, clean_y, corrupt_y):
    require(clean.u.shape == corrupt.u.shape and clean.c.shape == corrupt.c.shape
            and clean.nrows == corrupt.nrows and clean.softplus == corrupt.softplus
            and torch.equal(clean.a, corrupt.a) and torch.equal(clean.d, corrupt.d)
            and torch.equal(clean.bias, corrupt.bias), 'Frozen scan parameter or shape changed')

    def read(scan, c):
        return native(scan.u, scan.delta, scan.a, scan.b, c, scan.d, scan.bias, scan.softplus, scan.nrows)[0]

    return Readouts(clean_y, corrupt_y, read(corrupt, clean.c), read(clean, corrupt.c),
                    direct(corrupt) - direct(clean))


def mobius(values, mask):
    return sum((-1) ** (mask.bit_count() - i.bit_count()) * values[i]
               for i in range(8) if i & mask == i)


def geometry(scope, index, grid, scan):
    h, w = grid
    kind = scope.split('.')[0] if scope.startswith(('cross.', 'concat.')) else 'four'
    directions = {'cross': (index,), 'concat': (0, 1), 'four': (0, 1, 2, 3)}[kind]
    length = h * w * (2 if kind == 'concat' else 1)
    require(scan.u.shape[-1] == length and scan.c.shape[1] == len(directions), 'Scan layout differs')
    return {'layout': kind, 'grid': list(grid), 'directions': list(directions), 'length': length,
            'state_dim': scan.a.shape[-1], 'channels': scan.u.shape[1], 'nrows': scan.nrows,
            'endpoints': [[coordinates(kind, d, t, h, w) for t in (0, length - 1)] for d in directions]}


def watched(model, ss2d, family):
    found = {}
    for name, module in model.named_modules():
        if isinstance(module, ss2d):
            scope = 'encoder' if name.startswith('backbone.vssm.') else 'decoder'
            require(scope == 'encoder' or name.startswith('decode_head.'), f'Unknown SS2D path {name}')
            found[name] = (scope, module)
    if family == 'ssm':
        found.update({name: (name, m) for name, m in modules(model).items()})
    return found


@contextmanager
def intercept(model, vmamba, family, callback, module_output=None):
    native = vmamba.selective_scan_cuda.fwd
    active, handles, seen = [], [], {}
    paths = watched(model, vmamba.SS2D, family)

    def pre(name, scope):
        def hook(_, args):
            require(not active and args[0].ndim == 4, f'Unexpected scan nesting at {name}')
            seen[name] = seen.get(name, -1) + 1
            active.append([name, scope, tuple(args[0].shape[1:3]), seen[name], 0])
        return hook

    def post(_, args, output):
        name, scope, grid, invocation, calls = active.pop()
        require(calls == (2 if scope.startswith('cross.') else 1), f'Unexpected scan count at {name}')
        if module_output:
            module_output((name, invocation), output)

    def forward(*args):
        require(len(active) == 1, 'Selective scan is outside the frozen module map')
        name, scope, grid, invocation, index = active[0]
        active[0][-1] += 1
        return callback((name, invocation, index), scope, grid, native, args)

    try:
        for name, (scope, module) in paths.items():
            handles += [module.register_forward_pre_hook(pre(name, scope)), module.register_forward_hook(post)]
        with patch.object(vmamba.selective_scan_cuda, 'fwd', forward):
            yield
        require(not active and set(seen) == set(paths)
                and all(i == (1 if paths[n][0] == 'encoder' else 0) for n, i in seen.items()),
                'Incomplete or unexpected encoder/decoder/fusion invocations')
    finally:
        for handle in reversed(handles):
            handle.remove()


def verify_reference(clean, corrupt, clean_y, corrupt_y, clean_prefix, corrupt_prefix, factorial):
    from experiments.r2b.triton_state import trajectory

    def readout_fn(scan, h, skip):
        # The CUDA kernel initializes D*x and accumulates state components in
        # state-index order. Match that order instead of a reduction tree.
        group_width = h.shape[1] // scan.c.shape[1]
        value = direct(skip)
        for n in range(h.shape[2]):
            c = scan.c[:, :, n].repeat_interleave(group_width, dim=1)
            value = torch.addcmul(value, h[:, :, n], c)
        return value

    def audit(scan, h, native, prefix):
        length = h.shape[-1]
        ends = [min(i + 2048, length) - 1 for i in range(0, length, 2048)]
        return {'chunk_end_h': check_close(h[..., ends].transpose(-1, -2), prefix[..., 1::2], 'CUDA chunk-end h'),
                'output': check_close(readout_fn(scan, h, scan), native, 'CUDA output'),
                'chunk_end_indices': ends}

    hc, hx = trajectory(clean), trajectory(corrupt)
    audits = [audit(clean, hc, clean_y, clean_prefix), audit(corrupt, hx, corrupt_y, corrupt_prefix)]
    mixtures = []
    for m in range(8):
        state = hx if m & 1 else hc
        readout = corrupt if m & 2 else clean
        skip = corrupt if m & 4 else clean
        mixtures.append(check_close(factorial.output(m), readout_fn(readout, state, skip), f'Hybrid mask {m}'))
    return {'native': audits, 'hybrid': mixtures}
