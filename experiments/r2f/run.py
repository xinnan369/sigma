"""Reproduce and localize the first failing R2e call without causal replay."""
import argparse
from pathlib import Path
import math

import torch

from experiments.r1c.data import write_json
from experiments.r1c.pilot import construct_model, read_sample
from experiments.r1g.partition import ROOT, read, require
from experiments.r2a.run import model_digest
from experiments.r2b.state import Scan, coordinates
from experiments.r2b.triton_state import trajectory
from experiments.r2c.protocol import gpu_inputs
from experiments.r2d.models import fusion
from experiments.r2e.paths import intercept, geometry
from experiments.r2e.run import assets
from experiments.r2f.numerics import basis_states, native_scan, readout, sequential, summarize

TARGET = ('backbone.vssm.layers.2.blocks.2.op', 0, 0)


class Captured(Exception):
    """Stop after the selected invocation and let context managers clean up."""


def analyze(native, scan, y, prefix):
    oracle = basis_states(native, scan)
    old = trajectory(scan)
    old_readout = readout(scan, old)
    output = {'triton_vs_cuda_readout': summarize(old_readout, y),
              'triton_vs_cuda_state': summarize(old, oracle),
              'cuda_states_fused_readout_vs_cuda': summarize(readout(scan, oracle), y),
              'cuda_states_unfused_readout_vs_cuda': summarize(readout(scan, oracle, fused=False), y)}
    ends = [min(i + 2048, scan.u.shape[-1]) - 1 for i in range(0, scan.u.shape[-1], 2048)]
    output['cuda_basis_vs_cuda_chunk_end'] = summarize(oracle[..., ends].transpose(-1, -2), prefix[..., 1::2])
    output['triton_vs_cuda_chunk_end'] = summarize(old[..., ends].transpose(-1, -2), prefix[..., 1::2])
    for label, args in [('sequential', {}), ('sequential_exp2', {'exp2': True}),
                        ('sequential_exp2_fma', {'exp2': True, 'fused': True}),
                        ('sequential_exp2_fma_reordered_injection',
                         {'exp2': True, 'fused': True, 'alternate_injection': True})]:
        h = sequential(scan, **args)
        output[label] = {'readout': summarize(readout(scan, h), y), 'state': summarize(h, oracle),
                         'chunk_end': summarize(h[..., ends].transpose(-1, -2), prefix[..., 1::2])}
        del h

    dt = scan.step_sizes()
    require((dt > 0).all().item(), 'Positive step sizes required for zero-memory probe')
    single_b = torch.ones_like(scan.b[:, :, :1])
    isolated = native(torch.ones_like(scan.u), scan.delta, torch.full_like(scan.a[:, :1], -torch.inf),
                      single_b, single_b, None, scan.bias, scan.softplus, scan.nrows)[0]
    output['softplus_vs_cuda_probe'] = summarize(dt, isolated)
    h = sequential(scan, steps=isolated, exp2=True, fused=True)
    output['sequential_native_softplus_exp2_fma'] = {
        'readout': summarize(readout(scan, h), y), 'state': summarize(h, oracle)}
    del h
    for kind in ('worst_absolute', 'worst_mixed'):
        b, c, t = output['triton_vs_cuda_readout'][kind]['index']
        group = c // (scan.u.shape[1] // scan.c.shape[1])
        state_error = (old[b, c, :, t] - oracle[b, c, :, t]).abs()
        terms = oracle[b, c, :, t] * scan.c[b, group, :, t]
        n = int((state_error * scan.c[b, group, :, t].abs()).argmax().item())
        a = scan.a[c, :, None].contiguous()
        ones = torch.ones((1, a.shape[0], 2), device=a.device)
        u = ones.clone()
        u[..., 1] = 0
        step = ones.clone()
        step[..., 1] = isolated[b, c, t]
        coeff = torch.ones((1, 1, 1, 2), device=a.device)
        alpha = native(u, step, a, coeff, coeff, None, None, False, 1)[0][0, :, 1]
        basis = torch.zeros_like(scan.c)
        basis[:, :, n] = 1
        injection = native(scan.u, scan.delta, torch.full_like(scan.a, -torch.inf), scan.b, basis,
                           None, scan.bias, scan.softplus, scan.nrows)[0][b, c, t]
        dx = scan.d[c] * scan.u[b, c, t]
        native_skip = native_scan(native, scan, coeff=torch.zeros_like(scan.c))[0][b, c, t]
        output[kind] = {
            'batch': b, 'channel': c, 'direction': group, 'time': t,
            'coordinate': coordinates('four', group, t, 30, 40), 'largest_weighted_state_error_index': n,
            'cuda': float(y[b, c, t]), 'triton': float(old_readout[b, c, t]),
            'cuda_state': float(oracle[b, c, n, t]), 'triton_state': float(old[b, c, n, t]),
            'cuda_delta': float(isolated[b, c, t]), 'torch_delta': float(dt[b, c, t]),
            'cuda_alpha': float(alpha[n]), 'torch_exp_alpha': float(torch.exp(dt[b, c, t] * scan.a[c, n])),
            'torch_exp2_alpha': float(torch.exp2(dt[b, c, t] * (scan.a[c, n] * math.log2(math.e)))),
            'cuda_injection': float(injection),
            'delta_x_then_b': float((isolated[b, c, t] * scan.u[b, c, t]) * scan.b[b, group, n, t]),
            'delta_b_then_x': float((isolated[b, c, t] * scan.b[b, group, n, t]) * scan.u[b, c, t]),
            'dx': float(dx), 'cuda_dx': float(native_skip), 'sum_abs_readout_terms': float(terms.abs().sum() + dx.abs())}
    return output


def run(args):
    _, _, _, _, runtime = gpu_inputs(args, validation_only=True)
    all_assets = assets(args.checkpoint_dir, [3407, 3408, 3409, 3410, 3411])
    checkpoint, ledger, signature = all_assets[3407, 'ssm']
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    require(state['signature'] == signature and state['updates'] == 785 and state['sampler']['epoch'] == 5,
            'Fixed checkpoint metadata differs')
    widths = read(ROOT / 'configs/R2d_protocol.json')['widths']
    with construct_model(args.upstream, args.init) as (model, cfg, _), fusion(model, 'ssm', widths), torch.inference_mode():
        from models.encoders import vmamba

        model.load_state_dict(state['model'], strict=True)
        model.cuda().eval()
        require(model_digest(model) == ledger['model_sha256'], 'Loaded model differs')
        rgb, thermal, _ = read_sample(args.root, '00057D', cfg, False)

        def observe(key, scope, grid, native, args_):
            y, prefix = native(*args_)
            if key == TARGET:
                scan = Scan(*args_)
                shape = geometry(scope, key[2], grid, scan)
                require(shape['grid'] == [30, 40] and shape['state_dim'] == 16 and shape['length'] == 1200,
                        'Fixed failing layout differs')
                result = analyze(native, scan, y, prefix)
                write_json(args.evidence / 'fixed-call.json', {
                    'sample': '00057D', 'seed': 3407, 'arm': 'ssm', 'call': list(TARGET), 'layout': shape,
                    'runtime': runtime, 'checkpoint_sha256': ledger['last_sha256'], **result})
                print({k: v['max_mixed_ratio'] for k, v in result.items() if 'max_mixed_ratio' in v}, flush=True)
                raise Captured
            return y, prefix

        try:
            with intercept(model, vmamba, 'ssm', observe):
                model(rgb, thermal)
        except Captured:
            return
        raise ValueError('Fixed target was not captured')


def main():
    parser = argparse.ArgumentParser()
    for name in ('upstream', 'init', 'root', 'checkpoint-dir', 'evidence'):
        parser.add_argument('--' + name, required=True, type=lambda p: Path(p).resolve())
    run(parser.parse_args())


if __name__ == '__main__':
    main()
