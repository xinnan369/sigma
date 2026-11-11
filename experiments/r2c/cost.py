"""Count common forward MACs and SSM state work using actual execution shapes."""
import argparse
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.r1c.data import write_json
from experiments.r1c.pilot import construct_model
from experiments.r1g.partition import read, require
from experiments.r2a.hooks import modules
from experiments.r2b.controls import fusion, set_seed
from experiments.r2c.protocol import ARMS, gpu_inputs

LABEL = 'R2c counted separately: fusion'
MAC_OPS = frozenset(('aten::conv2d', 'aten::mm', 'aten::bmm', 'aten::addmm'))


def outside_fusion(event):
    while event is not None:
        if event.name == LABEL:
            return False
        event = event.cpu_parent
    return True


def profile(model, extension):
    scope = []
    original = extension.fwd
    scan_cells, scan_calls = Counter(), Counter()

    def before(*_):
        region = torch.profiler.record_function(LABEL)
        region.__enter__()
        scope.append(region)

    def after(*_):
        scope.pop().__exit__(None, None, None)

    def scan(*args):
        u, a = args[0], args[2]
        k = 'fusion' if scope else 'common'
        scan_cells[k] += u.numel() * a.shape[1]
        scan_calls[k] += 1
        return original(*args)

    image = torch.zeros((1, 3, 480, 640), device='cuda')
    with ExitStack() as stack:
        for module in modules(model).values():
            stack.callback(module.register_forward_pre_hook(before).remove)
            stack.callback(module.register_forward_hook(after, always_call=True).remove)
        stack.enter_context(patch.object(extension, 'fwd', scan))
        with torch.inference_mode(), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True, with_flops=True
        ) as p:
            result = model(image, image)
            torch.cuda.synchronize()
    require(not scope and result.shape == (1, 9, 480, 640) and torch.isfinite(result).all().item(),
            'Unexpected profiled forward')
    common = Counter()
    for event in p.events():
        if event.flops and outside_fusion(event):
            common[event.name] += event.flops
    dot_flops = sum(v for k, v in common.items() if k in MAC_OPS)
    require(dot_flops > 0 and scan_cells['common'] > 0 and dot_flops % 2 == 0, 'Missing shared cost')
    return {'common_ops_flops': dict(common), 'common_macs': dot_flops // 2,
            'other_profiled_common_scalar_flops': sum(common.values()) - dot_flops,
            'scan_scalar_state_updates': dict(scan_cells), 'scan_calls': dict(scan_calls)}


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'upstream', 'init', 'evidence'):
        parser.add_argument('--' + name, type=lambda p: Path(p).resolve(), required=True)
    args = parser.parse_args()
    cfg, _, _, _, signature = gpu_inputs(args, validation_only=True)
    rows = {}
    for arm in ARMS:
        training = read(args.evidence / 'training' / arm / 'initialization.json')
        require(training['signature'] == signature, 'Measured arm architecture differs')
        set_seed(cfg['seed'])
        with construct_model(args.upstream, args.init) as (model, _, _), fusion(model, arm):
            import selective_scan_cuda_core as extension

            r = profile(model.cuda().eval(), extension)
            require(r['scan_scalar_state_updates'].get('fusion', 0) == training['fusion_ssm_scalar_state_updates'],
                    'Fusion scan count differs from analytical cost')
            rows[arm] = {**r, 'fusion_macs': training['fusion_linear_conv_attention_macs'],
                         'combined_macs': r['common_macs'] + training['fusion_linear_conv_attention_macs'],
                         'combined_scan_scalar_state_updates': sum(r['scan_scalar_state_updates'].values())}
    require(len({r['common_macs'] for r in rows.values()}) == 1
            and len({r['scan_scalar_state_updates']['common'] for r in rows.values()}) == 1,
            'Common traced compute differs across arms')
    write_json(args.evidence / 'compute-profile.json',
               {'scope': 'synthetic full-resolution eval forward; counted ops and state updates only',
                'input_shape_each': [1, 3, 480, 640], 'arms': rows})
    print({k: (v['combined_macs'], v['combined_scan_scalar_state_updates']) for k, v in rows.items()})


if __name__ == '__main__':
    main()
