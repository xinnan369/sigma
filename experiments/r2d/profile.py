"""Static width audit and single-GPU synthetic resource measurements."""
import argparse
from pathlib import Path
import statistics
import time

import torch

from experiments.r1.assets import sha256
from experiments.r1c.data import canonical_sha, write_json
from experiments.r1c.pilot import construct_model
from experiments.r1c.state import complete_groups
from experiments.r1g.partition import ROOT, read, require
from experiments.r2b.controls import set_seed
from experiments.r2c.cost import profile
from experiments.r2c.protocol import ARMS, digest_common, gpu_inputs
from experiments.r2c.train import cost
from experiments.r2d.models import fusion, search


def design():
    cfg = read(ROOT / 'configs/R2d_protocol.json')
    require(cfg['seeds'] == [3407, 3408, 3409, 3410, 3411] and cfg['primary_epoch'] == 5
            and cfg['bootstrap_replicates'] == 10000 and cfg['training_base'] == 'R2c_protocol.json',
            'Frozen R2d protocol differs')
    choices = {arm: search(arm, cfg) for arm in ('cnn', 'attention')}
    require(all(x['widths'] == cfg['widths'][arm] for arm, x in choices.items()), 'Frozen widths differ')
    return cfg, choices


def measure(args):
    cfg, choices = design()
    protocol, _, _, _, runtime = gpu_inputs(args, validation_only=True)
    results = {}
    for arm in ARMS:
        set_seed(3407)
        with construct_model(args.upstream, args.init) as (model, _, _), fusion(model, arm, cfg['widths']):
            import selective_scan_cuda_core
            from utils.init_func import group_weight

            model = model.cuda().eval()
            parts = cost(model, arm)
            n = sum(p.numel() for p in model.parameters())
            require(abs(n - cfg['reference_parameters']) / cfg['reference_parameters']
                    <= cfg['max_parameter_fraction'], 'Total parameter match exceeds 1%')
            if arm == 'ssm':
                require(n == cfg['reference_parameters'], 'SSM reference size differs')
            else:
                require(parts['fusion_parameters'] == choices[arm]['fusion_parameters'], 'Static count differs')
            shape_cost = profile(model, selective_scan_cuda_core)
            require(shape_cost['scan_scalar_state_updates'].get('fusion', 0)
                    == parts['fusion_ssm_scalar_state_updates'], 'Scan cost differs')
            image = torch.randn((1, 3, 480, 640), device='cuda')
            thermal = torch.randn_like(image)
            torch.cuda.reset_peak_memory_stats()
            timings = []
            with torch.inference_mode():
                for i in range(25):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    output = model(image, thermal)
                    end.record()
                    end.synchronize()
                    if i >= 5:
                        timings.append(start.elapsed_time(end))
            require(output.shape == (1, 9, 480, 640) and torch.isfinite(output).all().item(), 'Invalid logits')
            peak = torch.cuda.max_memory_allocated()
            initial = digest_common(model)
            model.train()
            label = torch.randint(0, 9, (1, 480, 640), device='cuda')
            lr = protocol['lr']
            groups, coverage = complete_groups(model, group_weight([], model, torch.nn.BatchNorm2d, lr),
                                               lr, protocol['weight_decay'])
            opt = torch.optim.AdamW(groups, lr=lr, betas=tuple(protocol['betas']))
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(8):
                loss = model(image, thermal, label) / 8
                loss.backward()
            require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()),
                    'Synthetic backward lacks finite gradients')
            opt.step()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            require(len(opt.state) == coverage['tensor_count'], 'Synthetic optimizer coverage differs')
            results[arm] = {'parameters': n, 'covered_tensors': coverage['tensor_count'], 'common_init': initial,
                            **parts, **shape_cost, 'total_macs': shape_cost['common_macs'] +
                            parts['fusion_linear_conv_attention_macs'], 'inference_ms': timings,
                            'inference_median_ms': statistics.median(timings), 'inference_peak_bytes': peak,
                            'dummy_update_seconds': seconds, 'training_peak_bytes': torch.cuda.max_memory_allocated()}
            print(arm, n, statistics.median(timings), seconds, flush=True)
            del opt, groups, model, image, thermal, output, loss
        torch.cuda.empty_cache()
    require(len({r['common_init'] for r in results.values()}) == 1, 'Dummy common initialization differs')
    require(len({r['common_macs'] for r in results.values()}) == 1, 'Shared compute differs')
    estimate = sum(r['dummy_update_seconds'] for r in results.values()) * 785 / 3600
    write_json(args.evidence / 'resource-profile.json', {'status': 'PASS', 'config_sha256': canonical_sha(cfg),
               'model_source_sha256': sha256(ROOT / 'experiments/r2d/models.py'), 'runtime': runtime,
               'design': choices, 'arms': results, 'estimated_training_hours_per_seed_from_dummy': estimate,
               'estimated_training_hours_five_seeds_from_dummy': estimate * 5,
               'scope': 'synthetic FP32; estimate excludes dataset, validation, evaluation and I/O'})


def main():
    parser = argparse.ArgumentParser()
    for name in ('upstream', 'init', 'root', 'evidence'):
        parser.add_argument('--' + name, type=lambda p: Path(p).resolve(), required=True)
    measure(parser.parse_args())


if __name__ == '__main__':
    main()
