"""Complete fixed Sigma-Tiny synthetic forward, explicitly without a segmentation checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


def shape_tree(value):
    if isinstance(value, (tuple, list)):
        return [shape_tree(x) for x in value]
    return list(value.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('runs must be positive')
    upstream = args.upstream.resolve()
    result = {'scope': 'Untrained synthetic execution smoke; NOT official checkpoint inference or mIoU',
              'seed': 3407, 'segmentation_checkpoint': 'NOT LOADED', 'exception': None}
    hooks = []
    try:
        head = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
        if head != '98ba3c6938d29169bbd27f91519084bb462561df':
            raise ValueError('Pinned upstream mismatch')
        subprocess.run(['git', '-C', str(upstream), 'diff', '--quiet', 'HEAD', '--'], check=True)
        os.chdir(upstream)
        sys.path.insert(0, str(upstream))
        import torch
        from configs.config_MFNet import config
        from models.builder import EncoderDecoder
        torch.manual_seed(3407)
        torch.cuda.manual_seed_all(3407)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assert config.backbone == 'sigma_tiny' and config.decoder == 'MambaDecoder'
        model = EncoderDecoder(cfg=config, criterion=None).cuda().eval()
        observed = {}

        def observe(name):
            def hook(module, inputs, output):
                record = observed.setdefault(name, {'calls': 0})
                record.update(calls=record['calls'] + 1,
                              input_shapes=shape_tree(inputs), output_shapes=shape_tree(output))
            return hook

        modules = dict(model.named_modules())
        targets = ([f'backbone.cross_mamba.{i}' for i in range(4)]
                   + [f'backbone.channel_attn_mamba.{i}' for i in range(4)]
                   + [f'decode_head.layers_up.{i}' for i in range(4)] + ['decode_head.output'])
        for name in targets:
            hooks.append(modules[name].register_forward_hook(observe(name)))
        rgb = torch.randn(1, 3, 480, 640, device='cuda')
        thermal = torch.randn(1, 1, 480, 640, device='cuda').repeat(1, 3, 1, 1)
        with torch.inference_mode():
            warm_start = time.perf_counter()
            output = model(rgb, thermal)
            torch.cuda.synchronize()
            warm_seconds = time.perf_counter() - warm_start
            torch.cuda.reset_peak_memory_stats()
            baseline_bytes = torch.cuda.memory_allocated()
            timings = []
            wall_start = time.perf_counter()
            for _ in range(args.runs):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                output = model(rgb, thermal)
                end.record()
                torch.cuda.synchronize()
                timings.append(start.elapsed_time(end))
            wall_seconds = time.perf_counter() - wall_start
            assert output.shape == (1, 9, 480, 640) and torch.isfinite(output).all().item()
        assert all(observed[name]['calls'] == args.runs + 1 for name in targets)
        result.update(upstream_head=head, torch=torch.__version__, cuda_build=torch.version.cuda,
                      device=torch.cuda.get_device_name(0), dtype=str(rgb.dtype),
                      input_rgb=list(rgb.shape), input_thermal=list(thermal.shape), output=list(output.shape),
                      parameters=sum(p.numel() for p in model.parameters()),
                      decoder_class=type(model.decode_head).__name__,
                      warmup_runs=1, measured_runs=args.runs, warmup_wall_seconds=warm_seconds,
                      cuda_event_milliseconds=timings, median_milliseconds=statistics.median(timings),
                      measured_loop_wall_seconds=wall_seconds,
                      baseline_allocated_bytes=baseline_bytes,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      modules=observed, output_finite=True, status='PASS')
    except Exception as exc:
        result.update(status='FAIL', exception=type(exc).__name__ + ': ' + str(exc))
        traceback.print_exc()
    finally:
        for hook in hooks:
            hook.remove()
    print(json.dumps(result, indent=2))
    return int(result['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
