"""Resource feasibility only: official VMamba init + synthetic optimizer workload, no dataset training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

from experiments.r1.assets import sha256


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--init', type=Path, required=True)
    parser.add_argument('--batches', type=int, nargs='+', default=[1],
                        help='Default microbatch=1; larger batches can exhaust shared host memory')
    args = parser.parse_args()
    upstream, init = args.upstream.resolve(), args.init.resolve()
    if sha256(init) != 'dbc0cc4f5ec0e45db5fba7c939d2c7d9b617e891ac10766912a8d604c37c5e47':
        raise ValueError('VMamba initialization hash differs from pinned LFS object')
    head = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if head != '98ba3c6938d29169bbd27f91519084bb462561df':
        raise ValueError('Pinned upstream mismatch')
    subprocess.run(['git', '-C', str(upstream), 'diff', '--quiet', 'HEAD', '--'], check=True)
    sys.path.insert(0, str(upstream))
    torch.set_num_threads(4)
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    result = {'scope': 'Synthetic training resource probe only; NOT independent MFNet training or author checkpoint reproduction',
              'official_segmentation_checkpoint': 'NOT LOADED', 'seed': 3407, 'steps': []}
    with tempfile.TemporaryDirectory(prefix='sigma-r1b-probe-') as directory:
        # Satisfy the unmodified constructor's relative initialization path.
        location = Path(directory) / 'pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth'
        location.parent.mkdir(parents=True)
        location.symlink_to(init)
        os.chdir(directory)
        from configs.config_MFNet import config
        from models.builder import EncoderDecoder
        from utils.init_func import group_weight
        model = EncoderDecoder(cfg=config, criterion=torch.nn.CrossEntropyLoss(ignore_index=255)).cuda().train()
        groups = group_weight([], model, torch.nn.BatchNorm2d, config.lr)
        covered = {id(p) for group in groups for p in group['params']}
        omitted = [(name, p.numel()) for name, p in model.named_parameters() if id(p) not in covered]
        result['upstream_optimizer_omitted_parameters'] = omitted
        result['omitted_numel'] = sum(n for _, n in omitted)
        result['total_parameters'] = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(groups, lr=config.lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
        for batch in args.batches:
            if batch < 1:
                raise ValueError('Batch size must be positive')
            record = {'batch': batch, 'rgb_shape': [batch, 3, 480, 640],
                      'warmup_steps': 0, 'measured_steps': 1, 'mixed_precision': False}
            try:
                optimizer.zero_grad(set_to_none=True)
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                rgb = torch.randn(batch, 3, 480, 640, device='cuda')
                thermal = torch.randn(batch, 1, 480, 640, device='cuda').repeat(1, 3, 1, 1)
                label = torch.randint(0, 9, (batch, 480, 640), device='cuda')
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                loss = model(rgb, thermal, label)
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                record.update(status='PASS', finite_loss=bool(torch.isfinite(loss).item()),
                              elapsed_seconds=time.perf_counter() - start,
                              peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved())
                assert record['finite_loss']
                del loss, rgb, thermal, label
            except torch.cuda.OutOfMemoryError:
                record.update(status='OOM', peak_allocated_bytes=torch.cuda.max_memory_allocated())
                traceback.print_exc()
            result['steps'].append(record)
            print(json.dumps({'completed_synthetic_probe_step': record}), flush=True)
    print(json.dumps(result, indent=2))
    return 0 if result['steps'][0]['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
