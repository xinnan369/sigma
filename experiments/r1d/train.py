"""Train one independently initialized Sigma-Tiny epoch using R1c assets."""
import argparse
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

from experiments.r1.assets import sha256
from experiments.r1c.data import canonical_sha, internal_split, read_lists, write_json
from experiments.r1c.pilot import construct_model, emit, metrics, read_sample, validate
from experiments.r1c.state import accumulate_loss, complete_groups, normalize_gradients
from experiments.r1d.state import EpochSampler, load, save

BRIDGE = Path(__file__).resolve().parents[2]
CLASSES = ('unlabeled', 'car', 'person', 'bike', 'curve', 'car stop', 'guardrail', 'color cone', 'bump')


def class_metrics(hist):
    hist = np.asarray(hist, dtype=np.int64)
    if hist.shape != (9, 9) or np.any(hist < 0) or not hist.sum():
        raise ValueError('Expected a nonempty nine-class confusion matrix')
    out = metrics(hist)
    pixels = int(hist.sum())
    rows = []
    for i, name in enumerate(CLASSES):
        tp, actual, predicted = map(int, (hist[i, i], hist[i, :].sum(), hist[:, i].sum()))
        rows.append({'class_id': i, 'class_name': name, 'actual_pixels': actual,
                     'predicted_pixels': predicted, 'actual_fraction': actual / pixels,
                     'predicted_fraction': predicted / pixels, 'precision': tp / predicted if predicted else None,
                     'recall': tp / actual if actual else None, 'iou': out['iou_0_to_8'][i]})
    foreground = [row['iou'] for row in rows[1:] if row['iou'] is not None]
    return dict(out, classes=rows, foreground_miou=float(np.mean(foreground)) if foreground else None)


def audited_inputs(root, cfg):
    evidence = BRIDGE / 'reports/evidence/r1c'
    split = json.loads((evidence / 'split.json').read_text())
    if split != internal_split(read_lists(root), cfg['seed']) or split['split_sha256'] != cfg['split_sha256']:
        raise ValueError('R1c split changed')
    if (len(split['train_ids']), len(split['validation_ids'])) != (1410, 79):
        raise ValueError('Unexpected R1c split sizes')
    manifest = json.loads((evidence / 'manifest.json').read_text())
    if manifest['errors'] or canonical_sha(manifest['files']) != cfg['files_sha256']:
        raise ValueError('R1c file manifest changed')
    digests = {f['path']: f['sha256'] for f in manifest['files']}
    for sample in split['train_ids'] + split['validation_ids']:
        for folder in ('RGB', 'Modal', 'Label'):
            path = f'{folder}/{sample}.png'
            if sha256(root / path) != digests[path]:
                raise ValueError(f'Asset hash changed: {path}')
    return split


def verify_coverage(model, groups, coverage, reference):
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [id(p) for group in groups for p in group['params']]
    if len(actual) != 668 or set(actual) != expected or coverage != reference:
        raise ValueError('Optimizer coverage differs from the accepted R1c model')


def train_window(model, optimizer, scheduler, names, root, config):
    start = time.perf_counter()
    pixel_counts = np.zeros(9, dtype=np.int64)
    losses, counts, seconds = [], [], []
    for name in names:
        t0 = time.perf_counter()
        rgb, thermal, label = read_sample(root, name, config, True)
        values = label.cpu().numpy().ravel()
        labeled = values[values != 255]
        pixel_counts += np.bincount(labeled, minlength=9)
        loss = model(rgb, thermal, label)
        value = loss.item()
        if not math.isfinite(value):
            raise FloatingPointError(f'Nonfinite training CE for {name}')
        accumulate_loss(loss, labeled.size)
        torch.cuda.synchronize()
        losses.append(value)
        counts.append(int(labeled.size))
        seconds.append(time.perf_counter() - t0)
        del loss, rgb, thermal, label
    normalize_gradients(model, sum(counts))
    named = list(model.named_parameters())
    missing = [name for name, p in named if p.grad is None]
    if missing:
        raise RuntimeError(f'Missing gradients: {missing}')
    norms = torch.stack([p.grad.norm() for _, p in named])
    norm = norms.norm().item()
    if not math.isfinite(norm):
        raise FloatingPointError('Nonfinite gradient norm')
    zero = [name for (name, _), n in zip(named, norms.tolist()) if n == 0]
    lr = optimizer.param_groups[0]['lr']
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return {'sample_ids': names, 'microbatches': len(names), 'losses': losses, 'valid_pixel_counts': counts,
            'pixel_loss_sum': sum(loss * count for loss, count in zip(losses, counts)),
            'mean_pixel_loss': float(np.average(losses, weights=counts)), 'class_pixels': pixel_counts.tolist(),
            'microstep_seconds': seconds, 'window_seconds': time.perf_counter() - start,
            'gradient_l2': norm, 'gradient_tensor_count': len(named), 'zero_gradient_names': zero, 'lr': lr,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def run(args):
    import cv2
    import selective_scan_cuda_core

    cfg = json.loads((BRIDGE / 'configs/R1d_epoch.json').read_text())
    protocol = (cfg['epochs'], cfg['seed'], cfg['precision'], cfg['microbatch'], cfg['accumulation'],
                cfg['lr'], cfg['weight_decay'], cfg['betas'], cfg['scheduler'], cfg['test_evaluation'])
    if protocol != (1, 3407, 'fp32', 1, 8, 6e-5, .01, [.9, .999], 'constant', False):
        raise ValueError('R1d requires the approved single-epoch protocol')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU with CUDA_VISIBLE_DEVICES')
    if sha256(args.init) != cfg['init_sha256']:
        raise ValueError('Pinned VMamba initializer hash mismatch')
    git = ['git', '-C', str(args.upstream)]
    if subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip() != cfg['upstream_commit']:
        raise ValueError('Pinned upstream commit mismatch')
    subprocess.run(git + ['diff', '--quiet', 'HEAD', '--'], check=True)
    split = audited_inputs(args.root, cfg)
    ids = split['train_ids']
    total = math.ceil(len(ids) / cfg['accumulation'])
    stop = total if args.stop_after is None else args.stop_after
    if not 1 <= stop <= total:
        raise ValueError('Invalid stop update')
    checkpoint = args.checkpoint_dir / 'last.pt'
    if args.resume != checkpoint.is_file():
        raise ValueError('Use an empty checkpoint directory for a fresh experiment, or specify --resume')
    args.output.mkdir(parents=True, exist_ok=True)
    event_path = args.output / 'events.jsonl'
    event_path.touch(exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_files = ['experiments/r1d/train.py', 'experiments/r1d/state.py',
                    'experiments/r1c/pilot.py', 'experiments/r1c/state.py', 'experiments/r1c/data.py']
    signature = {'config': cfg, 'source_sha256': {f: sha256(BRIDGE / f) for f in source_files},
                 'python': sys.version, 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                 'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(0),
                 'cv2': cv2.__version__, 'numpy': np.__version__,
                 'selective_scan_cuda_core_sha256': sha256(Path(selective_scan_cuda_core.__file__))}
    emit(event_path, {'event': 'start', 'signature': signature, 'resume': args.resume,
                      'training_entries': len(ids), 'validation_images': len(split['validation_ids']),
                      'test_pixels_read': 0, 'tf32': False, 'cudnn_deterministic': True})
    with construct_model(args.upstream, args.init) as (model, config, init_keys):
        from utils.init_func import group_weight
        model = model.cuda().train()
        legacy = group_weight([], model, torch.nn.BatchNorm2d, cfg['lr'])
        groups, coverage = complete_groups(model, legacy, cfg['lr'], cfg['weight_decay'])
        reference = json.loads((BRIDGE / 'reports/evidence/r1c/pilot/parameter-coverage.json').read_text())
        verify_coverage(model, groups, coverage, reference)
        write_json(args.output / 'parameter-coverage.json', coverage)
        emit(event_path, {'event': 'init', **init_keys, 'covered_tensors': len(list(model.parameters())),
                          'covered_numel': coverage['total_numel']})
        optimizer = torch.optim.AdamW(groups, lr=cfg['lr'], betas=tuple(cfg['betas']), weight_decay=cfg['weight_decay'])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        sampler = EpochSampler(len(ids), cfg['seed'])
        progress = {'updates': 0, 'class_pixels': [0] * 9, 'loss_sum': 0., 'train_seconds': 0.}
        if args.resume:
            progress, keys = load(checkpoint, model, optimizer, scheduler, sampler, signature)
            emit(event_path, {'event': 'restore', 'epoch': sampler.epoch, 'cursor': sampler.cursor,
                              'updates': progress['updates'], 'scheduler_last_epoch': scheduler.last_epoch,
                              'strict_model_keys': keys, 'optimizer_state_tensors': len(optimizer.state),
                              'sha256': sha256(checkpoint)})
        if sampler.epoch != 1 or sampler.cursor != min(len(ids), progress['updates'] * cfg['accumulation']):
            raise ValueError('Checkpoint epoch cursor inconsistent with update count')
        if progress['updates'] > stop:
            raise ValueError('Stop boundary precedes saved checkpoint')
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        while progress['updates'] < stop:
            names = [ids[i] for i in sampler.next_window(cfg['accumulation'])]
            result = train_window(model, optimizer, scheduler, names, args.root, config)
            progress['updates'] += 1
            progress['class_pixels'] = [a + b for a, b in zip(progress['class_pixels'], result['class_pixels'])]
            progress['loss_sum'] += result['pixel_loss_sum']
            progress['train_seconds'] += result['window_seconds']
            step = progress['updates']
            emit(event_path, {'event': 'update', 'epoch': sampler.epoch, 'cursor': sampler.cursor,
                              'update': step, **result})
            if step % cfg['checkpoint_every_updates'] == 0 or step == stop:
                t0 = time.perf_counter()
                save(checkpoint, model, optimizer, scheduler, sampler, progress, signature)
                emit(event_path, {'event': 'save', 'update': step, 'cursor': sampler.cursor,
                                  'sha256': sha256(checkpoint), 'bytes': checkpoint.stat().st_size,
                                  'seconds': time.perf_counter() - t0})
        if sampler.cursor == len(ids):
            result = validate(model, args.root, split['validation_ids'], config)
            r1c_events = BRIDGE / 'reports/evidence/r1c/pilot/resume-events.jsonl'
            reference = next(e for e in map(json.loads, r1c_events.read_text().splitlines())
                             if e['event'] == 'validation' and e['update'] == 13)
            comparison = {'r1c_update_13': dict(reference, **class_metrics(reference['confusion_matrix'])),
                          'r1d_epoch_1': dict(result, **class_metrics(result['confusion_matrix']))}
            write_json(args.output / 'validation.json', comparison)
            emit(event_path, {'event': 'validation', **comparison['r1d_epoch_1']})
        emit(event_path, {'event': 'complete' if sampler.cursor == len(ids) else 'stopped',
                          'epoch': sampler.epoch, 'cursor': sampler.cursor, **progress,
                          'mean_pixel_loss': progress['loss_sum'] / sum(progress['class_pixels']),
                          'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                          'peak_reserved_bytes': torch.cuda.max_memory_reserved()})


def main():
    import fcntl

    parser = argparse.ArgumentParser()
    for name in ('upstream', 'init', 'root', 'output', 'checkpoint-dir'):
        parser.add_argument('--' + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.checkpoint_dir.is_relative_to(BRIDGE) or args.root.is_relative_to(BRIDGE):
        parser.error('Place datasets and checkpoint payloads outside the bridge')
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (args.checkpoint_dir / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == '__main__':
    main()
