"""Bounded independent-training pilot on MFNet training-only images.

Imports the MIT-licensed pinned Sigma model and transforms in place. Does not
change upstream source, evaluate the test set, or implement a 500-epoch job.
"""
from __future__ import annotations

import argparse
import collections
import collections.abc
import contextlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from experiments.r1.assets import sha256
from experiments.r1c.data import canonical_sha, internal_split, read_lists, write_json
from experiments.r1c.state import (Sampler, accumulate_loss, complete_groups, load_checkpoint,
                                   normalize_gradients, save_checkpoint)


def emit(path, event):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, allow_nan=False) + '\n')
        stream.flush()
    print(json.dumps(event, allow_nan=False), flush=True)


def metrics(hist):
    diagonal = np.diag(hist)
    union = hist.sum(0) + hist.sum(1) - diagonal
    ious = [float(diagonal[i] / union[i]) if union[i] else None for i in range(9)]
    valid = [x for x in ious if x is not None]
    return {'iou_0_to_8': ious, 'miou_including_0': float(np.mean(valid)),
            'absent_union_classes': [i for i in range(9) if union[i] == 0],
            'confusion_matrix': hist.tolist(), 'pixel_count': int(hist.sum())}


def read_sample(root, name, config, training):
    import cv2
    from dataloader.dataloader import TrainPre
    from utils.transforms import normalize
    # Match pinned RGBXDataset's numeric imread flag (cv2.COLOR_BGR2RGB=4).
    # It decodes BGR, despite its misleading symbolic name. No silent correction.
    rgb = cv2.imread(str(root / 'RGB' / (name + '.png')), cv2.COLOR_BGR2RGB)
    thermal = cv2.imread(str(root / 'Modal' / (name + '.png')), cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(str(root / 'Label' / (name + '.png')), cv2.IMREAD_GRAYSCALE)
    if rgb is None or thermal is None or label is None:
        raise ValueError(f'Missing triplet {name}')
    thermal = cv2.merge([thermal, thermal, thermal])
    if training:
        rgb, label, thermal = TrainPre(config.norm_mean, config.norm_std, config)(rgb, label, thermal)
    else:
        rgb = normalize(rgb, config.norm_mean, config.norm_std).transpose(2, 0, 1)
        thermal = normalize(thermal, config.norm_mean, config.norm_std).transpose(2, 0, 1)
    return (torch.from_numpy(np.ascontiguousarray(rgb)).float().unsqueeze(0).cuda(),
            torch.from_numpy(np.ascontiguousarray(thermal)).float().unsqueeze(0).cuda(),
            torch.from_numpy(np.ascontiguousarray(label)).long().unsqueeze(0).cuda())


def validate(model, root, ids, config):
    model.eval()
    hist = np.zeros((9, 9), dtype=np.int64)
    start = time.perf_counter()
    loss_sum, pixels = 0., 0
    with torch.inference_mode():
        for name in ids:
            rgb, thermal, label = read_sample(root, name, config, False)
            logits = model(rgb, thermal)
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Nonfinite validation logits')
            loss_sum += torch.nn.functional.cross_entropy(logits, label, ignore_index=255, reduction='sum').item()
            pred, target = logits.argmax(1).cpu().numpy(), label.cpu().numpy()
            mask = (target >= 0) & (target < 9)
            pixels += int(mask.sum())
            hist += np.bincount(9 * target[mask] + pred[mask], minlength=81).reshape(9, 9)
    torch.cuda.synchronize()
    model.train()
    return dict(metrics(hist), scope='internal training holdout only; NOT final test',
                sample_count=len(ids), mean_pixel_loss=loss_sum / pixels,
                seconds=time.perf_counter() - start)


@contextlib.contextmanager
def construct_model(upstream, init):
    sys.path.insert(0, str(upstream))
    previous = Path.cwd()
    # Python 3.11 compatibility for the original transform helper, no algorithm change.
    if not hasattr(collections, 'Iterable'):
        collections.Iterable = collections.abc.Iterable
    with tempfile.TemporaryDirectory(prefix='sigma-r1c-') as directory:
        location = Path(directory) / 'pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth'
        location.parent.mkdir(parents=True)
        location.symlink_to(init)
        try:
            os.chdir(directory)
            from configs.config_MFNet import config
            from models.builder import EncoderDecoder
            model = EncoderDecoder(cfg=config, criterion=torch.nn.CrossEntropyLoss(ignore_index=255))
            # Upstream constructor swallows load errors. Repeat same load fail-closed,
            # and check the exact expected transfer-learning incompatibilities.
            state = torch.load(init, map_location='cpu', weights_only=False)['model']
            keys = model.backbone.vssm.load_state_dict(state, strict=False)
            expected_missing = {f'outnorm{i}.{part}' for i in range(4) for part in ('weight', 'bias')}
            expected_extra = {'classifier.norm.weight', 'classifier.norm.bias', 'classifier.head.weight', 'classifier.head.bias'}
            if set(keys.missing_keys) != expected_missing or set(keys.unexpected_keys) != expected_extra:
                raise ValueError(f'Unexpected VMamba transfer result: {keys}')
            del state
            yield model, config, {'missing_keys': keys.missing_keys, 'unexpected_keys': keys.unexpected_keys,
                                 'strict': False, 'scope': 'VMamba backbone initialization, not segmentation checkpoint'}
        finally:
            os.chdir(previous)


def run(args):
    upstream, init, root = args.upstream.resolve(), args.init.resolve(), args.root.resolve()
    output, checkpoints = args.output.resolve(), args.checkpoints.resolve()
    resume = args.resume.resolve() if args.resume else None
    bridge = Path(__file__).resolve().parents[2]
    if checkpoints.is_relative_to(bridge) or root.is_relative_to(bridge):
        raise ValueError('Data and checkpoint payloads must be outside the repository')
    cfg = json.loads(args.config.read_text())
    if (cfg['microbatch'] != 1 or not 1 <= cfg['accumulation'] <= 8 or not 1 <= cfg['max_updates'] <= 20
            or cfg['validation_interval'] <= 0
            or cfg['test_evaluation'] or cfg['precision'] != 'fp32' or cfg['device'] != 'cuda:0'
            or cfg['scheduler'] != 'constant-pilot-lr' or cfg['num_workers'] != 0):
        raise ValueError('This driver supports only bounded, single-GPU FP32 pilots')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU using CUDA_VISIBLE_DEVICES=0')
    if sha256(init) != cfg['init_sha256']:
        raise ValueError('Pinned VMamba hash mismatch')
    if subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != cfg['upstream_commit']:
        raise ValueError('Pinned source mismatch')
    subprocess.run(['git', '-C', str(upstream), 'diff', '--quiet', 'HEAD', '--'], check=True)
    split = json.loads(args.split.read_text())
    if split != internal_split(read_lists(root), cfg['seed']):
        raise ValueError('Training/validation split changed or overlaps test')
    manifest = json.loads(args.manifest.read_text())
    if manifest['errors'] or canonical_sha(manifest['files']) != manifest['files_sha256']:
        raise ValueError('Data audit not complete')
    expected = {f'{folder}/{sample}.png' for sample in split['train_ids'] + split['validation_ids']
                for folder in ('RGB', 'Modal', 'Label')}
    records = {r['path']: r for r in manifest['files']}
    if not expected <= records.keys():
        raise ValueError('Data manifest lacks pilot assets')
    for path in sorted(expected):
        if sha256(root / path) != records[path]['sha256']:
            raise ValueError(f'Asset changed after audit: {path}')
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    if resume is None and any((checkpoints / name).exists() for name in ('last.pt', 'best.pt')):
        raise FileExistsError('Start a new run in a fresh checkpoint directory or resume explicitly')
    event_path = output / ('resume-events.jsonl' if args.resume else 'pilot-events.jsonl')
    if event_path.exists():
        raise FileExistsError('Use a new evidence directory; never overwrite an earlier run')
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
    signature = {'config': cfg, 'split_sha256': split['split_sha256'], 'files_sha256': manifest['files_sha256'],
                 'driver_sha256': {f: sha256(Path(__file__).parent / f) for f in ('pilot.py', 'state.py', 'data.py')},
                 'torch': torch.__version__, 'cuda': torch.version.cuda,
                 'device': torch.cuda.get_device_name(0)}
    emit(event_path, {'event': 'start', 'signature': signature, 'resume': bool(args.resume),
                      'python': sys.version, 'numpy': np.__version__, 'test_pixels_read': 0,
                      'cudnn_deterministic': True, 'tf32': False,
                      'custom_cuda_kernel_bitwise_determinism': 'not guaranteed'})
    with construct_model(upstream, init) as (model, config, init_keys):
        from utils.init_func import group_weight
        model = model.cuda().train()
        legacy = group_weight([], model, torch.nn.BatchNorm2d, cfg['lr'])
        groups, coverage = complete_groups(model, legacy, cfg['lr'], cfg['weight_decay'])
        write_json(output / 'parameter-coverage.json', coverage)
        emit(event_path, {'event': 'initialization', **init_keys, **{k: v for k, v in coverage.items() if k != 'parameters'}})
        optimizer = torch.optim.AdamW(groups, lr=cfg['lr'], betas=tuple(cfg['betas']), weight_decay=cfg['weight_decay'])
        sampler = Sampler(len(split['train_ids']), cfg['seed'])
        update, best = 0, {'miou': -1., 'update': None}
        if resume:
            update, best, keys = load_checkpoint(resume, model, optimizer, sampler, signature)
            emit(event_path, {'event': 'restored', 'update': update, 'best': best, 'strict_model_keys': keys,
                              'checkpoint_sha256': sha256(resume), 'sampler_cursor': sampler.cursor,
                              'optimizer_state_tensor_count': len(optimizer.state)})
        stop = args.stop_after if args.stop_after is not None else cfg['max_updates']
        if not update < stop <= cfg['max_updates']:
            raise ValueError('Invalid stop boundary')
        named = dict(model.named_parameters())
        gradient_records = {n: {'numel': p.numel(), 'gradient_seen': False, 'nonzero_seen': False,
                                'changed_seen': False, 'max_grad_l2': 0.,
                                'max_nonzero_gradient_elements': 0, 'max_changed_elements': 0} for n, p in named.items()}
        optimizer.zero_grad(set_to_none=True)
        run_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        while update < stop:
            train_start = time.perf_counter()
            losses, names, durations, valid_counts = [], [], [], []
            for micro in range(cfg['accumulation']):
                name = split['train_ids'][sampler.next()]
                t0 = time.perf_counter()
                rgb, thermal, label = read_sample(root, name, config, True)
                loss = model(rgb, thermal, label)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                valid_pixels = int((label != 255).sum().item())
                accumulate_loss(loss, valid_pixels)
                valid_counts.append(valid_pixels)
                torch.cuda.synchronize()
                losses.append(loss.item())
                names.append(name)
                durations.append(time.perf_counter() - t0)
                del loss, rgb, thermal, label
            normalize_gradients(model, sum(valid_counts))
            grad_norm_squared = 0.
            missing, zero = [], []
            for n, p in named.items():
                if p.grad is None:
                    missing.append(n)
                    continue
                if not torch.isfinite(p.grad).all():
                    raise FloatingPointError(f'Nonfinite gradient {n}')
                norm = p.grad.norm().item()
                gradient_records[n]['gradient_seen'] = True
                gradient_records[n]['nonzero_seen'] |= norm > 0
                gradient_records[n]['max_grad_l2'] = max(gradient_records[n]['max_grad_l2'], norm)
                gradient_records[n]['max_nonzero_gradient_elements'] = max(
                    gradient_records[n]['max_nonzero_gradient_elements'], int(torch.count_nonzero(p.grad).item()))
                grad_norm_squared += norm ** 2
                if norm == 0:
                    zero.append(n)
            # One extra parameter-sized GPU copy (~184 MiB) to inspect real changes.
            before = {n: p.detach().clone() for n, p in named.items()}
            optimizer.step()
            unchanged = []
            for n, p in named.items():
                if not torch.isfinite(p).all():
                    raise FloatingPointError(f'Nonfinite parameter after AdamW update: {n}')
                changed_count = int(torch.count_nonzero(before[n] != p).item())
                changed = changed_count > 0
                gradient_records[n]['changed_seen'] |= changed
                gradient_records[n]['max_changed_elements'] = max(gradient_records[n]['max_changed_elements'], changed_count)
                if not changed:
                    unchanged.append(n)
            del before
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            update += 1
            event = {'event': 'update', 'update': update, 'microbatches': cfg['accumulation'],
                     'sample_ids': names, 'losses': losses, 'mean_loss': float(np.mean(losses)),
                     'valid_pixel_counts': valid_counts,
                     'optimized_pixel_mean_loss': float(np.average(losses, weights=valid_counts)),
                     'microstep_seconds': durations, 'update_seconds_including_audit': time.perf_counter() - train_start,
                     'gradient_l2': grad_norm_squared ** .5, 'missing_gradient_names': missing,
                     'zero_gradient_names': zero, 'unchanged_parameter_names': unchanged,
                     'lr': optimizer.param_groups[0]['lr'], 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                     'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
            emit(event_path, event)
            write_json(output / ('resume-gradient-audit.json' if args.resume else 'gradient-audit.json'), gradient_records)
            if missing:
                raise RuntimeError('Unexpected unused parameters; inspect evidence before changing model')
            improved = False
            if update % cfg['validation_interval'] == 0 or update == cfg['max_updates']:
                result = validate(model, root, split['validation_ids'], config)
                emit(event_path, {'event': 'validation', 'update': update, **result})
                if result['miou_including_0'] > best['miou']:
                    best = {'miou': result['miou_including_0'], 'update': update}
                    improved = True
            t0 = time.perf_counter()
            save_checkpoint(checkpoints / 'last.pt', model, optimizer, sampler, update, best, signature)
            if improved:
                save_checkpoint(checkpoints / 'best.pt', model, optimizer, sampler, update, best, signature)
            emit(event_path, {'event': 'checkpoint', 'update': update, 'best': best,
                              'last_sha256': sha256(checkpoints / 'last.pt'),
                              'last_bytes': (checkpoints / 'last.pt').stat().st_size,
                              'save_and_hash_seconds': time.perf_counter() - t0})
        emit(event_path, {'event': 'complete', 'update': update, 'seconds': time.perf_counter() - run_start,
                          'best': best, 'checkpoint_directory_payload_published': False,
                          'gradient_tensor_count': sum(v['gradient_seen'] for v in gradient_records.values()),
                          'updated_tensor_count': sum(v['changed_seen'] for v in gradient_records.values()),
                          'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                          'peak_reserved_bytes': torch.cuda.max_memory_reserved()})


def main():
    parser = argparse.ArgumentParser()
    for name in ('upstream', 'init', 'root', 'config', 'split', 'manifest', 'output', 'checkpoints'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--resume', type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
