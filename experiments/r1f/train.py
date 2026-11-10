"""Five-epoch fresh-start training gated by the frozen visual split audit."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from experiments.r1.assets import sha256
from experiments.r1c.data import canonical_sha, read_lists, write_json
from experiments.r1c.pilot import construct_model, emit, validate
from experiments.r1c.state import capture_rng, complete_groups
from experiments.r1d.state import EpochSampler
from experiments.r1d.train import BRIDGE, class_metrics, train_window, verify_coverage
from experiments.r1e.state import N, STEPS, WINDOW, require, restore
from experiments.r1e.train import check_best, read, signatures
from experiments.r1f.scene import RULES
from experiments.r1f.split import components, verify

EVIDENCE = BRIDGE / 'reports/evidence/r1f'
VERSION = 'r1f-visual-split-v1'


def frozen_split(cfg, lists):
    scene, split, audit = (read(EVIDENCE / name) for name in ('scene-audit.json', 'split.json', 'isolation.json'))
    require(scene['rules'] == RULES and not scene['flip_mismatches'] and scene['test_pixels_read'] == 0
            and scene['originals'] == 784 and scene['all_pairs'] == 784 * 783 // 2,
            'Scene audit is incomplete')
    require(split['graph_sha256'] == canonical_sha(scene)
            and split['split_sha256'] == cfg['split_sha256'] == audit['split_sha256']
            and audit['status'] == 'PASS', 'Frozen audit or split digest mismatch')
    pairs = [(e['a'], e['b']) for e in scene['edges']]
    require(components(scene['label_pixels'], pairs) == scene['groups'], 'Scene component mismatch')
    support = verify(split, scene['groups'], pairs, lists['train.txt'], lists['test.txt'], scene['label_pixels'])
    require(support == audit['class_support'], 'Class support differs from audited split')
    return split


def save(last, model, optimizer, scheduler, sampler, progress, history, signature):
    require(all(p.grad is None for p in model.parameters()), 'Checkpoint requires a complete update')
    temp = last.with_suffix('.pt.tmp')
    with temp.open('wb') as stream:
        torch.save({'format': VERSION, 'signature': signature, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'sampler': sampler.state_dict(), 'progress': progress, 'history': history,
                    'rng': capture_rng()}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, last)


def run(args):
    cfg = read(BRIDGE / 'configs/R1f_scene.json')
    require(cfg['version'] == VERSION and cfg['epochs'] == 5, 'Only the frozen five-epoch protocol is supported')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expose one CUDA GPU')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime, _ = signatures(read(BRIDGE / 'configs/R1e_convergence.json'))
    protocol = runtime['config']
    split = frozen_split(cfg, read_lists(args.root))
    manifest = read(BRIDGE / 'reports/evidence/r1c/manifest.json')
    require(split['r1c_files_sha256'] == canonical_sha(manifest['files']) == protocol['files_sha256'],
            'Pinned MFNet asset manifest differs')
    files = {f['path']: f['sha256'] for f in manifest['files']}
    for name in split['train_ids'] + split['validation_ids']:
        for folder in ('RGB', 'Modal', 'Label'):
            path = f'{folder}/{name}.png'
            require(sha256(args.root / path) == files[path], f'Changed asset {path}')
    require(sha256(args.init) == protocol['init_sha256'], 'Pinned VMamba initialization mismatch')
    git = ['git', '-C', str(args.upstream)]
    require(subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip() == protocol['upstream_commit'],
            'Pinned Sigma source mismatch')
    subprocess.run(git + ['diff', '--quiet', 'HEAD', '--'], check=True)
    seed, lr, decay = protocol['seed'], protocol['lr'], protocol['weight_decay']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    signature = {'config': cfg, 'runtime': runtime, 'graph_sha256': split['graph_sha256'],
                 'sources': {f: sha256(BRIDGE / f) for f in ('experiments/r1f/train.py', 'experiments/r1f/split.py',
                              'experiments/r1f/scene.py', 'experiments/r1e/train.py', 'experiments/r1e/state.py')}}
    last = args.checkpoint_dir / 'last.pt'
    if args.resume:
        state = torch.load(last, map_location='cpu', weights_only=False)
        require(state['format'] == VERSION and state['signature'] == signature,
                'Only a matching checkpoint from this R1f experiment may resume')
    else:
        require(not any(args.checkpoint_dir.glob('*.pt')) and not (args.checkpoint_dir / 'best.json').exists(),
                'Use a fresh directory for the VMamba-initialized experiment')
        state = None
    args.output.mkdir(parents=True, exist_ok=True)
    events = args.output / 'events.jsonl'
    events.touch(exist_ok=False)
    emit(events, {'event': 'start', 'signature': signature, 'initialization': 'R1f resume' if state else 'official VMamba-T only',
                  'test_pixels_read': 0, 'training_entries': N, 'validation_images': 79})
    with construct_model(args.upstream, args.init) as (model, config, keys):
        from utils.init_func import group_weight

        model = model.cuda().train()
        groups, coverage = complete_groups(model, group_weight([], model, torch.nn.BatchNorm2d, lr), lr, decay)
        verify_coverage(model, groups, coverage, read(BRIDGE / 'reports/evidence/r1d/first/parameter-coverage.json'))
        optimizer = torch.optim.AdamW(groups, lr=lr, betas=tuple(protocol['betas']), weight_decay=decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        sampler = EpochSampler(N, seed)
        progress = {'updates': 0, 'class_pixels': [0] * 9, 'loss_sum': 0., 'train_seconds': 0.,
                    'peak_allocated_bytes': 0, 'peak_reserved_bytes': 0}
        history = []
        emit(events, {'event': 'initialization', 'optimizer_states': len(optimizer.state),
                      'covered_tensors': coverage['tensor_count'], **keys})
        if state:
            completed = state['sampler']['epoch'] - (state['sampler']['cursor'] < N)
            history = state['history']
            require([x['epoch'] for x in history] == list(range(1, completed + 1)), 'Invalid R1f history')
            strict = restore(state, model, optimizer, scheduler, sampler, seed)
            progress = state['progress']
            emit(events, {'event': 'restore', 'strict_model_keys': strict, 'epoch': sampler.epoch,
                          'cursor': sampler.cursor, 'updates': progress['updates']})
            if history:
                check_best(args.checkpoint_dir, history, sampler, last)
            del state

        def persist():
            t0 = time.perf_counter()
            save(last, model, optimizer, scheduler, sampler, progress, history, signature)
            record = {'event': 'save', 'epoch': sampler.epoch, 'cursor': sampler.cursor,
                      'updates': progress['updates'], 'sha256': sha256(last), 'bytes': last.stat().st_size,
                      'seconds': time.perf_counter() - t0}
            emit(events, record)
            return record

        ids = split['train_ids']
        while len(history) < cfg['epochs']:
            if sampler.cursor == N:
                sampler.advance_epoch()
                progress = {'updates': progress['updates'], 'class_pixels': [0] * 9, 'loss_sum': 0.,
                            'train_seconds': 0., 'peak_allocated_bytes': 0, 'peak_reserved_bytes': 0}
            torch.cuda.reset_peak_memory_stats()
            while sampler.cursor < N:
                names = [ids[i] for i in sampler.next_window(WINDOW)]
                row = train_window(model, optimizer, scheduler, names, args.root, config)
                progress['updates'] += 1
                progress['class_pixels'] = [a + b for a, b in zip(progress['class_pixels'], row['class_pixels'])]
                progress['loss_sum'] += row['pixel_loss_sum']
                progress['train_seconds'] += row['window_seconds']
                for field in ('peak_allocated_bytes', 'peak_reserved_bytes'):
                    progress[field] = max(progress[field], row[field])
                local = progress['updates'] - (sampler.epoch - 1) * STEPS
                emit(events, {'event': 'update', 'epoch': sampler.epoch, 'epoch_update': local,
                              'cursor': sampler.cursor, 'update': progress['updates'], **row})
                if local % 32 == 0 and sampler.cursor < N:
                    persist()
            result = validate(model, args.root, split['validation_ids'], config)
            validation = dict(result, **class_metrics(result['confusion_matrix']))
            history.append({'epoch': sampler.epoch, 'train_mean_pixel_loss': progress['loss_sum'] / sum(progress['class_pixels']),
                            'train_class_pixels': progress['class_pixels'], 'train_seconds': progress['train_seconds'],
                            'peak_allocated_bytes': max(progress['peak_allocated_bytes'], torch.cuda.max_memory_allocated()),
                            'peak_reserved_bytes': max(progress['peak_reserved_bytes'], torch.cuda.max_memory_reserved()),
                            'validation': validation})
            checkpoint = persist()
            best = check_best(args.checkpoint_dir, history, sampler, last)
            write_json(args.output / 'epochs.json', history)
            emit(events, {'event': 'epoch_complete', **history[-1], 'checkpoint': checkpoint, 'best': best})
        emit(events, {'event': 'complete', 'epochs': len(history), 'updates': progress['updates']})


def main():
    import fcntl

    parser = argparse.ArgumentParser()
    for name in ('upstream', 'init', 'root', 'output', 'checkpoint-dir'):
        parser.add_argument('--' + name, type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    require(not args.root.is_relative_to(BRIDGE) and not args.checkpoint_dir.is_relative_to(BRIDGE),
            'Keep assets and weights outside the bridge')
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (args.checkpoint_dir / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == '__main__':
    main()
