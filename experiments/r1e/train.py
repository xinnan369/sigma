"""Continue the accepted R1d checkpoint for at most five total epochs."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

from experiments.r1.assets import sha256
from experiments.r1c.data import write_json
from experiments.r1c.pilot import construct_model, emit, validate
from experiments.r1c.state import complete_groups
from experiments.r1d.state import EpochSampler
from experiments.r1d.train import BRIDGE, audited_inputs, class_metrics, train_window, verify_coverage
from experiments.r1e.state import N, STEPS, WINDOW, gate, read_parent, replace_best, require, restore, save


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def signatures(cfg):
    import cv2
    import selective_scan_cuda_core

    r1d = BRIDGE / 'reports/evidence/r1d/resume/events.jsonl'
    accepted = json.loads(r1d.open(encoding='utf-8').readline())['signature']
    current = {'config': read(BRIDGE / 'configs/R1d_epoch.json'),
               'source_sha256': {f: sha256(BRIDGE / f) for f in accepted['source_sha256']},
               'python': sys.version, 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
               'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(0),
               'cv2': cv2.__version__, 'numpy': np.__version__,
               'selective_scan_cuda_core_sha256': sha256(Path(selective_scan_cuda_core.__file__))}
    require(current == accepted, 'Runtime or accepted R1d sources differ')
    sources = ('experiments/r1e/train.py', 'experiments/r1e/state.py')
    return current, {'parent_signature': current, 'config': cfg,
                     'sources': {f: sha256(BRIDGE / f) for f in sources}}


def best_epoch(history):
    return max(history, key=lambda x: x['validation']['miou_including_0'])['epoch']


def check_best(directory, history, sampler, last):
    best, metadata = directory / 'best.pt', directory / 'best.json'
    epoch = best_epoch(history)
    if metadata.is_file() and best.is_file():
        try:
            record = read(metadata)
            if record['epoch'] == epoch and record['sha256'] == sha256(best):
                return record
        except (ValueError, KeyError, TypeError):
            pass
    require(sampler.epoch == epoch and sampler.cursor == N,
            'Best checkpoint missing or damaged and last does not contain its weights')
    # Recovery when the process stopped between committing last and copying best.
    replace_best(last, best)
    record = {'epoch': epoch, 'sha256': sha256(best), 'bytes': best.stat().st_size}
    temporary = metadata.with_suffix('.json.tmp')
    write_json(temporary, record)
    temporary.replace(metadata)
    return record


def inherited_epoch():
    root = BRIDGE / 'reports/evidence/r1d'
    summary = read(root / 'summary.json')
    return {'epoch': 1, 'source': 'accepted R1d', 'train_mean_pixel_loss': summary['epoch_mean_pixel_loss'],
            'train_class_pixels': summary['train_class_pixels'], 'train_seconds': summary['train_seconds'],
            'peak_allocated_bytes': summary['peak_allocated_bytes'], 'peak_reserved_bytes': summary['peak_reserved_bytes'],
            'validation': read(root / 'resume/validation.json')['r1d_epoch_1'],
            'parent_checkpoint': summary['final_checkpoint']}


def persist(last, args, model, optimizer, scheduler, sampler, progress, history, signature, event_path):
    t0 = time.perf_counter()
    save(last, model, optimizer, scheduler, sampler, progress, history, signature)
    emit(event_path, {'event': 'save', 'epoch': sampler.epoch, 'cursor': sampler.cursor,
                      'updates': progress['updates'], 'sha256': sha256(last), 'bytes': last.stat().st_size,
                      'seconds': time.perf_counter() - t0})
    if sampler.cursor == N:
        record = check_best(args.checkpoint_dir, history, sampler, last)
        emit(event_path, {'event': 'best', **record})
        write_json(args.output / 'epochs.json', history)


def run(args):
    cfg = read(BRIDGE / 'configs/R1e_convergence.json')
    require((cfg['version'], cfg['max_epoch'], cfg['gate_epoch'], cfg['best_metric'])
            == ('r1e-convergence-v1', 5, 3, 'miou_including_0'), 'Invalid R1e protocol version or bounds')
    require(args.through in (3, 5) and (args.parent is None or args.through == 3), 'First run must stop at epoch 3')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expose one CUDA GPU')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parent_signature, signature = signatures(cfg)
    protocol = parent_signature['config']
    seed, lr, decay = protocol['seed'], protocol['lr'], protocol['weight_decay']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    require(sha256(args.init) == protocol['init_sha256'], 'VMamba init mismatch')
    git = ['git', '-C', str(args.upstream)]
    require(subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip() == protocol['upstream_commit'],
            'Upstream commit mismatch')
    subprocess.run(git + ['diff', '--quiet', 'HEAD', '--'], check=True)
    split = audited_inputs(args.root, protocol)
    last = args.checkpoint_dir / 'last.pt'
    if args.parent:
        require(not any(args.checkpoint_dir.glob('*.pt')) and not (args.checkpoint_dir / 'best.json').exists(),
                'Start in a fresh checkpoint directory')
        state = read_parent(args.parent, cfg['parent_checkpoint_sha256'], parent_signature)
        history = [inherited_epoch()]
        source = cfg['parent_checkpoint_sha256']
    else:
        source = sha256(last)
        state = torch.load(last, map_location='cpu', weights_only=False)
        require(state['format'] == cfg['version'] and state['signature'] == signature,
                'R1e format or signature mismatch')
        history = state['history']
    sampler = EpochSampler(N, seed)
    completed = state['sampler']['epoch'] - (state['sampler']['cursor'] < N)
    require([x['epoch'] for x in history] == list(range(1, completed + 1))
            and history[0] == inherited_epoch(), 'Checkpoint history does not match the epoch boundary')
    if args.through == 5:
        checks = gate(history[:3], cfg)
        require(all(checks.values()), f'Epoch 3 gate failed: {checks}')
    require(state['sampler']['epoch'] <= args.through, 'Checkpoint beyond requested stop')
    args.output.mkdir(parents=True, exist_ok=True)
    event_path = args.output / 'events.jsonl'
    event_path.touch(exist_ok=False)
    emit(event_path, {'event': 'start', 'signature': signature, 'through': args.through,
                      'source_checkpoint_sha256': source, 'test_pixels_read': 0})
    with construct_model(args.upstream, args.init) as (model, config, init_keys):
        from utils.init_func import group_weight

        model = model.cuda().train()
        groups, coverage = complete_groups(model, group_weight([], model, torch.nn.BatchNorm2d, lr), lr, decay)
        verify_coverage(model, groups, coverage,
                        read(BRIDGE / 'reports/evidence/r1d/first/parameter-coverage.json'))
        optimizer = torch.optim.AdamW(groups, lr=lr, betas=tuple(protocol['betas']), weight_decay=decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        keys = restore(state, model, optimizer, scheduler, sampler, seed)
        progress = state['progress']
        del state
        emit(event_path, {'event': 'restored', 'epoch': sampler.epoch, 'cursor': sampler.cursor,
                          'updates': progress['updates'], 'strict_model_keys': keys,
                          'optimizer_states': len(optimizer.state), 'scheduler_step': scheduler.last_epoch,
                          'rng_streams': ['python', 'numpy', 'torch', 'cuda', 'sampler'], **init_keys})
        if args.parent:
            persist(last, args, model, optimizer, scheduler, sampler, progress, history, signature, event_path)
        else:
            check_best(args.checkpoint_dir, history, sampler, last)
        ids = split['train_ids']
        while len(history) < args.through:
            if sampler.cursor == N:
                sampler.advance_epoch()
                progress = {'updates': progress['updates'], 'class_pixels': [0] * 9,
                            'loss_sum': 0., 'train_seconds': 0.,
                            'peak_allocated_bytes': 0, 'peak_reserved_bytes': 0}
            torch.cuda.reset_peak_memory_stats()
            while sampler.cursor < N:
                names = [ids[i] for i in sampler.next_window(WINDOW)]
                out = train_window(model, optimizer, scheduler, names, args.root, config)
                progress['updates'] += 1
                progress['class_pixels'] = [a + b for a, b in zip(progress['class_pixels'], out['class_pixels'])]
                progress['loss_sum'] += out['pixel_loss_sum']
                progress['train_seconds'] += out['window_seconds']
                for field in ('peak_allocated_bytes', 'peak_reserved_bytes'):
                    progress[field] = max(progress[field], out[field])
                local = progress['updates'] - (sampler.epoch - 1) * STEPS
                emit(event_path, {'event': 'update', 'epoch': sampler.epoch, 'cursor': sampler.cursor,
                                  'update': progress['updates'], 'epoch_update': local, **out})
                if sampler.cursor < N and local % cfg['checkpoint_every_updates'] == 0:
                    persist(last, args, model, optimizer, scheduler, sampler, progress, history, signature, event_path)
            result = validate(model, args.root, split['validation_ids'], config)
            validation = dict(result, **class_metrics(result['confusion_matrix']))
            row = {'epoch': sampler.epoch, 'source': 'R1e', 'train_mean_pixel_loss': progress['loss_sum'] / sum(progress['class_pixels']),
                   'train_class_pixels': progress['class_pixels'], 'train_seconds': progress['train_seconds'],
                   'peak_allocated_bytes': max(progress['peak_allocated_bytes'], torch.cuda.max_memory_allocated()),
                   'peak_reserved_bytes': max(progress['peak_reserved_bytes'], torch.cuda.max_memory_reserved()),
                   'validation': validation}
            history.append(row)
            persist(last, args, model, optimizer, scheduler, sampler, progress, history, signature, event_path)
            emit(event_path, {'event': 'epoch_complete', **row})
        write_json(args.output / 'epochs.json', history)
        emit(event_path, {'event': 'complete', 'epoch': len(history), 'updates': progress['updates']})


def main():
    import fcntl

    parser = argparse.ArgumentParser()
    for name in ('upstream', 'init', 'root', 'output', 'checkpoint-dir'):
        parser.add_argument('--' + name, type=lambda p: Path(p).resolve(), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--parent', type=lambda p: Path(p).resolve())
    mode.add_argument('--resume', action='store_true')
    parser.add_argument('--through', type=int, choices=(3, 5), required=True)
    args = parser.parse_args()
    require(not args.checkpoint_dir.is_relative_to(BRIDGE) and not args.root.is_relative_to(BRIDGE),
            'Keep checkpoints and data outside the repository')
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (args.checkpoint_dir / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == '__main__':
    main()
