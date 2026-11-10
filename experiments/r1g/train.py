"""Five fresh Sigma-Tiny epochs, gated on the certified 157-image holdout."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from experiments.r1.assets import sha256
from experiments.r1c.data import read_lists, write_json
from experiments.r1c.pilot import construct_model, emit, read_sample
from experiments.r1c.state import capture_rng, complete_groups
from experiments.r1d.state import EpochSampler
from experiments.r1d.train import BRIDGE, train_window, verify_coverage
from experiments.r1e.state import replace_best
from experiments.r1e.train import signatures
from experiments.r1g.partition import EVIDENCE, TRAIN, UPDATES, VAL, read, references, require, selected_files, verify
from experiments.r1g.statistics import REPLICATES, SEED, aggregate, cohorts, grouped


def write_rows(path, records):
    path.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records),
                    encoding='utf-8', newline='\n')


def evaluate(model, root, config, split, scene, output, epoch):
    model.eval()
    start = time.perf_counter()
    assignment = {i: group[0] for group in scene['groups'] for i in group}
    records = []
    with torch.inference_mode():
        for sample in split['validation_ids']:
            rgb, thermal, label = read_sample(root, sample, config, False)
            logits = model(rgb, thermal)
            require(torch.isfinite(logits).all().item(), f'Nonfinite validation logits: {sample}')
            loss = torch.nn.functional.cross_entropy(logits, label, ignore_index=255, reduction='sum').item()
            pred, target = logits.argmax(1).cpu().numpy(), label.cpu().numpy()
            mask = (target >= 0) & (target < 9)
            hist = np.bincount(9 * target[mask] + pred[mask], minlength=81).reshape(9, 9)
            require(hist.sum(1).tolist() == scene['label_pixels'][sample] and math.isfinite(loss),
                    f'Validation support or CE differs: {sample}')
            records.append({'id': sample, 'group': assignment[sample], 'loss_sum': loss,
                            'confusion_matrix': hist.tolist()})
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    model.train()
    write_rows(output / f'images-epoch{epoch}.jsonl', records)
    write_rows(output / f'groups-epoch{epoch}.jsonl', grouped(records))
    return {name: aggregate([r for r in records if r['id'] in ids])
            for name, ids in cohorts(split).items()}, seconds


def save_checkpoint(path, model, optimizer, scheduler, sampler, history, signature):
    require(all(p.grad is None for p in model.parameters()), 'Checkpoint requires completed gradients')
    tmp = path.with_suffix('.pt.tmp')
    with tmp.open('wb') as stream:
        torch.save({'format': 'r1g-expanded-scene-v1', 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'sampler': sampler.state_dict(), 'rng': capture_rng(),
                    'history': history, 'signature': signature}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def run(args):
    cfg = read(BRIDGE / 'configs/R1g_scene.json')
    require((cfg['version'], cfg['epochs'], cfg['primary_epoch'], cfg['seed'], cfg['bootstrap_replicates'])
            == ('r1g-expanded-scene-v1', 5, 5, SEED, REPLICATES), 'R1g protocol changed')
    scene, prior = references()
    split, cert = (read(EVIDENCE / f) for f in ('split.json', 'isolation.json'))
    require(cert['status'] == 'PASS' and cfg['split_sha256'] == cert['split_sha256'] == split['split_sha256']
            and cfg['selected_assets_sha256'] == cert['selected_assets_sha256'] == split['selected_assets_sha256'],
            'Independent isolation certificate differs')
    require(verify(split, scene, prior, read_lists(args.root)) == cert['class_support'],
            'Recomputed data isolation differs')
    for asset in selected_files(split):
        require(sha256(args.root / asset['path']) == asset['sha256'], f'Changed asset {asset["path"]}')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expose one CUDA GPU')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime, _ = signatures(read(BRIDGE / 'configs/R1e_convergence.json'))
    protocol = runtime['config']
    require(sha256(args.init) == protocol['init_sha256'], 'Pinned VMamba initializer changed')
    git = ['git', '-C', str(args.upstream)]
    require(subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip() == protocol['upstream_commit'],
            'Pinned Sigma source mismatch')
    subprocess.run(git + ['diff', '--quiet', 'HEAD', '--'], check=True)
    seed, lr, decay = protocol['seed'], protocol['lr'], protocol['weight_decay']
    require(seed == SEED and protocol['accumulation'] == 8 and protocol['microbatch'] == 1, 'Runtime protocol mismatch')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    sources = ('experiments/r1g/train.py', 'experiments/r1g/partition.py', 'experiments/r1g/statistics.py')
    signature = {'config': cfg, 'runtime': runtime, 'graph_sha256': split['graph_sha256'],
                 'sources': {name: sha256(BRIDGE / name) for name in sources}}
    last, best = [args.checkpoint_dir / f'{name}.pt' for name in ('last', 'best')]
    require(not any(args.checkpoint_dir.glob('*.pt')) and not (args.checkpoint_dir / 'best.json').exists(),
            'Fresh R1g experiment requires an unused checkpoint directory')
    args.output.mkdir(parents=True, exist_ok=True)
    events = args.output / 'events.jsonl'
    events.touch(exist_ok=False)
    emit(events, {'event': 'start', 'signature': signature, 'initialization': 'official VMamba-T only',
                  'test_pixels_read': 0, 'training_entries': TRAIN, 'validation_images': VAL,
                  'verified_input_files': cert['selected_file_count']})
    with construct_model(args.upstream, args.init) as (model, config, init_keys):
        from utils.init_func import group_weight

        model = model.cuda().train()
        groups, coverage = complete_groups(model, group_weight([], model, torch.nn.BatchNorm2d, lr), lr, decay)
        verify_coverage(model, groups, coverage, read(BRIDGE / 'reports/evidence/r1d/first/parameter-coverage.json'))
        optimizer = torch.optim.AdamW(groups, lr=lr, betas=tuple(protocol['betas']), weight_decay=decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        sampler = EpochSampler(TRAIN, seed)
        history, updates, best_score = [], 0, -1.
        emit(events, {'event': 'initialization', 'optimizer_states': len(optimizer.state),
                      'covered_tensors': coverage['tensor_count'], **init_keys})
        for epoch in range(1, 6):
            require(sampler.epoch == epoch and sampler.cursor == 0, 'Unexpected epoch boundary')
            torch.cuda.reset_peak_memory_stats()
            totals = np.zeros(9, dtype=np.int64)
            loss_sum, train_seconds = 0., 0.
            while sampler.cursor < TRAIN:
                names = [split['train_ids'][i] for i in sampler.next_window(8)]
                result = train_window(model, optimizer, scheduler, names, args.root, config)
                updates += 1
                totals += np.asarray(result['class_pixels'])
                loss_sum += result['pixel_loss_sum']
                train_seconds += result['window_seconds']
                emit(events, {'event': 'update', 'epoch': epoch, 'update': updates,
                              'epoch_update': updates - (epoch - 1) * UPDATES, 'cursor': sampler.cursor, **result})
            require(updates == epoch * UPDATES, 'Incorrect update budget')
            val, val_seconds = evaluate(model, args.root, config, split, scene, args.output, epoch)
            history.append({'epoch': epoch, 'train_mean_pixel_loss': loss_sum / totals.sum(),
                            'train_class_pixels': totals.tolist(), 'train_seconds': train_seconds,
                            'validation_seconds': val_seconds,
                            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                            'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'validation': val})
            t0 = time.perf_counter()
            save_checkpoint(last, model, optimizer, scheduler, sampler, history, signature)
            digest = sha256(last)
            if val['all']['miou_including_0'] > best_score:
                best_score = val['all']['miou_including_0']
                replace_best(last, best)
                write_json(args.checkpoint_dir / 'best.json', {'epoch': epoch, 'sha256': digest})
            write_json(args.output / 'epochs.json', history)
            emit(events, {'event': 'epoch_complete', 'epoch': epoch, 'updates': updates, 'sha256': digest,
                          'checkpoint_bytes': last.stat().st_size, 'checkpoint_seconds': time.perf_counter() - t0,
                          'overall_miou': val['all']['miou_including_0'], 'foreground_miou': val['all']['foreground_miou']})
            if epoch < 5:
                sampler.advance_epoch()
        emit(events, {'event': 'complete', 'epochs': len(history), 'updates': updates})


def main():
    import fcntl

    parser = argparse.ArgumentParser()
    for name in ('root', 'upstream', 'init', 'output', 'checkpoint-dir'):
        parser.add_argument('--' + name, type=lambda p: Path(p).resolve(), required=True)
    args = parser.parse_args()
    require(not args.root.is_relative_to(BRIDGE) and not args.checkpoint_dir.is_relative_to(BRIDGE),
            'Keep dataset and weights outside the repository')
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (args.checkpoint_dir / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == '__main__':
    main()
