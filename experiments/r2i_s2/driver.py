"""Train a single frozen S1 cell, with checkpoints at optimizer boundaries."""
import argparse
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import numpy as np
import torch

from experiments.r1.assets import sha256
from experiments.r1c.data import write_json
from experiments.r1c.pilot import construct_model, emit, read_sample
from experiments.r1c.state import capture_rng, complete_groups
from experiments.r1d.state import EpochSampler, load, save
from experiments.r1d.train import train_window
from experiments.r1g.partition import ROOT, read, require
from experiments.r1g.train import write_rows
from experiments.r2a.run import model_digest
from experiments.r2b.controls import set_seed
from experiments.r2c.protocol import gpu_inputs
from experiments.r2i.augment import train_occlusion
from experiments.r2i.gate import attach

from experiments.r2i_s2.protocol import CONFIG, STAGE, require_seed
EPOCHS, UPDATES, SIZE = 5, 157, 1254


def same(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a.detach().cpu(), b.detach().cpu())
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def coordinates(index):
    epoch, window = divmod(index, UPDATES)
    cursor = min(SIZE, 8 * window)
    return epoch + 1, cursor, min(8, SIZE - cursor)


def verify_history(history, ids, seed, *, complete=False):
    expected = EpochSampler(SIZE, seed)
    require(len(ids) == SIZE and (not complete or len(history) == EPOCHS * UPDATES), 'Train budget differs')
    for index, row in enumerate(history):
        epoch, cursor, n = coordinates(index)
        if expected.epoch != epoch:
            expected.advance_epoch()
        seq = [ids[i] for i in expected.next_window(8)]
        require(row['epoch'] == epoch and row['update'] == index + 1
                and row['cursor'] == cursor + n and row['sample_ids'] == seq
                and row['microbatches'] == n and len(row['valid_pixel_counts']) == n,
                'Epoch, optimizer update, or sample ordering differs')
    return expected


def restart(path, model, optimizer, scheduler, sampler, signature):
    progress, _ = load(path, model, optimizer, scheduler, sampler, signature)
    raw = torch.load(path, map_location='cpu', weights_only=False)
    require(same(model.state_dict(), raw['model']) and same(optimizer.state_dict(), raw['optimizer'])
            and same(scheduler.state_dict(), raw['scheduler']) and same(sampler.state_dict(), raw['sampler'])
            and same(capture_rng(), raw['rng']), 'Restored tensors or RNG differ')
    return progress


def run(args):
    import selective_scan_cuda_core
    from experiments.r2i.run import resources
    from experiments.r2i_s1.evaluate import evaluate_cell

    seed = require_seed(args.seed)
    cfg = read(CONFIG)
    require(cfg['seed'] == 3407 and cfg['accumulation'] == 8 and args.arm in cfg['arms']
            and args.track in cfg['tracks'], 'Frozen configuration differs')
    require(not args.output.is_relative_to(ROOT), 'Use external run storage')
    args.output.mkdir(parents=True, exist_ok=True)
    base, r2a, split, graph, inherited = gpu_inputs(args)
    require(len(split['train_ids']) == SIZE and base['lr'] == 6e-5 and base['precision'] == 'fp32',
            'Training inputs differ')
    sources = sorted(list((ROOT / 'experiments/r2i').glob('*.py'))
                     + list(Path(__file__).parent.glob('*.py')))
    sources += [ROOT / n for n in ('experiments/r1d/state.py', 'experiments/r1d/train.py',
                'experiments/r1c/state.py', 'experiments/r1c/pilot.py', 'experiments/r2c/protocol.py',
                'experiments/r1g/partition.py', 'experiments/r2i_s1/evaluate.py')]
    signature = {'protocol_sha256': sha256(CONFIG), 'sources': {p.relative_to(ROOT).as_posix(): sha256(p)
                 for p in sources}, 'frozen_inputs': inherited, 'epochs': EPOCHS, 'seed': seed, 'stage_sha256': sha256(STAGE),
                 'track': args.track, 'arm': args.arm, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                 'gpu': torch.cuda.get_device_name(), 'extension_sha256': sha256(Path(selective_scan_cuda_core.__file__))}
    folder = args.output / f'seed{seed}' / f'{args.track}-{args.arm}'
    folder.mkdir(exist_ok=True, parents=True)
    last = folder / 'local.pt'
    require(not (folder / 'failure.log').exists(), 'Unresolved group failure; inspect failure.log')
    require(not (folder / 'complete.json').exists(), 'Cell is already complete')
    set_seed(seed)
    with construct_model(args.upstream, args.init) as (model, config, _):
        from utils.init_func import group_weight

        common = model_digest(model)
        with attach(model, args.arm):
            set_seed(seed)
            model.cuda()
            cost = resources(model, args.arm)
            model.train()
            lr = base['lr']
            groups, cov = complete_groups(model, group_weight([], model, torch.nn.BatchNorm2d, lr),
                                          lr, base['weight_decay'])
            optimizer = torch.optim.AdamW(groups, lr=lr, betas=tuple(base['betas']))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
            sampler = EpochSampler(SIZE, seed)
            history = []
            if args.resume:
                prior = restart(last, model, optimizer, scheduler, sampler, signature)
                history = prior['history']
                require(prior['mode'] == ('smoke' if args.smoke else 'formal'), 'Run mode differs')
                require(same(verify_history(history, split['train_ids'], seed).state_dict(), sampler.state_dict()),
                        'Resumed sampler differs from frozen permutation')
                emit(folder / 'runtime.jsonl', {'event': 'resume', 'updates': len(history),
                     'model_sha256': model_digest(model), 'checkpoint_sha256': sha256(last)})
            else:
                require(not last.exists(), 'Use --resume for existing checkpoint')
            max_updates = 2 if args.smoke else EPOCHS * UPDATES
            if args.smoke and not args.resume:
                max_updates = 1
            torch.cuda.reset_peak_memory_stats()
            try:
                while len(history) < max_updates:
                    if sampler.cursor == SIZE:
                        sampler.advance_epoch()
                    epoch = sampler.epoch
                    ids = [split['train_ids'][i] for i in sampler.next_window(8)]
                    aug = []

                    def sample(root, name, config, training):
                        rgb, thermal, label = read_sample(root, name, config, training)
                        thermal, roi = train_occlusion(thermal, config.norm_mean, config.norm_std, seed,
                                                       name, cfg['train_probability'] if args.track == 'augmented'
                                                       else 0., cfg['train_roi'], epoch=epoch)
                        aug.append(roi)
                        return rgb, thermal, label

                    with patch('experiments.r1d.train.read_sample', side_effect=sample):
                        row = train_window(model, optimizer, scheduler, ids, args.root, config)
                    row.update(epoch=epoch, cursor=sampler.cursor, update=len(history)+1, augmentation=aug)
                    require(row['gradient_tensor_count'] == cov['tensor_count'], 'Gradient coverage differs')
                    history.append(row)
                    emit(folder / 'runtime.jsonl', {'event': 'window', **row})
                    if args.smoke or len(history) % 10 == 0 or sampler.cursor == SIZE:
                        save(last, model, optimizer, scheduler, sampler,
                             {'updates': len(history), 'history': history, 'mode': 'smoke' if args.smoke else 'formal'}, signature)
                    if sampler.cursor == SIZE or args.smoke:
                        print(args.track, args.arm, epoch, len(history), flush=True)
            except Exception:
                (folder / 'failure.log').write_text(traceback.format_exc(), encoding='utf-8')
                raise
            if args.smoke:
                return
            verify_history(history, split['train_ids'], seed, complete=True)
            require(len(optimizer.state) == cov['tensor_count']
                    and all(x['step'].item() == 785 for x in optimizer.state.values()),
                    'Final optimizer coverage differs')
            write_rows(folder / 'training.jsonl', history)
            started = time.perf_counter()
            evaluate_cell(model, args.root, config, cfg, split, graph, r2a, folder)
            write_json(folder / 'complete.json', {'status': 'PASS', 'signature': signature,
                       'checkpoint_sha256': sha256(last), 'model_sha256': model_digest(model),
                       'common_initial_sha256': common, 'coverage': cov['tensor_count'], 'resources': cost,
                       'updates': len(history), 'training_seconds': sum(r['window_seconds'] for r in history),
                       'training_peak_bytes': max(r['peak_allocated_bytes'] for r in history),
                       'evaluation_seconds': time.perf_counter()-started})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('root', 'init', 'upstream', 'output'):
        p.add_argument('--' + key, type=lambda s: Path(s).resolve(), required=True)
    p.add_argument('--seed', type=int, choices=(3408, 3409), required=True)
    p.add_argument('--track', choices=('clean', 'augmented'), required=True)
    p.add_argument('--arm', choices=('baseline', 'gated', 'control'), required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    run(p.parse_args())
