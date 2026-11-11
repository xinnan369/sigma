"""Train the three frozen fusion arms, with a mandatory all-arm Epoch 1 gate."""
import argparse
import os
from pathlib import Path
import random
import time
from unittest.mock import patch

import numpy as np
import torch

from experiments.r1.assets import sha256
from experiments.r1c.data import canonical_sha, write_json
from experiments.r1c.pilot import construct_model, emit, read_sample
from experiments.r1c.state import capture_rng, complete_groups, restore_rng
from experiments.r1d.state import EpochSampler
from experiments.r1d.train import train_window
from experiments.r1e.state import replace_best
from experiments.r1g.partition import ROOT, TRAIN, UPDATES, read, require
from experiments.r1g.train import evaluate
from experiments.r2a.metrics import digest
from experiments.r2a.run import model_digest
from experiments.r2b.controls import GRIDS, WIDTHS, fusion, set_seed, ssm_cost
from experiments.r2c.protocol import ARMS, digest_common, gpu_inputs, optimizer_steps


def save_last(path, model, optimizer, scheduler, sampler, history, signature, arm, init, update):
    require(all(p.grad is None for p in model.parameters()), 'Checkpoint must be at a completed update')
    state = {'format': 'r2c-matched-fusion-v1', 'family': arm, 'signature': signature,
             'initialization': init, 'updates': update, 'model': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'sampler': sampler.state_dict(), 'history': history, 'rng': capture_rng()}
    temp = path.with_suffix('.pt.tmp')
    with temp.open('xb') as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def restore_last(last, model, optimizer, scheduler, sampler, signature, family, init, ledger, cfg):
    meta = ledger[-1]
    require(sha256(last) == meta['last_sha256'], f'{family}: saved last.pt digest differs')
    state = torch.load(last, map_location='cpu', weights_only=False)
    epoch = meta['epoch']
    expected = EpochSampler(TRAIN, cfg['seed'])
    for e in range(1, epoch + 1):
        for _ in range(UPDATES):
            expected.next_window(8)
        if e < epoch:
            expected.advance_epoch()
    saved = state['sampler']
    reference = expected.state_dict()
    require(state['format'] == cfg['version'] and state['family'] == family and state['signature'] == signature
            and state['initialization'] == init and state['updates'] == epoch * UPDATES
            and [r['epoch'] for r in state['history']] == list(range(1, epoch + 1))
            and all(saved[k] == reference[k] for k in ('epoch', 'cursor', 'size', 'order'))
            and torch.equal(saved['generator'], reference['generator']), f'{family}: incompatible epoch checkpoint')
    model.load_state_dict(state['model'], strict=True)
    require(model_digest(model) == meta['model_sha256'], f'{family}: loaded weights differ')
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    sampler.load_state_dict(saved)
    restore_rng(state['rng'])
    require(canonical_sha(random.getstate()) == meta['augmentation_rng_end_sha256'],
            f'{family}: restored augmentation RNG differs')
    optimizer_steps(optimizer, scheduler, init['covered_tensors'], epoch * UPDATES, cfg)
    return state['history']


def first_window(model, optimizer, scheduler, names, root, config):
    inputs = []

    def audited_read(*args):
        values = read_sample(*args)
        inputs.append({'id': args[1], 'input_sha256': [digest(t.cpu().numpy()) for t in values]})
        return values

    with patch('experiments.r1d.train.read_sample', audited_read):
        result = train_window(model, optimizer, scheduler, names, root, config)
    require(len(inputs) == 8 and not result['zero_gradient_names'], 'First update has missing inputs or zero gradients')
    return result, inputs


def cost(model, family):
    blocks = (model.backbone.cross_mamba, model.backbone.channel_attn_mamba)
    count = sum(p.numel() for group in blocks for p in group.parameters())
    if family == 'ssm':
        macs, cells = ssm_cost(blocks)
    else:
        macs = sum(m.macs(h, w, c) for group in blocks for m, (h, w), c in zip(group, GRIDS, WIDTHS, strict=True))
        cells = 0
    return {'fusion_parameters': count, 'fusion_linear_conv_attention_macs': macs,
            'fusion_ssm_scalar_state_updates': cells}


def train_arm(args, cfg, split, graph, signature, family, through):
    evidence = args.evidence / 'training' / family
    checkpoints = args.checkpoint_dir / family
    last, best = checkpoints / 'last.pt', checkpoints / 'best.pt'
    ledger_path = evidence / 'checkpoint-ledger.json'
    seed = cfg['seed']
    set_seed(seed)
    with construct_model(args.upstream, args.init) as (model, upstream_cfg, init_keys), fusion(model, family):
        from utils.init_func import group_weight

        model = model.cuda().train()
        lr, decay = cfg['lr'], cfg['weight_decay']
        groups, coverage = complete_groups(model, group_weight([], model, torch.nn.BatchNorm2d, lr), lr, decay)
        n = coverage['tensor_count']
        require(n == {'ssm': 668, 'cnn': 556, 'attention': 556}[family], 'Unexpected trainable tensor count')
        optimizer = torch.optim.AdamW(groups, lr=lr, betas=tuple(cfg['betas']), weight_decay=decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        sampler = EpochSampler(TRAIN, seed)
        init = {'family': family, 'signature': signature, 'source': 'fresh official VMamba-T classifier initialization',
                'common_state_sha256': digest_common(model), 'whole_initial_state_sha256': model_digest(model),
                'covered_tensors': n, 'total_parameters': coverage['total_numel'],
                'optimizer_state_entries_before_training': len(optimizer.state), **cost(model, family), **init_keys}
        common = args.evidence / 'training/ssm/initialization.json'
        if family != 'ssm':
            require(init['common_state_sha256'] == read(common)['common_state_sha256'], 'Common initial state differs')
        if through == 1:
            evidence.mkdir(parents=True, exist_ok=False)
            checkpoints.mkdir(parents=True, exist_ok=False)
            write_json(evidence / 'initialization.json', init)
            write_json(evidence / 'parameter-coverage.json', coverage)
            ledger, history = [], []
            set_seed(seed)
        else:
            require(init == read(evidence / 'initialization.json')
                    and coverage == read(evidence / 'parameter-coverage.json'), 'Resume initialization or coverage differs')
            ledger = read(ledger_path)
            require([r['epoch'] for r in ledger] == list(range(1, len(ledger) + 1))
                    and 1 <= len(ledger) <= 5 and sha256(best) == ledger[-1]['best_sha256'],
                    'Checkpoint ledger or best.pt differs')
            history = restore_last(last, model, optimizer, scheduler, sampler, signature, family, init, ledger, cfg)
            require(history == read(evidence / 'epochs.json'), 'Epoch history differs from saved checkpoint')
            if len(history) < through:
                sampler.advance_epoch()
        for epoch in range(len(history) + 1, through + 1):
            require((sampler.epoch, sampler.cursor) == (epoch, 0), 'Incorrect epoch boundary')
            event_file = evidence / f'updates-epoch{epoch}.jsonl'
            event_file.touch(exist_ok=False)
            totals = np.zeros(9, dtype=np.int64)
            loss_sum, train_seconds = 0., 0.
            torch.cuda.reset_peak_memory_stats()
            for local in range(1, UPDATES + 1):
                names = [split['train_ids'][i] for i in sampler.next_window(8)]
                require(len(names) == (6 if local == UPDATES else 8), 'Accumulation window length differs')
                before = canonical_sha(random.getstate())
                if epoch == local == 1:
                    result, inputs = first_window(model, optimizer, scheduler, names, args.root, upstream_cfg)
                    write_json(evidence / 'first-input-fingerprints.json', inputs)
                    optimizer_steps(optimizer, scheduler, n, 1, cfg)
                else:
                    result = train_window(model, optimizer, scheduler, names, args.root, upstream_cfg)
                totals += result['class_pixels']
                loss_sum += result['pixel_loss_sum']
                train_seconds += result['window_seconds']
                after = canonical_sha(random.getstate())
                emit(event_file, {'epoch': epoch, 'epoch_update': local, 'update': (epoch - 1) * UPDATES + local,
                                  'cursor': sampler.cursor, 'augmentation_rng_before_sha256': before,
                                  'augmentation_rng_after_sha256': after, **result})
                if local in (1, 40, 80, 120, UPDATES):
                    print(f'{family} epoch {epoch} update {local}/{UPDATES} CE {result["mean_pixel_loss"]:.6f}',
                          flush=True)
            update = epoch * UPDATES
            require(sampler.cursor == TRAIN, 'Epoch did not traverse all records')
            optimizer_steps(optimizer, scheduler, n, update, cfg)
            val, val_seconds = evaluate(model, args.root, upstream_cfg, split, graph, evidence, epoch)
            require(canonical_sha(random.getstate()) == after, 'Validation changed the augmentation RNG')
            history.append({'epoch': epoch, 'updates': update, 'train_mean_pixel_loss': loss_sum / totals.sum(),
                            'train_class_pixels': totals.tolist(), 'train_seconds': train_seconds,
                            'validation_seconds': val_seconds, 'validation': val,
                            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                            'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
            start = time.perf_counter()
            save_last(last, model, optimizer, scheduler, sampler, history, signature, family, init, update)
            last_digest = sha256(last)
            best_epoch = max(history, key=lambda r: r['validation']['all']['miou_including_0'])['epoch']
            if best_epoch == epoch:
                replace_best(last, best)
            best_digest = sha256(best)
            require(best_digest == (last_digest if best_epoch == epoch else ledger[best_epoch - 1]['last_sha256']),
                    'best.pt does not match its selected epoch')
            ledger.append({'epoch': epoch, 'updates': update, 'last_sha256': last_digest,
                           'best_epoch': best_epoch, 'best_sha256': best_digest, 'model_sha256': model_digest(model),
                           'augmentation_rng_end_sha256': after, 'optimizer_state_entries': len(optimizer.state),
                           'scheduler_updates': scheduler.last_epoch, 'checkpoint_seconds': time.perf_counter() - start})
            write_json(evidence / 'epochs.json', history)
            write_json(ledger_path, ledger)
            print(f'{family} epoch {epoch} complete, overall mIoU {val["all"]["miou_including_0"]:.8f}', flush=True)
            if epoch < through:
                sampler.advance_epoch()


def run(args):
    cfg, _, split, graph, signature = gpu_inputs(args)
    through = 1 if args.phase == 'first' else cfg['epochs']
    if through == 1:
        require(not any(args.checkpoint_dir.glob('*/last.pt')) and not (args.evidence / 'training').exists(),
                'Fresh comparison requires unused output directories')
        write_json(args.evidence / 'training-preflight.json',
                   {'status': 'PASS', 'signature': signature, 'verified_files': 4233,
                    'training_entries': TRAIN, 'validation_images': len(split['validation_ids']),
                    'test_pixels_read': 0, 'initialization': 'no trained segmentation checkpoint loaded'})
    else:
        audit = read(args.evidence / 'audit-epoch1.json')
        require(audit['status'] == 'PASS' and audit['through_epoch'] == 1 and audit['signature'] == signature
                and tuple(audit['arms']) == ARMS, 'All-arm Epoch 1 gate required')
        for family in ARMS:
            location = args.evidence / 'training' / family
            require(sha256(location / 'updates-epoch1.jsonl') == audit['arms'][family]['epoch1_updates_sha256']
                    and read(location / 'checkpoint-ledger.json')[0]['last_sha256']
                    == audit['arms'][family]['epoch1_checkpoint_sha256'], 'Epoch 1 evidence changed after audit')
    for family in ARMS:
        train_arm(args, cfg, split, graph, signature, family, through)
        torch.cuda.empty_cache()
    print(f'Training phase {args.phase} completed for all three arms.', flush=True)


def main():
    import fcntl

    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('first', 'finish'))
    for name in ('upstream', 'init', 'root', 'evidence', 'checkpoint-dir'):
        parser.add_argument('--' + name, required=True, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    require(not args.checkpoint_dir.is_relative_to(ROOT) and not args.root.is_relative_to(ROOT)
            and args.checkpoint_dir.name == 'seed3407' and args.checkpoint_dir.parent.name == 'sigma-r2c-runs',
            'Use a dedicated external sigma-r2c-runs/seed3407 checkpoint directory')
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.evidence.mkdir(parents=True, exist_ok=True)
    with (args.checkpoint_dir / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == '__main__':
    main()
