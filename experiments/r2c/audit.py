"""Independently reconcile actual training events, checkpoints, and clean metrics."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.r1.assets import sha256
from experiments.r1c.data import canonical_sha, write_json
from experiments.r1g.partition import ROOT, TRAIN, UPDATES, read, references, require
from experiments.r1g.statistics import aggregate
from experiments.r2a.run import records
from experiments.r2c.protocol import ARMS, load_protocol


def expected_windows(ids, seed, epochs):
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(ids), generator=generator).tolist()
        yield [[ids[i] for i in order[k:k + 8]] for k in range(0, len(order), 8)]


def reconcile_window(event, ids, *, epoch, index, cursor, tensor_count, lr, previous):
    require(event['sample_ids'] == ids and event['microbatches'] == len(ids)
            and len(event['valid_pixel_counts']) == len(event['losses']) == len(ids)
            and event['cursor'] == cursor and event['update'] == (epoch - 1) * UPDATES + index
            and event['epoch_update'] == index and event['epoch'] == epoch
            and event['gradient_tensor_count'] == tensor_count and event['lr'] == lr
            and (previous is None or event['augmentation_rng_before_sha256'] == previous)
            and all(p > 0 for p in event['valid_pixel_counts'])
            and all(math.isfinite(v) for v in [event['gradient_l2'], *event['losses']])
            and event['gradient_l2'] > 0, 'Training event fails schedule, loss or gradient audit')
    pixels = sum(event['valid_pixel_counts'])
    weighted_loss = sum(p * loss for p, loss in zip(event['valid_pixel_counts'], event['losses'], strict=True))
    require(sum(event['class_pixels']) == pixels and math.isclose(weighted_loss, event['pixel_loss_sum'])
            and math.isclose(weighted_loss / pixels, event['mean_pixel_loss']), 'Valid pixel or CE totals differ')
    return [event[k] for k in ('sample_ids', 'valid_pixel_counts', 'class_pixels',
                               'augmentation_rng_before_sha256', 'augmentation_rng_after_sha256')]


def audit(evidence, checkpoint_dir, through):
    cfg, _, split = load_protocol()
    graph, _ = references()
    group = {i: g[0] for g in graph['groups'] for i in g}
    preflight = read(evidence / 'training-preflight.json')
    signature = preflight['signature']
    require(preflight['status'] == 'PASS' and signature['config_sha256'] == canonical_sha(cfg)
            and all(sha256(ROOT / p) == v for p, v in signature['sources'].items())
            and preflight['test_pixels_read'] == 0, 'Executed training preflight differs')
    summary, reference_replay, reference_input, common = {}, None, None, None
    windows = list(expected_windows(split['train_ids'], cfg['seed'], through))
    for family in ARMS:
        arm = evidence / 'training' / family
        init, coverage, history, ledger = (read(arm / f'{name}.json') for name in (
            'initialization', 'parameter-coverage', 'epochs', 'checkpoint-ledger'))
        n = init['covered_tensors']
        params = coverage['parameters']
        require(init['family'] == family and init['signature'] == signature
                and init['optimizer_state_entries_before_training'] == 0
                and coverage['missing_count'] == coverage['duplicate_count'] == 0
                and len({p['name'] for p in params}) == n == coverage['tensor_count']
                and sum(p['numel'] for p in params) == coverage['total_numel'] == init['total_parameters']
                and [r['epoch'] for r in history] == [r['epoch'] for r in ledger] == list(range(1, through + 1)),
                'Independent initialization/coverage/history audit failed')
        input_hashes = read(arm / 'first-input-fingerprints.json')
        require([x['id'] for x in input_hashes] == windows[0][0]
                and all(len(x['input_sha256']) == 3 for x in input_hashes), 'First GPU inputs differ')
        if reference_input is None:
            reference_input, common = input_hashes, init['common_state_sha256']
        require(input_hashes == reference_input and init['common_state_sha256'] == common,
                'Actual inputs or common parameters differ across arms')
        replay, epoch_evidence, prev = [], [], None
        for epoch, (expected, result, meta) in enumerate(zip(windows, history, ledger, strict=True), 1):
            path = arm / f'updates-epoch{epoch}.jsonl'
            events = records(path)
            require(len(expected) == len(events) == UPDATES and len(expected[-1]) == 6
                    and len({i for ws in expected for i in ws}) == TRAIN, 'Expected epoch schedule differs')
            cursor, ids = 0, []
            for index, (event, names) in enumerate(zip(events, expected, strict=True), 1):
                cursor += len(names)
                ids.extend(names)
                replay.append(reconcile_window(event, names, epoch=epoch, index=index, cursor=cursor,
                                               tensor_count=n, lr=cfg['lr'], previous=prev))
                prev = event['augmentation_rng_after_sha256']
            totals = np.sum([e['class_pixels'] for e in events], axis=0)
            train_ce = sum(e['pixel_loss_sum'] for e in events) / int(totals.sum())
            require(meta['optimizer_state_entries'] == n and meta['updates'] == meta['scheduler_updates']
                    == result['updates'] == epoch * UPDATES
                    and meta['augmentation_rng_end_sha256'] == prev
                    and totals.tolist() == result['train_class_pixels']
                    and math.isclose(train_ce, result['train_mean_pixel_loss']), 'Epoch optimizer/CE audit failed')
            images = records(arm / f'images-epoch{epoch}.jsonl')
            require([r['id'] for r in images] == split['validation_ids']
                    and all(r['group'] == group[r['id']] and math.isfinite(r['loss_sum'])
                            and np.array(r['confusion_matrix']).sum(1).tolist() == graph['label_pixels'][r['id']]
                            for r in images)
                    and aggregate(images) == result['validation']['all'], 'Clean validation evidence differs')
            epoch_evidence.append({'epoch': epoch, 'update_count': UPDATES, 'tail_size': len(expected[-1]),
                                   'order_sha256': canonical_sha(ids), 'updates_sha256': sha256(path),
                                   'validated_images': len(images), 'optimizer_entries': n})
        require(not records(arm / 'updates-epoch1.jsonl')[0]['zero_gradient_names'], 'First update had zero gradients')
        if reference_replay is None:
            reference_replay = replay
        require(replay == reference_replay, 'Real sample order, augmentation RNG or pixel support differs across arms')
        checkpoints = checkpoint_dir / family
        require(sha256(checkpoints / 'last.pt') == ledger[-1]['last_sha256']
                and sha256(checkpoints / 'best.pt') == ledger[-1]['best_sha256'], 'Checkpoint file hash differs')
        summary[family] = {'epochs': epoch_evidence, 'epoch1_updates_sha256': epoch_evidence[0]['updates_sha256'],
                           'epoch1_checkpoint_sha256': ledger[0]['last_sha256'],
                           'last_sha256': ledger[-1]['last_sha256'], 'best_sha256': ledger[-1]['best_sha256'],
                           'best_epoch': ledger[-1]['best_epoch']}
    require(len({group[i] for i in split['validation_ids']}) == 54, 'Visual group count differs')
    cert = {'status': 'PASS', 'through_epoch': through, 'signature': signature, 'arms': summary,
            'common_initial_sha256': common, 'first_gpu_input_fingerprints_match': True,
            'all_window_sampling_and_augmentation_replay_sha256': canonical_sha(reference_replay),
            'test_pixels_read': 0}
    write_json(evidence / ('audit-epoch1.json' if through == 1 else 'audit-epoch5.json'), cert)
    print(json.dumps({k: v for k, v in cert.items() if k not in ('signature', 'arms')}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--through', required=True, type=int, choices=(1, 5))
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--checkpoint-dir', required=True, type=Path)
    args = parser.parse_args()
    audit(args.evidence, args.checkpoint_dir, args.through)


if __name__ == '__main__':
    main()
