"""Reconcile actual R1e samples, epoch metrics and checkpoint lineage."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.r1.assets import sha256
from experiments.r1c.data import base_id, write_json
from experiments.r1d.train import BRIDGE, class_metrics
from experiments.r1e.state import N, STEPS, gate, require
from experiments.r1e.train import best_epoch, inherited_epoch, read


def events(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def audit(root, checkpoint_dir, through):
    phases = [root / 'phase3'] + ([root / 'phase5'] if through == 5 else [])
    streams = [events(p / 'events.jsonl') for p in phases]
    combined = [e for stream in streams for e in stream]
    history = read(phases[-1] / 'epochs.json')
    require(history[0] == inherited_epoch() and [r['epoch'] for r in history] == list(range(1, through + 1)),
            'Epoch history incomplete or changed')
    cfg = read(BRIDGE / 'configs/R1e_convergence.json')
    signature = streams[0][0]['signature']
    protocol = signature['parent_signature']['config']
    require(signature['config'] == cfg and all(x[0]['signature'] == signature and x[0]['test_pixels_read'] == 0
                                                for x in streams), 'Signature changed or test pixels read')
    for name, digest in signature['sources'].items():
        require(sha256(BRIDGE / name) == digest, 'Executed source changed')
    require([s[-1]['epoch'] for s in streams] == ([3, 5] if through == 5 else [3])
            and all(s[-1]['event'] == 'complete' for s in streams), 'GPU segment incomplete')
    split = read(BRIDGE / 'reports/evidence/r1c/split.json')
    require(split['split_sha256'] == protocol['split_sha256']
            and not set(map(base_id, split['train_ids'])) & set(split['validation_ids']),
            'Internal validation isolation changed')
    previous_leakage = read(BRIDGE / 'reports/evidence/r1c/leakage-audit.json')
    require(not previous_leakage['training_test_exact_rgb_thermal_pair_duplicates']
            and not previous_leakage['internal_train_val_exact_rgb_thermal_pair_duplicates'],
            'Frozen R1c asset overlap')
    generator = torch.Generator().manual_seed(protocol['seed'])
    by_epoch = []
    target_pixels = [c['actual_pixels'] for c in history[0]['validation']['classes']]
    for epoch in range(1, through + 1):
        permutation = torch.randperm(N, generator=generator).tolist()
        if epoch == 1:
            continue
        rows = [e for e in combined if e['event'] == 'update' and e['epoch'] == epoch]
        require([r['update'] for r in rows] == list(range((epoch - 1) * STEPS + 1, epoch * STEPS + 1)),
                'Global update sequence changed')
        require([len(r['sample_ids']) for r in rows] == [8] * 176 + [2]
                and [n for r in rows for n in r['sample_ids']] == [split['train_ids'][i] for i in permutation],
                'Epoch order or tail changed')
        for i, r in enumerate(rows, 1):
            counts = r['valid_pixel_counts']
            require(r['cursor'] == min(i * 8, N) and r['epoch_update'] == i
                    and r['microbatches'] == len(counts) == len(r['sample_ids']) == len(r['losses'])
                    and min(counts) > 0 and sum(counts) == sum(r['class_pixels'])
                    and math.isclose(sum(a * b for a, b in zip(counts, r['losses'])), r['pixel_loss_sum'])
                    and math.isfinite(r['gradient_l2']) and r['gradient_tensor_count'] == 668
                    and r['lr'] == protocol['lr'], 'Window normalization, gradients, or optimizer protocol changed')
        record = history[epoch - 1]
        validation = record['validation']
        reconstructed = class_metrics(validation['confusion_matrix'])
        require(all(validation[k] == v for k, v in reconstructed.items())
                and validation['sample_count'] == 79 and validation['pixel_count'] == 79 * 480 * 640
                and target_pixels == [c['actual_pixels'] for c in validation['classes']]
                and math.isfinite(validation['mean_pixel_loss']), 'Validation statistics differ')
        counts = np.sum([r['class_pixels'] for r in rows], axis=0)
        require(counts.tolist() == record['train_class_pixels']
                and math.isclose(sum(r['pixel_loss_sum'] for r in rows) / counts.sum(), record['train_mean_pixel_loss']),
                'Epoch training totals differ')
        saves = [e for e in combined if e['event'] == 'save' and e['epoch'] == epoch]
        require([e['cursor'] for e in saves] == [256, 512, 768, 1024, 1280, 1410], 'Checkpoint cadence differs')
        by_epoch.append({'epoch': epoch, 'updates': len(rows), 'samples': N,
                         'tail_pixels': rows[-1]['valid_pixel_counts'],
                         'gradient_l2_range': [min(r['gradient_l2'] for r in rows), max(r['gradient_l2'] for r in rows)],
                         'zero_gradient_windows': sum(bool(r['zero_gradient_names']) for r in rows),
                         'checkpoint': saves[-1], 'save_seconds': sum(r['seconds'] for r in saves)})
    restored = [e for e in combined if e['event'] == 'restored']
    require([(e['epoch'], e['updates'], e['cursor']) for e in restored]
            == ([(1, 177, N), (3, 531, N)] if through == 5 else [(1, 177, N)])
            and all(e['optimizer_states'] == 668 and e['scheduler_step'] == e['updates']
                    and e['strict_model_keys'] == '<All keys matched successfully>' for e in restored),
            'Restore evidence differs')
    require(streams[0][0]['source_checkpoint_sha256'] == cfg['parent_checkpoint_sha256']
            and (through == 3 or streams[1][0]['source_checkpoint_sha256'] == by_epoch[1]['checkpoint']['sha256']),
            'Source checkpoint lineage differs')
    best = read(checkpoint_dir / 'best.json')
    require(sha256(checkpoint_dir / 'last.pt') == by_epoch[-1]['checkpoint']['sha256']
            and sha256(checkpoint_dir / 'best.pt') == best['sha256'] and best['epoch'] == best_epoch(history),
            'Final last/best snapshot mismatch')
    checks = gate(history[:3], cfg)
    return {'audit': 'PASS', 'through_epoch': through, 'gate': checks, 'continue_to_5': all(checks.values()),
            'by_epoch': by_epoch, 'best': best, 'restored': restored, 'test_evaluation': 'NOT RUN'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--through', type=int, choices=(3, 5), required=True)
    args = parser.parse_args()
    result = audit(args.root, args.checkpoint_dir, args.through)
    write_json(args.root / f'audit-through-{args.through}.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
