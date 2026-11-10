"""Reconcile the five new training epochs with the certified split and assets."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.r1.assets import sha256
from experiments.r1c.data import base_id, write_json
from experiments.r1d.train import BRIDGE, class_metrics
from experiments.r1e.state import N, STEPS, require
from experiments.r1e.train import best_epoch, read


def audit(root, checkpoint_dir):
    events = [json.loads(line) for line in (root / 'gpu/events.jsonl').read_text().splitlines()]
    split = read(root / 'split.json')
    isolation = read(root / 'isolation.json')
    history = read(root / 'gpu/epochs.json')
    start, init, end = events[0], events[1], events[-1]
    require(start['event'] == 'start' and start['initialization'] == 'official VMamba-T only'
            and start['test_pixels_read'] == 0 and init['event'] == 'initialization'
            and init['optimizer_states'] == 0 and init['covered_tensors'] == 668,
            'Did not start a fresh VMamba-only run')
    require(end == {'event': 'complete', 'epochs': 5, 'updates': STEPS * 5}
            and [e['epoch'] for e in history] == list(range(1, 6)), 'Training did not complete five epochs')
    signature = start['signature']
    require(signature['config']['split_sha256'] == split['split_sha256'] == isolation['split_sha256']
            and signature['graph_sha256'] == split['graph_sha256'] and isolation['status'] == 'PASS',
            'Used split differs from the audit certificate')
    for filename, digest in signature['sources'].items():
        require(sha256(BRIDGE / filename) == digest, 'Executed source changed')
    protocol = signature['runtime']['config']
    generator = torch.Generator().manual_seed(protocol['seed'])
    table = []
    for epoch in range(1, 6):
        indices = torch.randperm(N, generator=generator).tolist()
        updates = [e for e in events if e['event'] == 'update' and e['epoch'] == epoch]
        require([e['update'] for e in updates] == list(range((epoch - 1) * STEPS + 1, epoch * STEPS + 1))
                and [len(e['sample_ids']) for e in updates] == [8] * 176 + [2]
                and [x for e in updates for x in e['sample_ids']] == [split['train_ids'][i] for i in indices],
                'Actual sampler changed, duplicated or omitted an entry')
        for e in updates:
            weights = e['valid_pixel_counts']
            weighted = sum(n * x for n, x in zip(weights, e['losses']))
            require(len(weights) == len(e['losses']) == len(e['sample_ids']) == e['microbatches']
                    and min(weights) > 0 and sum(weights) == sum(e['class_pixels'])
                    and math.isclose(weighted, e['pixel_loss_sum'])
                    and math.isclose(weighted / sum(weights), e['mean_pixel_loss'])
                    and math.isfinite(e['gradient_l2']) and e['gradient_tensor_count'] == 668
                    and e['lr'] == protocol['lr'], 'Window loss, gradient or protocol mismatch')
        row = history[epoch - 1]
        val = row['validation']
        require(all(val[k] == v for k, v in class_metrics(val['confusion_matrix']).items())
                and val['sample_count'] == 79 and val['pixel_count'] == 79 * 480 * 640
                and [c['actual_pixels'] for c in val['classes']] == isolation['class_support']['validation']['pixels']
                and math.isfinite(val['mean_pixel_loss']), 'Evaluation mask totals differ from audited validation')
        counts = np.sum([e['class_pixels'] for e in updates], axis=0)
        require(counts.tolist() == row['train_class_pixels']
                and math.isclose(sum(e['pixel_loss_sum'] for e in updates) / counts.sum(), row['train_mean_pixel_loss']),
                'Training totals do not reconcile')
        saves = [e for e in events if e['event'] == 'save' and e['epoch'] == epoch]
        require([e['cursor'] for e in saves] == [256, 512, 768, 1024, 1280, 1410], 'Unexpected checkpoint cadence')
        table.append({'epoch': epoch, 'updates': len(updates), 'samples': N,
                      'tail_pixels': updates[-1]['valid_pixel_counts'],
                      'gradient_l2_range': [min(e['gradient_l2'] for e in updates), max(e['gradient_l2'] for e in updates)],
                      'zero_gradient_windows': sum(bool(e['zero_gradient_names']) for e in updates),
                      'checkpoint': saves[-1], 'save_seconds': sum(e['seconds'] for e in saves)})
    best = read(checkpoint_dir / 'best.json')
    require(sha256(checkpoint_dir / 'last.pt') == table[-1]['checkpoint']['sha256']
            and sha256(checkpoint_dir / 'best.pt') == best['sha256']
            and best['epoch'] == best_epoch(history)
            and table[best['epoch'] - 1]['checkpoint']['sha256'] == best['sha256'],
            'Independent best/last files differ from the logged checkpoints')
    old = read(BRIDGE / 'reports/evidence/r1c/split.json')
    new_val = set(split['validation_ids'])
    return {'audit': 'PASS', 'epochs': table, 'best': best, 'last': table[-1]['checkpoint'],
            'fresh_official_VMamba_initialization': True, 'test_evaluation': 'NOT RUN',
            'holdout_overlap_with_r1e': len(new_val & set(old['validation_ids'])),
            'r1f_val_previously_in_r1e_train': len(new_val & set(map(base_id, old['train_ids'])))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.checkpoint_dir)
    write_json(args.root / 'run-audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
