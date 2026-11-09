"""Independently reconcile the two R1d training segments and saved checkpoint."""
import argparse
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch

from experiments.r1.assets import sha256
from experiments.r1c.data import write_json
from experiments.r1d.train import BRIDGE, class_metrics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(evidence, checkpoint):
    def events(segment):
        return [json.loads(line) for line in (evidence / segment / 'events.jsonl').read_text().splitlines()]

    first, resume = events('first'), events('resume')
    combined = first + resume
    updates = [e for e in combined if e['event'] == 'update']
    saves = {e['update']: e for e in combined if e['event'] == 'save'}
    restored = next(e for e in resume if e['event'] == 'restore')
    validation = next(e for e in resume if e['event'] == 'validation')
    start = first[0]
    end = resume[-1]
    cfg = start['signature']['config']
    require(start['signature'] == resume[0]['signature'], 'Resume signature differs')
    require(start['test_pixels_read'] == resume[0]['test_pixels_read'] == 0, 'Unexpected test pixels')
    require([e['update'] for e in updates] == list(range(1, 178)), 'Invalid update sequence')
    require([e['microbatches'] for e in updates] == [8] * 176 + [2], 'Incorrect window lengths')
    require(all(e['epoch'] == 1 and e['cursor'] == min(e['update'] * 8, 1410) for e in updates),
            'Epoch boundary or cursor mismatch')
    train = json.loads((BRIDGE / 'reports/evidence/r1c/split.json').read_text())['train_ids']
    indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(cfg['seed'])).tolist()
    require([name for e in updates for name in e['sample_ids']] == [train[i] for i in indices],
            'Actual sample sequence differs from the full seeded permutation')
    require(all(e['gradient_tensor_count'] == 668 and math.isfinite(e['gradient_l2']) and e['lr'] == cfg['lr']
                for e in updates), 'Gradient coverage, finite norm, or learning rate mismatch')
    for e in updates:
        weights = e['valid_pixel_counts']
        require(len(weights) == len(e['losses']) == len(e['sample_ids']) == e['microbatches']
                and sum(weights) == sum(e['class_pixels']) and min(weights) > 0,
                'Invalid actual window pixel counts')
        weighted = sum(x * n for x, n in zip(e['losses'], weights))
        require(math.isclose(weighted, e['pixel_loss_sum'])
                and math.isclose(weighted / sum(weights), e['mean_pixel_loss']),
                'Logged window loss does not use actual valid-pixel weights')
    require(list(saves) == [32, 64, 96, 128, 160, 177], 'Checkpoint frequency differs')
    require(first[-1]['event'] == 'stopped' and restored['updates'] == restored['scheduler_last_epoch'] == 96
            and restored['epoch'] == 1 and restored['cursor'] == 768 and restored['optimizer_state_tensors'] == 668
            and restored['strict_model_keys'] == '<All keys matched successfully>'
            and restored['sha256'] == saves[96]['sha256'], 'Incorrect checkpoint restoration')
    require(end['event'] == 'complete' and end['updates'] == 177 and end['cursor'] == 1410, 'Incomplete epoch')
    require(sha256(checkpoint) == saves[177]['sha256'], 'On-disk checkpoint hash differs')
    counts = np.sum([e['class_pixels'] for e in updates], axis=0)
    loss_sum = sum(e['pixel_loss_sum'] for e in updates)
    require(counts.tolist() == end['class_pixels'] and math.isclose(loss_sum, end['loss_sum']),
            'Restored running totals differ from event totals')
    comparison = json.loads((evidence / 'resume/validation.json').read_text())
    for record in comparison.values():
        for key, value in class_metrics(record['confusion_matrix']).items():
            require(value == record[key], f'Metric mismatch: {key}')
    r1c, r1d = comparison['r1c_update_13'], comparison['r1d_epoch_1']
    require(r1d == {k: v for k, v in validation.items() if k != 'event'}
            and r1d['sample_count'] == 79 and r1d['pixel_count'] == 79 * 480 * 640
            and all(a['actual_pixels'] == b['actual_pixels'] for a, b in zip(r1c['classes'], r1d['classes'])),
            'Validation comparison did not use the same target distribution')
    durations = [t for e in updates for t in e['microstep_seconds']]
    return {'status': 'PASS', 'epoch': 1, 'train_samples': len(durations), 'updates': len(updates),
            'exact_seeded_permutation': True, 'test_evaluation': 'NOT RUN',
            'tail': updates[-1], 'epoch_mean_pixel_loss': end['mean_pixel_loss'],
            'train_class_pixels': counts.tolist(), 'train_class_fractions': (counts / counts.sum()).tolist(),
            'first_window_loss': updates[0]['mean_pixel_loss'], 'last_window_loss': updates[-1]['mean_pixel_loss'],
            'gradient_l2_min': min(e['gradient_l2'] for e in updates),
            'gradient_l2_max': max(e['gradient_l2'] for e in updates),
            'zero_gradient_windows': sum(bool(e['zero_gradient_names']) for e in updates),
            'train_seconds': end['train_seconds'], 'validation_seconds': r1d['seconds'],
            'checkpoint_seconds': sum(e['seconds'] for e in saves.values()),
            'median_microstep_seconds': statistics.median(durations),
            'p95_microstep_seconds': statistics.quantiles(durations, n=100, method='inclusive')[94],
            'peak_allocated_bytes': max(first[-1]['peak_allocated_bytes'], end['peak_allocated_bytes']),
            'peak_reserved_bytes': max(first[-1]['peak_reserved_bytes'], end['peak_reserved_bytes']),
            'restore': restored, 'final_checkpoint': saves[177]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.evidence, args.checkpoint)
    write_json(args.evidence / 'summary.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
