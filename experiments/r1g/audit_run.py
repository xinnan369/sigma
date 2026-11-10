"""Independent reconciliation of the executed training and confusion hierarchy."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.r1.assets import sha256
from experiments.r1c.data import write_json
from experiments.r1d.train import class_metrics
from experiments.r1g.partition import ROOT, TRAIN, UPDATES, VAL, read, references, require, verify
from experiments.r1g.statistics import aggregate, cohorts, grouped


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def audit(evidence, checkpoint_dir):
    graph, prior = references()
    split = read(evidence / 'split.json')
    certificate = read(evidence / 'isolation.json')
    lists = {k: v['ids'] for k, v in read(ROOT / 'reports/evidence/r1c/lists.json')['lists'].items()}
    require(verify(split, graph, prior, lists) == certificate['class_support'], 'Frozen split is invalid')
    history = read(evidence / 'gpu/epochs.json')
    events = rows(evidence / 'gpu/events.jsonl')
    start, init, finish = events[0], events[1], events[-1]
    require(start['event'] == 'start' and start['initialization'] == 'official VMamba-T only'
            and start['test_pixels_read'] == 0 and start['verified_input_files'] == 4233
            and init['event'] == 'initialization' and init['optimizer_states'] == 0
            and init['covered_tensors'] == 668, 'Fresh initialization or pinned file verification missing')
    require(finish == {'event': 'complete', 'epochs': 5, 'updates': 785}
            and [e['epoch'] for e in history] == [1, 2, 3, 4, 5], 'Five epochs not complete')
    sig = start['signature']
    require(sig['config']['split_sha256'] == split['split_sha256']
            and sig['config']['selected_assets_sha256'] == split['selected_assets_sha256']
            and sig['graph_sha256'] == split['graph_sha256'], 'Run signature differs from the certificate')
    for name, digest in sig['sources'].items():
        require(sha256(ROOT / name) == digest, f'Executed source changed: {name}')
    generator = torch.Generator().manual_seed(3407)
    group_by_id = {i: group[0] for group in graph['groups'] for i in group}
    results = []
    for epoch in range(1, 6):
        updates = [r for r in events if r['event'] == 'update' and r['epoch'] == epoch]
        indices = torch.randperm(TRAIN, generator=generator).tolist()
        require([r['update'] for r in updates] == list(range((epoch - 1) * UPDATES + 1, epoch * UPDATES + 1))
                and [r['microbatches'] for r in updates] == [8] * 156 + [6]
                and [i for r in updates for i in r['sample_ids']] == [split['train_ids'][i] for i in indices],
                'Actual epoch sampling or six-sample tail differs')
        for i, r in enumerate(updates, 1):
            counts = r['valid_pixel_counts']
            weighted = sum(n * loss for n, loss in zip(counts, r['losses']))
            require(r['epoch_update'] == i and r['cursor'] == min(i * 8, TRAIN)
                    and len(counts) == len(r['losses']) == len(r['sample_ids']) == r['microbatches']
                    and min(counts) > 0 and sum(counts) == sum(r['class_pixels'])
                    and math.isclose(weighted, r['pixel_loss_sum'])
                    and math.isclose(weighted / sum(counts), r['mean_pixel_loss'])
                    and math.isfinite(r['gradient_l2']) and r['gradient_tensor_count'] == 668
                    and r['lr'] == 6e-5, 'Invalid actual window accounting or optimizer settings')
        images = rows(evidence / f'gpu/images-epoch{epoch}.jsonl')
        groups = rows(evidence / f'gpu/groups-epoch{epoch}.jsonl')
        require([x['id'] for x in images] == split['validation_ids']
                and all(x['group'] == group_by_id[x['id']]
                        and np.asarray(x['confusion_matrix']).sum(1).tolist() == graph['label_pixels'][x['id']]
                        for x in images)
                and grouped(images) == groups, 'Validation image/group totals differ')
        row = history[epoch - 1]
        for cohort, ids in cohorts(split).items():
            require(aggregate([x for x in images if x['id'] in ids]) == row['validation'][cohort],
                    f'{cohort} metrics did not reconcile')
        metrics = row['validation']['all']
        stat = certificate['class_support']['validation']
        require(metrics['sample_count'] == VAL and metrics['visual_groups'] == stat['visual_group_count']
                and metrics['class_image_support'] == stat['images'] and metrics['class_group_support'] == stat['groups']
                and metrics['pixel_count'] == VAL * 480 * 640, 'Validation supports changed')
        counts = np.sum([r['class_pixels'] for r in updates], axis=0)
        require(counts.tolist() == row['train_class_pixels']
                and math.isclose(sum(r['pixel_loss_sum'] for r in updates) / counts.sum(), row['train_mean_pixel_loss']),
                'Training metrics did not reconcile')
        details = [dict(class_metrics(x['confusion_matrix']), group=x['group'], ids=x['ids']) for x in groups]
        (evidence / f'gpu/group-metrics-epoch{epoch}.jsonl').write_text(
            ''.join(json.dumps(x, allow_nan=False) + '\n' for x in details), encoding='utf-8', newline='\n')
        done = next(r for r in events if r['event'] == 'epoch_complete' and r['epoch'] == epoch)
        require(done['updates'] == epoch * UPDATES
                and done['overall_miou'] == metrics['miou_including_0']
                and done['foreground_miou'] == metrics['foreground_miou'], 'Checkpoint metrics differ')
        results.append({'epoch': epoch, 'updates': len(updates), 'microsteps': TRAIN,
                        'tail_pixels': updates[-1]['valid_pixel_counts'],
                        'zero_gradient_windows': sum(bool(r['zero_gradient_names']) for r in updates),
                        'gradient_l2_range': [min(r['gradient_l2'] for r in updates), max(r['gradient_l2'] for r in updates)],
                        'checkpoint': done})
    best = read(checkpoint_dir / 'best.json')
    winner = max(history, key=lambda x: x['validation']['all']['miou_including_0'])['epoch']
    last = results[-1]['checkpoint']
    require(best['epoch'] == winner and sha256(checkpoint_dir / 'best.pt') == best['sha256']
            == results[winner - 1]['checkpoint']['sha256']
            and sha256(checkpoint_dir / 'last.pt') == last['sha256'], 'Checkpoint digest mismatch')
    return {'audit': 'PASS', 'fresh_official_VMamba_initialization': True, 'epochs': results,
            'primary_epoch': 5, 'best': best, 'last': last, 'official_test_evaluation': 'NOT RUN'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.evidence, args.checkpoint_dir)
    write_json(args.evidence / 'run-audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
