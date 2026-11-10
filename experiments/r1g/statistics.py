"""Visual-group confusion aggregation and nonparametric cluster bootstrap."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.r1c.data import write_json
from experiments.r1d.train import class_metrics
from experiments.r1g.partition import require, read

REPLICATES = 10_000
SEED = 3407


def cohorts(split):
    ids = split['validation_ids']
    return {'all': set(ids), 'day': {x for x in ids if x.endswith('D')},
            'night': {x for x in ids if x.endswith('N')}, 'retained_79': set(split['retained_ids']),
            'added_78': set(split['added_ids'])}


def grouped(records):
    out = {}
    ids = set()
    for row in records:
        ident, group = row['id'], row['group']
        require(ident not in ids, 'Repeated validation image')
        ids.add(ident)
        matrix = np.asarray(row['confusion_matrix'], dtype=np.int64)
        require(matrix.shape == (9, 9) and np.all(matrix >= 0), 'Invalid image confusion matrix')
        if group not in out:
            out[group] = {'group': group, 'ids': [], 'confusion_matrix': np.zeros((9, 9), dtype=np.int64)}
        out[group]['ids'].append(ident)
        out[group]['confusion_matrix'] += matrix
    return [dict(out[g], confusion_matrix=out[g]['confusion_matrix'].tolist()) for g in sorted(out)]


def aggregate(records):
    groups = grouped(records)
    require(len(groups) > 0, 'Validation cohort is empty')
    image_matrices = np.array([r['confusion_matrix'] for r in records], dtype=np.int64)
    group_matrices = np.array([r['confusion_matrix'] for r in groups], dtype=np.int64)
    return dict(class_metrics(group_matrices.sum(0)), sample_count=len(records), visual_groups=len(groups),
                class_image_support=(image_matrices.sum(2) > 0).sum(0).tolist(),
                class_group_support=(group_matrices.sum(2) > 0).sum(0).tolist(),
                mean_pixel_loss=sum(r['loss_sum'] for r in records) / image_matrices.sum())


def iou_draws(matrices, weights):
    """Compute pooled-count IoU for each draw, preserving group multiplicities."""
    sampled = (weights @ matrices.reshape(len(matrices), 81)).reshape(len(weights), 9, 9)
    target = sampled.sum(2)
    diag = np.diagonal(sampled, axis1=1, axis2=2)
    union = target + sampled.sum(1) - diag
    iou = np.divide(diag, union, where=union > 0, out=np.full(diag.shape, np.nan))
    return iou, target


def bootstrap(groups, replicates=REPLICATES, seed=SEED):
    require(len(groups) > 0 and replicates > 0, 'Bootstrap requires groups and positive draws')
    matrices = np.asarray([r['confusion_matrix'] for r in groups], dtype=np.int64)
    n = len(matrices)
    # A multinomial vector counts n independent uniform draws with replacement
    # from the n observed groups. Counts weight complete group matrices.
    weights = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=replicates)
    iou, target = iou_draws(matrices, weights)
    support = (matrices.sum(2) > 0).sum(0)
    observed = class_metrics(matrices.sum(0))

    def interval(values, point):
        valid = values[np.isfinite(values)]
        return {'point': point, 'ci95': np.percentile(valid, [2.5, 97.5]).tolist() if len(valid) else None,
                'valid_replicates': len(valid), 'undefined_replicates': replicates - len(valid)}

    def macro(x):
        k = np.isfinite(x).sum(1)
        return np.divide(np.nansum(x, axis=1), k, where=k > 0, out=np.full(len(x), np.nan))

    classes = [dict(interval(iou[:, i], observed['iou_0_to_8'][i]),
                    class_id=i, truth_groups=int(support[i]), statistically_unstable=bool(support[i] < 10),
                    no_ground_truth_replicates=int((target[:, i] == 0).sum())) for i in range(9)]
    return {'resampling_unit': 'frozen visual proxy group', 'seed': seed, 'replicates': replicates,
            'groups': n, 'overall': interval(macro(iou), observed['miou_including_0']),
            'foreground': interval(macro(iou[:, 1:]), observed['foreground_miou']), 'classes': classes,
            'undefined_policy': 'Union=0 is undefined and omitted; no truth with predicted positives gives IoU=0.'}


def run(evidence):
    history = read(evidence / 'gpu/epochs.json')
    split = read(evidence / 'split.json')
    require([r['epoch'] for r in history] == [1, 2, 3, 4, 5], 'Five complete epochs required')
    best = max(history, key=lambda r: r['validation']['all']['miou_including_0'])['epoch']
    output = {'primary_epoch': 5, 'secondary_best_epoch': best,
              'scope': 'conditional on the trained model and visual groups; no test images'}
    for label, epoch in [('primary', 5), ('secondary', best)]:
        if label == 'secondary' and epoch == 5:
            output[label] = {'epoch': 5, 'same_as_primary': True}
            continue
        records = [json.loads(line) for line in (evidence / f'gpu/images-epoch{epoch}.jsonl').read_text().splitlines()]
        values = {}
        for name, ids in cohorts(split).items():
            selected = [r for r in records if r['id'] in ids]
            require({r['id'] for r in selected} == ids and len(selected) == len(ids), 'Incomplete cohort')
            values[name] = bootstrap(grouped(selected))
        output[label] = {'epoch': epoch, 'cohorts': values}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.evidence)
    write_json(args.evidence / 'bootstrap.json', result)
    primary = result['primary']['cohorts']['all']
    print(json.dumps({'replicates': primary['replicates'], 'groups': primary['groups'], 'overall': primary['overall'],
                      'foreground': primary['foreground'], 'classes': primary['classes']}, indent=2))


if __name__ == '__main__':
    main()
