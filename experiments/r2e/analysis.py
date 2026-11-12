"""Group and seed paired inference for the gated factorial experiment."""
import argparse
from pathlib import Path

import numpy as np

from experiments.r1.assets import sha256
from experiments.r1c.data import write_json
from experiments.r1g.partition import ROOT, read, require
from experiments.r2a.analyze import counts_vector, ratio
from experiments.r2a.run import records
from experiments.r2d.analysis import bootstrap_weights, iou, summarize
from experiments.r2e.paths import mobius


def metrics(rows, weights, group_ids):
    row_group = [group_ids.index(r['group']) for r in rows]
    clean = np.zeros((len(group_ids), 81))
    np.add.at(clean, row_group, [np.ravel(r['clean_matrix']) for r in rows])

    def macro(matrix):
        x = iou((weights @ matrix).reshape(-1, 9, 9))
        return ratio(np.nansum(x, axis=1), np.isfinite(x).sum(1))

    clean_iou = macro(clean)
    result = {}
    group_counts = {}
    conditions = tuple(rows[0]['outcomes'])
    require(all(tuple(r['outcomes']) == conditions for r in rows), 'Factorial conditions differ')
    for condition in conditions:
        values = [r['outcomes'][condition] for r in rows]
        counts = np.zeros((len(group_ids), 5))
        np.add.at(counts, row_group, [counts_vector(r['outside']) for r in values])
        group_counts[condition] = counts.tolist()
        totals = weights @ counts
        metrics = {name: ratio(totals[:, j], totals[:, 0]) for j, name in enumerate(
            ('pixels', 'new', 'corrected', 'net', 'disagree')) if j}
        changed = np.zeros_like(clean)
        np.add.at(changed, row_group, [np.ravel(r['corrupt_matrix']) for r in values])
        metrics['delta_miou'] = macro(changed) - clean_iou
        for band in values[0]['by_distance']:
            table = np.zeros_like(counts)
            np.add.at(table, row_group, [counts_vector(r['by_distance'][band]) for r in values])
            sums = weights @ table
            for j, name in enumerate(('new', 'corrected', 'net', 'disagree'), 1):
                metrics[f'{band}/{name}'] = ratio(sums[:, j], sums[:, 0])
        result[condition] = metrics
    return result, group_counts


def contrasts(values):
    result = dict(values)
    scopes = sorted({name.rsplit('/', 1)[0] for name in values if name != 'natural_corrupt'})
    for scope in scopes:
        for metric in values[f'{scope}/1']:
            masks = [np.zeros_like(values[f'{scope}/1'][metric])]
            masks.extend(values[f'{scope}/{m}'][metric] for m in range(1, 8))
            for mask in (3, 5, 6, 7):
                result.setdefault(f'{scope}/interaction-{mask}', {})[metric] = mobius(masks, mask)
            for label, a, b in (('h-minus-c', 1, 2), ('h-minus-d', 1, 4), ('c-minus-d', 2, 4)):
                result.setdefault(f'{scope}/{label}', {})[metric] = masks[a] - masks[b]
    return result


def inference(points, draws, seed_weights):
    from scipy.stats import t

    out = summarize(points, draws, seed_weights)
    delta = t.ppf(.975, len(points) - 1) * points.std(ddof=1) / np.sqrt(len(points))
    out['seed_t95'] = [float(points.mean() - delta), float(points.mean() + delta)]
    out['seed_directions'] = {'positive': int((points > 0).sum()), 'negative': int((points < 0).sum()),
                              'zero': int((points == 0).sum())}
    return out


def analyze(evidence):
    config = read(ROOT / 'configs/R2e_protocol.json')
    folder = evidence / 'full'
    cert = read(folder / 'certificate.json')
    require(cert['status'] == 'PASS' and cert['phase'] == 'full' and len(cert['ids']) == 157
            and len(cert['summary']) == 15, 'Gated full results are required')
    seeds = config['seeds']
    seed_weights, group_weights = bootstrap_weights(len(seeds), 54, config['bootstrap_draws'], config['bootstrap_seed'])
    weights = np.vstack((np.ones((1, 54)), group_weights))
    inputs, per_seed_counts, groups = {}, {}, None
    for rec in cert['summary']:
        key = (rec['seed'], rec['arm'])
        path = folder / f'{key[0]}-{key[1]}.jsonl'
        require(sha256(path) == rec['records_sha256'], 'Full result hash differs')
        rows = records(path)
        this_groups = sorted({r['group'] for r in rows})
        if groups is None:
            groups = this_groups
        require([r['id'] for r in rows] == cert['ids'] and len(groups) == 54 and this_groups == groups,
                'Visual group cohort differs')
        values, counts = metrics(rows, weights, groups)
        inputs[key] = contrasts(values)
        per_seed_counts[f'{key[0]}/{key[1]}'] = counts
    output = {}
    for arm in ('ssm', 'cnn', 'attention'):
        output[arm] = {}
        for name, measurements in inputs[seeds[0], arm].items():
            output[arm][name] = {}
            for metric in measurements:
                values = np.stack([inputs[s, arm][name][metric] for s in seeds])
                output[arm][name][metric] = inference(values[:, 0], values[:, 1:], seed_weights)
    write_json(evidence / 'paired-statistics.json',
               {'seeds': seeds, 'groups': groups, 'replicates': len(seed_weights),
                'outcome_definition': 'all outcomes relative to clean; mIoU difference in ratio units',
                'statistics': output, 'per_seed_group_outside_counts': per_seed_counts})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    analyze(parser.parse_args().evidence)


if __name__ == '__main__':
    main()
