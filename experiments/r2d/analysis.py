"""Paired crossed resampling of training seeds and shared visual groups."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.r1.assets import sha256
from experiments.r1c.data import canonical_sha, write_json
from experiments.r1g.partition import read, require
from experiments.r2a.analyze import interval, ratio
from experiments.r2a.run import cases, records
from experiments.r2c.analysis import risk_columns, risk_values
from experiments.r2c.protocol import ARMS, load_protocol
from experiments.r2d.profile import design


def bootstrap_weights(seeds, groups, replicates, seed):
    rng = np.random.default_rng(seed)
    return (rng.multinomial(seeds, np.full(seeds, 1 / seeds), size=replicates),
            rng.multinomial(groups, np.full(groups, 1 / groups), size=replicates).astype(np.float64))


def seed_mean(values, weights):
    valid = np.isfinite(values)
    return ratio((np.where(valid, values, 0) * weights.T).sum(0), (valid * weights.T).sum(0))


def summarize(points, values, seed_weights):
    valid = points[np.isfinite(points)]
    avg = float(valid.mean()) if len(valid) else float('nan')
    return {**interval(avg, seed_mean(values, seed_weights)),
            'seed_sd': float(valid.std(ddof=1)) if len(valid) >= 2 else None,
            'per_seed': [float(v) if np.isfinite(v) else None for v in points]}


def sample_matrices(groups, field, weights):
    m = np.array([g[field] for g in groups], dtype=np.float64)
    return (weights @ m.reshape(len(m), 81)).reshape(-1, 9, 9)


def iou(matrix):
    diagonal = np.diagonal(matrix, axis1=1, axis2=2)
    return ratio(diagonal, matrix.sum(1) + matrix.sum(2) - diagonal)


def sampled_metrics(groups, weights):
    counts = weights @ risk_columns(groups)
    result = {k: ratio(counts[:, j], counts[:, 0])
              for j, k in enumerate(('pixels', 'new', 'corrected', 'net', 'disagree')) if j}
    result.update(risk_values(counts))
    a = iou(sample_matrices(groups, 'clean_matrix', weights))
    b = iou(sample_matrices(groups, 'corrupt_matrix', weights))
    for name, matrix in (('clean', a), ('corrupt', b)):
        for k, x in (('miou', matrix), ('foreground_miou', matrix[:, 1:])):
            result[f'{name}_{k}'] = ratio(np.nansum(x, axis=1), np.isfinite(x).sum(1))
        for k in range(9):
            result[f'{name}_iou_{k}'] = matrix[:, k]
    for k in ('miou', 'foreground_miou', *(f'iou_{i}' for i in range(9))):
        result[f'delta_{k}'] = result[f'corrupt_{k}'] - result[f'clean_{k}']
    for band in groups[0]['by_distance']:
        table = np.array([[g['by_distance'][band][k] for k in ('pixels', 'disagree')] for g in groups])
        sampled = weights @ table
        result[f'disagree_distance_{band}'] = ratio(sampled[:, 1], sampled[:, 0])
    return result


def cohort(evidence, seeds, profile):
    results, completed = {}, []
    for seed in seeds:
        path = evidence / f'seed{seed}'
        if not (path / 'analysis/summary.json').exists():
            continue
        stats = read(path / 'analysis/summary.json')
        audit = read(path / 'audit-epoch5.json')
        require(stats['status'] == audit['status'] == 'PASS' and audit['through_epoch'] == 5
                and stats['visual_groups'] == 54 and stats['images'] == 157
                and audit['signature']['seed'] == seed
                and audit['signature']['resource_profile_sha256'] == sha256(profile),
                f'Incompatible seed {seed} evidence')
        index = {}
        for r in records(path / 'analysis/visual-group-counts.jsonl'):
            index.setdefault((r['case'], r['arm']), []).append(r)
        completed.append(seed)
        results[seed] = index
    return completed, results


def analyze(evidence):
    design_config, _ = design()
    _, perturbations, _ = load_protocol()
    profile = evidence / 'resource-profile.json'
    require(read(profile)['config_sha256'] == canonical_sha(design_config), 'Resource profile differs')
    seeds, inputs = cohort(evidence, design_config['seeds'], profile)
    require(len(seeds) >= 3, 'At least three audited complete seeds are required for multi-seed statistics')
    seed_weights, group_weights = bootstrap_weights(len(seeds), 54, design_config['bootstrap_replicates'],
                                                   design_config['bootstrap_seed'])
    combined_groups = np.concatenate([np.ones((1, 54)), group_weights], axis=0)
    output = {}
    for spec in cases(perturbations):
        name = spec['id']
        baseline = [g['group'] for g in inputs[seeds[0]][name, 'ssm']]
        require(len(set(baseline)) == len(baseline) == 54, 'Unexpected visual groups')
        raw, model = {}, {}
        for family in ARMS:
            blocks = [inputs[seed][name, family] for seed in seeds]
            require(all([g['group'] for g in groups] == baseline for groups in blocks), 'Group pairing differs')
            maps = [sampled_metrics(groups, combined_groups) for groups in blocks]
            raw[family] = {k: np.stack([m[k] for m in maps]) for k in maps[0]}
            model[family] = {k: summarize(v[:, 0], v[:, 1:], seed_weights) for k, v in raw[family].items()}
        contrasts = {}
        for other in ('cnn', 'attention'):
            differences = {}
            for k, a in raw['ssm'].items():
                delta = a - raw[other][k]
                s = summarize(delta[:, 0], delta[:, 1:], seed_weights)
                s['seed_directions'] = {'positive': int((delta[:, 0] > 0).sum()),
                                        'negative': int((delta[:, 0] < 0).sum()),
                                        'zero': int((delta[:, 0] == 0).sum())}
                differences[k] = s
            contrasts[f'ssm_minus_{other}'] = differences
        output[name] = {'arms': model, 'contrasts': contrasts}
    result = {'status': 'PASS', 'seeds': seeds, 'groups': 54, 'draws': len(seed_weights),
              'exploratory_seed_count': len(seeds) < 5,
              'estimand': 'equal seed mean; pool pixels within each resampled visual cohort before computing rates',
              'resampling': 'independent seed and visual-group draws; both weights shared across architectures; '
                            'group weights shared across seeds because the same validation groups recur',
              'cases': output}
    write_json(evidence / 'multiseed-summary.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    analyze(parser.parse_args().evidence)


if __name__ == '__main__':
    main()
