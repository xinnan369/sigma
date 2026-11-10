"""Shared visual-group bootstrap for the matched three-arm experiment."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.r1.assets import sha256
from experiments.r1c.data import write_json
from experiments.r1g.partition import ROOT, read, require
from experiments.r1g.statistics import iou_draws
from experiments.r2a.analyze import audit_pair, collect_case, counts_vector, interval, ratio
from experiments.r2a.metrics import COUNTS, mask_record
from experiments.r2a.run import append, cases, masks, records
from experiments.r2c.protocol import ARMS, load_protocol


def rate_contrast(ssm, comparator, weights):
    require(ssm.shape == comparator.shape and np.array_equal(ssm[:, 0], comparator[:, 0]),
            'Cross-model denominators differ')
    diff = ssm - comparator
    point, draws = diff.sum(0), weights @ diff
    denom = ssm[:, 0].sum()
    sampled = weights @ ssm[:, 0]
    return {k: interval(ratio(point[i], denom), ratio(draws[:, i], sampled))
            for i, k in enumerate(COUNTS) if i}


def risk_columns(rows):
    counts = np.array([counts_vector(r['outside']) for r in rows])
    # Column order is R2a counts followed by absolute errors.
    a = np.array([np.trace(r['clean_matrix_outside']) for r in rows])
    b = np.array([np.trace(r['corrupt_matrix_outside']) for r in rows])
    result = np.column_stack([counts, counts[:, 0] - a, counts[:, 0] - b])
    require(np.array_equal(result[:, 6] - result[:, 5], result[:, 3]), 'Absolute risk and net errors differ')
    return result


def risk_values(x):
    return {
        'clean_error_rate': ratio(x[..., 5], x[..., 0]),
        'corrupt_error_rate': ratio(x[..., 6], x[..., 0]),
        'relative_net_vs_clean_error': ratio(x[..., 3], x[..., 5]),
        'new_per_clean_correct': ratio(x[..., 1], x[..., 0] - x[..., 5]),
        'correction_per_clean_error': ratio(x[..., 2], x[..., 5]),
    }


def risk_summary(x, weights):
    point, draws = risk_values(x.sum(0)), risk_values(weights @ x)
    return {k: interval(point[k], draws[k]) for k in point}


def risk_contrast(a, b, weights):
    pa, pb = (risk_values(x.sum(0)) for x in (a, b))
    da, db = (risk_values(weights @ x) for x in (a, b))
    return {k: interval(pa[k] - pb[k], da[k] - db[k]) for k in pa}


def compare_iou(a, b, weights):
    def draws(groups, field):
        x = np.array([g[field] for g in groups], dtype=np.int64)
        return iou_draws(x, np.ones((1, len(x)), dtype=np.int64))[0], iou_draws(x, weights)[0]

    def macro(x):
        return ratio(np.nansum(x, axis=1), np.isfinite(x).sum(1))

    pair = {k: tuple(draws(g, k) for g in (a, b)) for k in ('clean_matrix', 'corrupt_matrix')}
    result = {}
    for metric, columns in (('miou', slice(None)), ('foreground_miou', slice(1, None))):
        ca, cb = pair['clean_matrix']
        ra, rb = pair['corrupt_matrix']
        clean = [macro(x[:, columns]) - macro(y[:, columns]) for x, y in zip(ca, cb)]
        corrupt = [macro(x[:, columns]) - macro(y[:, columns]) for x, y in zip(ra, rb)]
        result[metric] = {'clean_difference': interval(clean[0][0], clean[1]),
                          'corrupt_difference': interval(corrupt[0][0], corrupt[1]),
                          'change_difference': interval(corrupt[0][0] - clean[0][0], corrupt[1] - clean[1])}
    return result


def read_arm(out, family, audit, ids, spec_count):
    path = out / family
    c = read(path / 'certificate.json')
    require(c['status'] == 'PASS' and c['family'] == family and c['signature'] == audit['signature']
            and c['checkpoint_sha256'] == audit['arms'][family]['last_sha256']
            and c['images'] == c['identity_exact'] == len(ids) and c['clean_per_image_confusion_exact']
            and c['pair_count'] == spec_count * len(ids) and c['test_pixels_read'] == 0
            and len(c['evidence_sha256']) == spec_count + 1
            and all(sha256(path / n) == h for n, h in c['evidence_sha256'].items()), 'Invalid inference certificate')
    clean = records(path / 'clean.jsonl')
    require([r['id'] for r in clean] == ids, 'Clean cohort order differs')
    return {r['id']: r for r in clean}


def analyze(evidence):
    cfg, r2a, split = load_protocol()
    ids = split['validation_ids']
    audit = read(evidence / 'audit-epoch5.json')
    require(audit['status'] == 'PASS' and audit['through_epoch'] == 5, 'Training audit failed')
    inputs = evidence / 'perturbation'
    specs = list(cases(r2a))
    support = read(inputs / 'support.json')
    require(read(inputs / 'cases.json') == specs
            and support == {k: mask_record(m) for k, m in masks(r2a).items()}, 'R2a case or support differs')
    refs = {arm: read_arm(inputs, arm, audit, ids, len(specs)) for arm in ARMS}
    groups = sorted({r['group'] for r in refs['ssm'].values()})
    require(len(groups) == 54 and all([r['group'] for r in ref.values()] ==
                                     [r['group'] for r in refs['ssm'].values()] for ref in refs.values()),
            'Frozen visual group membership differs')
    n = len(groups)
    weights = np.random.default_rng(cfg['seed']).multinomial(
        n, np.full(n, 1 / n), size=cfg['bootstrap_replicates'])
    output = evidence / 'analysis'
    output.mkdir(exist_ok=True)
    summaries = {}
    with (output / 'visual-group-counts.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
        for spec in specs:
            name = spec['id']
            region = 'identity' if name == 'identity' else spec['position']
            legacy = records(ROOT / 'reports/evidence/r2a/full' / f'pairs-{name}.jsonl')
            require([r['id'] for r in legacy] == ids, 'R2a reference order differs')
            results, group_rows, risks, raw_hashes = {}, {}, {}, None
            for arm in ARMS:
                rows = records(inputs / arm / f'pairs-{name}.jsonl')
                require([r['id'] for r in rows] == ids, 'Pair cohort order differs')
                inputs_by_image = [r['raw_perturbed_sha256'] for r in rows]
                if raw_hashes is None:
                    raw_hashes = inputs_by_image
                require(raw_hashes == inputs_by_image, 'Actual perturbation differs across arms')
                for row, original in zip(rows, legacy, strict=True):
                    audit_pair(row, refs[arm][row['id']], region, name, support[region]['true_count'])
                    require(all(row[k] == original[k] for k in ('support', 'effective_seed', 'changed_input_pixels')),
                            'Perturbation parameters or support differ from R2a')
                values, by_group, _ = collect_case(rows, refs[arm], groups, weights)
                group_rows[arm] = by_group
                risks[arm] = risk_columns(by_group)
                values['risk'] = risk_summary(risks[arm], weights)
                results[arm] = values
                for row in by_group:
                    append(stream, {'arm': arm, 'case': name, **row})
            contrasts = {}
            for other in ('cnn', 'attention'):
                a, b = group_rows['ssm'], group_rows[other]
                contrast = {'outside': rate_contrast(risks['ssm'][:, :5], risks[other][:, :5], weights),
                            'risk': risk_contrast(risks['ssm'], risks[other], weights),
                            'confusion': compare_iou(a, b, weights)}
                contrast['by_distance'] = {band: rate_contrast(
                    np.array([counts_vector(g['by_distance'][band]) for g in a]),
                    np.array([counts_vector(g['by_distance'][band]) for g in b]), weights)
                    for band in a[0]['by_distance']}
                contrast['by_true_class_outside'] = [rate_contrast(
                    np.array([counts_vector(g['by_true_class_outside'][i]) for g in a]),
                    np.array([counts_vector(g['by_true_class_outside'][i]) for g in b]), weights)
                    for i in range(9)]
                contrasts[f'ssm_minus_{other}'] = contrast
            summaries[name] = {'arms': results, 'contrasts': contrasts}
    result = {'status': 'PASS', 'images': len(ids), 'visual_groups': n, 'seed': cfg['seed'],
              'replicates': len(weights), 'primary_case': cfg['primary_case'], 'primary_rates': cfg['primary_rates'],
              'scope': 'paired visual-group uncertainty conditional on one trained checkpoint per arm',
              'cases': summaries}
    write_json(output / 'summary.json', result)
    primary = summaries[cfg['primary_case']]
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))
    print(json.dumps({k: {m: v['outside'][m] for m in cfg['primary_rates']}
                      for k, v in primary['contrasts'].items()}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    analyze(parser.parse_args().evidence)


if __name__ == '__main__':
    main()
