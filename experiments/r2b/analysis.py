"""Aggregate paired state-reset outcomes using the frozen visual groups."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from experiments.r1.assets import sha256
from experiments.r1c.data import canonical_sha, write_json
from experiments.r1g.partition import ROOT, read, require
from experiments.r2a.analyze import counts_vector, interval, iou_intervals, rate_intervals, ratio
from experiments.r2a.metrics import COUNTS
from experiments.r2a.run import records


def audit_rows(evidence, phase, ids, r2a, cfg):
    cert = read(evidence / phase / 'certificate.json')
    path = evidence / phase / 'images.jsonl'
    require(cert['status'] == 'PASS' and cert['ids'] == ids and cert['records_sha256'] == sha256(path)
            and cert['signature']['config_sha256'] == canonical_sha(cfg)
            and cert['signature']['r2a_sha256'] == canonical_sha(r2a), f'{phase}: certificate differs')
    for name, expected in cert['signature']['sources'].items():
        require(sha256(ROOT / 'experiments/r2b' / name) == expected, f'Executed source differs: {name}')
    rows = records(path)
    require([r['id'] for r in rows] == ids and all(len(r['audits']) == 24 for r in rows), 'Incomplete audit')
    return cert, rows


def aggregate(rows, condition, refs, groups, weights):
    by_group = {g: [] for g in groups}
    for r in rows:
        stats = r['base_corrupt'] if condition == 'base_corrupt' else r['conditions'][condition]
        ref = refs[r['id']]
        require(r['group'] == ref['group'] and r['clean_prediction_sha256'] == ref['prediction_sha256'],
                'Pair alignment differs')
        by_group[r['group']].append((stats, ref))
    result = []
    for g, pairs in by_group.items():
        data = [p for p, _ in pairs]
        result.append({'group': g, 'images': len(data),
                       'outside': np.sum([counts_vector(p['outside']) for p in data], axis=0).tolist(),
                       'by_distance': {band: np.sum([counts_vector(p['by_distance'][band]) for p in data], axis=0).tolist()
                                       for band in data[0]['by_distance']},
                       'clean_matrix': np.sum([ref['confusion_matrix'] for _, ref in pairs], axis=0).tolist(),
                       'changed_matrix': np.sum([p['corrupt_matrix'] for p in data], axis=0).tolist(),
                       'whole_image_flips_vs_corrupt': sum(p.get('prediction_flips_vs_input_baseline', 0) for p in data)})
    counts = np.array([g['outside'] for g in result])
    by_distance = {band: np.array([g['by_distance'][band] for g in result]) for band in result[0]['by_distance']}
    require(np.array_equal(sum(by_distance.values()), counts) and np.array_equal(counts[:, 1] - counts[:, 2], counts[:, 3]),
            'Paired metrics do not reconcile')
    matrices = [np.array([g[name] for g in result]) for name in ('clean_matrix', 'changed_matrix')]
    values = {'outside': rate_intervals(counts, weights),
              'by_distance': {band: rate_intervals(c, weights) for band, c in by_distance.items()},
              'confusion': iou_intervals(*matrices, weights),
              'whole_image_flips_vs_corrupt': sum(g['whole_image_flips_vs_corrupt'] for g in result)}
    return values, result, {'outside': counts, **by_distance}


def native_errors(rows):
    return {kind: {field: max(a[kind][field] for r in rows for a in r['audits'])
                   for field in ('max_abs', 'peak_relative', 'rms_relative', 'max_mixed_ratio')}
            for kind in ('output', 'chunk_end_h')}


def traces(rows):
    values = defaultdict(list)
    for row in rows:
        for a in row['trajectory_summary']:
            for state in a['states']:
                values[a['module'], state['direction']].append(state)
    return [{'module': k[0], 'direction': k[1], 'images': len(rs),
             'images_with_state_response': sum(r['nonzero_state_steps'] > 0 for r in rs),
             'mean_rms_h': float(np.mean([r['rms_state_difference'] for r in rs])),
             'mean_rms_c': float(np.mean([r['readout_c_rms_difference'] for r in rs])),
             'peak_mean_rms_bin_index': int(np.argmax(np.mean([r['bin_mean_step_rms'] for r in rs], axis=0)))}
            for k, rs in sorted(values.items())]


def analyze(evidence):
    cfg = read(ROOT / 'configs/R2b_protocol.json')
    r2a = read(ROOT / 'configs/R2a_exp0.json')
    ids = read(ROOT / 'reports/evidence/r1g/split.json')['validation_ids']
    p, pilot = audit_rows(evidence, 'pilot', r2a['pilot_ids'], r2a, cfg)
    f, full = audit_rows(evidence, 'full', ids, r2a, cfg)
    require(p['signature'] == f['signature'] and p['model_state_sha256'] == f['model_state_sha256'],
            'Pilot and full model lineage differ')
    for r in pilot:
        for k in ('identity', 'corrupt_self'):
            require(r['conditions'][k]['changed_reset_state'] == r['conditions'][k]['logit_max_delta']
                    == r['conditions'][k]['prediction_flips_vs_input_baseline'] == 0, 'Nonzero Identity')
    refs = {r['id']: r for r in records(ROOT / 'reports/evidence/r2a/clean/clean.jsonl')}
    primary = {r['id']: r for r in records(ROOT / 'reports/evidence/r2a/full/pairs-primary.jsonl')}
    for r in full:
        expect = primary[r['id']]
        require(r['corrupt_prediction_sha256'] == expect['prediction_sha256']
                and all(expect[k] == v for k, v in r['base_corrupt'].items()), 'R2a primary pair changed')
    groups = sorted({r['group'] for r in full})
    n = len(groups)
    require(n == 54, 'Visual group count differs')
    weights = np.random.default_rng(cfg['seed']).multinomial(n, np.full(n, 1 / n),
                                                            size=cfg['bootstrap_replicates'])
    conditions, group_rows, counts = {}, {}, {}
    for name in ('base_corrupt', 'primary_reset', 'wrong_time'):
        conditions[name], group_rows[name], counts[name] = aggregate(full, name, refs, groups, weights)
    for name in ('primary_reset', 'wrong_time'):
        contrast = {}
        for region, base in counts['base_corrupt'].items():
            reset = counts[name][region]
            require(np.array_equal(base[:, 0], reset[:, 0]), 'Paired denominator changed')
            diff = reset - base
            draws = weights @ diff
            contrast[region] = {k: interval(ratio(diff[:, j].sum(), base[:, 0].sum()),
                                            ratio(draws[:, j], weights @ base[:, 0]))
                                for j, k in enumerate(COUNTS) if j}
        conditions[name]['vs_base_corrupt'] = contrast
    out = {'status': 'PASS', 'images': len(full), 'visual_groups': n, 'replicates': len(weights),
           'seed': cfg['seed'], 'native_error_maxima': {'pilot': native_errors(pilot), 'full': native_errors(full)},
           'pilot_chains_to_argmax': p['causal_chain_with_prediction_flip'],
           'full_chains_to_argmax': f['causal_chain_with_prediction_flip'],
           'state_direction_summaries': traces(full), 'conditions': conditions}
    write_json(evidence / 'summary.json', out)
    (evidence / 'visual-group-counts.jsonl').write_text(
        ''.join(json.dumps({'condition': k, **r}, separators=(',', ':')) + '\n'
                for k, rs in group_rows.items() for r in rs), encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in out.items() if k not in ('conditions', 'state_direction_summaries')}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', required=True, type=Path)
    analyze(parser.parse_args().evidence)


if __name__ == '__main__':
    main()
