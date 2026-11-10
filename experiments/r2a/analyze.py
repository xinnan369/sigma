"""Audit paired records and bootstrap whole visual groups with shared weights."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from experiments.r1.assets import sha256
from experiments.r1c.data import canonical_sha, write_json
from experiments.r1g.partition import ROOT, read, require
from experiments.r1g.statistics import iou_draws
from experiments.r2a.metrics import COUNTS, mask_record, scores
from experiments.r2a.run import cases, masks, records

MEASURES = ('normalized_l2', 'outside_response_ratio', 'inside_mean', 'outside_mean', 'outside_area_fraction')


def ratio(numer, denom):
    return np.divide(numer, denom, where=denom != 0, out=np.full(np.shape(numer), np.nan))


def interval(point, values):
    good = values[np.isfinite(values)]
    return {'point': float(point) if np.isfinite(point) else None,
            'ci95': np.percentile(good, [2.5, 97.5]).tolist() if len(good) else None,
            'valid_draws': len(good)}


def counts_vector(counts):
    return [counts[key] for key in COUNTS]


def rate_intervals(counts, weights):
    point = counts.sum(0)
    draws = weights @ counts
    return {'counts': dict(zip(COUNTS, point.tolist())),
            'rates': {key: interval(ratio(point[i], point[0]), ratio(draws[:, i], draws[:, 0]))
                      for i, key in enumerate(COUNTS) if i}}


def iou_intervals(clean, corrupt, weights):
    clean_iou = iou_draws(clean, weights)[0]
    corrupt_iou = iou_draws(corrupt, weights)[0]
    a, b = scores(clean.sum(0)), scores(corrupt.sum(0))

    def macro(x):
        return ratio(np.nansum(x, axis=1), np.isfinite(x).sum(1))

    result = {}
    for field, sl in [('miou', slice(None)), ('foreground_miou', slice(1, None))]:
        x, y = macro(clean_iou[:, sl]), macro(corrupt_iou[:, sl])
        result[field] = {'clean': interval(a[field], x), 'corrupt': interval(b[field], y),
                         'paired_delta': interval(b[field] - a[field], y - x)}
    result['class_iou'] = [{'clean': interval(a['iou_0_to_8'][i], clean_iou[:, i]),
                            'corrupt': interval(b['iou_0_to_8'][i], corrupt_iou[:, i]),
                            'paired_delta': interval(b['iou_0_to_8'][i] - a['iou_0_to_8'][i],
                                                     corrupt_iou[:, i] - clean_iou[:, i])} for i in range(9)]
    return result


def audit_pair(row, ref, expect_support, expect_case, mask_size):
    require(row['group'] == ref['group'] and row['clean_prediction_sha256'] == ref['prediction_sha256']
            and row['support'] == expect_support and row['case'] == expect_case, 'Pair identity differs')
    clean = np.array(ref['confusion_matrix'])
    corrupt = np.array(row['corrupt_matrix'])
    require(row['clean'] == scores(clean) and row['corrupt'] == scores(corrupt)
            and np.array_equal(clean.sum(1), corrupt.sum(1)), 'Full-image metrics changed')
    a, b, t = [np.array(row[k]) for k in ('clean_matrix_outside', 'corrupt_matrix_outside', 'pred_transition_outside')]
    stats = row['outside']
    require(a.sum() == b.sum() == t.sum() == stats['pixels'] == clean.sum() - mask_size
            and np.array_equal(a.sum(1), b.sum(1))
            and np.array_equal(a.sum(0), t.sum(1)) and np.array_equal(b.sum(0), t.sum(0))
            and stats['net'] == stats['new'] - stats['corrected'] == np.trace(a) - np.trace(b)
            and stats['disagree'] == stats['pixels'] - np.trace(t)
            and stats['new'] + stats['corrected'] <= stats['disagree'], 'Outside error/transition counts do not reconcile')
    for collection in (row['by_true_class_outside'], list(row['by_distance'].values())):
        if collection:
            require(np.sum([counts_vector(x) for x in collection], axis=0).tolist() == counts_vector(stats),
                    'Stratified totals do not reconcile')
    if expect_case == 'identity':
        require(row['prediction_sha256'] == ref['prediction_sha256'] and row['changed_input_pixels'] == 0
                and stats['disagree'] == 0 and not row['by_distance'], 'Identity is not negative')


def collect_case(rows, refs, groups, weights):
    grouped = {g: [] for g in groups}
    for r in rows:
        grouped[r['group']].append(r)
    group_rows = []
    for g, rs in grouped.items():
        def sum_counts(collection):
            return dict(zip(COUNTS, np.sum([counts_vector(r) for r in collection], axis=0).tolist()))
        matrices = {k: np.sum([r[k] for r in rs], axis=0).tolist()
                    for k in ('corrupt_matrix', 'clean_matrix_outside', 'corrupt_matrix_outside', 'pred_transition_outside')}
        group_rows.append({'group': g, 'images': len(rs),
                           'images_with_new_error': sum(r['outside']['new'] > 0 for r in rs),
                           'clean_matrix': np.sum([refs[r['id']]['confusion_matrix'] for r in rs], axis=0).tolist(),
                           **matrices, 'outside': sum_counts([r['outside'] for r in rs]),
                           'by_distance': {k: sum_counts([r['by_distance'][k] for r in rs]) for k in rs[0]['by_distance']},
                           'by_true_class_outside': [sum_counts([r['by_true_class_outside'][i] for r in rs]) for i in range(9)]})
    counts = np.array([counts_vector(g['outside']) for g in group_rows], dtype=np.int64)
    matrices = [np.array([g[k] for g in group_rows], dtype=np.int64) for k in ('clean_matrix', 'corrupt_matrix')]
    summary = {'paired_images': len(rows), 'groups_with_new_errors': sum(g['outside']['new'] > 0 for g in group_rows),
               'images_with_new_errors': sum(g['images_with_new_error'] for g in group_rows),
               'outside': rate_intervals(counts, weights),
               'by_distance': {k: rate_intervals(np.array([counts_vector(g['by_distance'][k]) for g in group_rows]), weights)
                               for k in group_rows[0]['by_distance']},
               'by_true_class_outside': [rate_intervals(np.array([counts_vector(g['by_true_class_outside'][i]) for g in group_rows]), weights)
                                         for i in range(9)],
               'confusion': iou_intervals(*matrices, weights)}
    return summary, group_rows, counts


def responses(rows, refs, groups, weights):
    by_slot = defaultdict(list)
    for row in rows:
        require(row['clean_prediction_sha256'] == refs[row['id']]['prediction_sha256']
                and len(row['response']) == 28, 'Response pair or shape count differs')
        for r in row['response']:
            by_slot[(r['module'], r['slot'])].append((row['group'], r))
    require(len(by_slot) == 28 and all(len(v) == len(rows) for v in by_slot.values()), 'Response slots differ')
    result, group_rows = [], []
    for (module, slot), values in sorted(by_slot.items()):
        slot_shape = values[0][1]['shape_bhwc']
        require(all(v['shape_bhwc'] == slot_shape for _, v in values), 'Feature grid shape varies unexpectedly')
        sums = np.zeros((len(groups), len(MEASURES)), dtype=np.float64)
        sizes = np.zeros_like(sums, dtype=np.int64)
        idx = {g: i for i, g in enumerate(groups)}
        for group, row in values:
            for j, key in enumerate(MEASURES):
                if row[key] is not None:
                    sums[idx[group], j] += row[key]
                    sizes[idx[group], j] += 1
        point = ratio(sums.sum(0), sizes.sum(0))
        draws = ratio(weights @ sums, weights @ sizes)
        result.append({'module': module, 'slot': slot, 'shape_bhwc': slot_shape,
                       'zero_response_images': sum(v['normalized_l2'] == 0 for _, v in values),
                       **{name: interval(point[j], draws[:, j]) for j, name in enumerate(MEASURES)}})
        group_rows.extend({'group': group, 'module': module, 'slot': slot,
                           'sums': dict(zip(MEASURES, sums[i].tolist())),
                           'valid_images': dict(zip(MEASURES, sizes[i].tolist()))}
                          for i, group in enumerate(groups))
    return result, group_rows


def analyze(evidence):
    cfg = read(ROOT / 'configs/R2a_exp0.json')
    clean, pilot, full = [read(evidence / p / 'certificate.json') for p in ('clean', 'pilot', 'full')]
    require(all(c['status'] == 'PASS' and c['signature'] == clean['signature']
                and c['model_state_sha256'] == clean['model_state_sha256'] for c in (clean, pilot, full))
            and pilot['identity_replacement_passes'] == 96 and pilot['plain_forward_rechecks'] == 12
            and full['identity_inputs_passed'] == 157 and full['paired_predictions'] == 22 * 157,
            'Engineering gates failed')
    require(canonical_sha(cfg) == clean['signature']['config_sha256'], 'Frozen analysis configuration differs')
    for path, value in clean['signature']['sources'].items():
        require(sha256(ROOT / path) == value, 'Executed experiment source changed')
    clean_path = evidence / 'clean/clean.jsonl'
    require(sha256(clean_path) == clean['records_sha256'], 'Clean digest changed')
    refs = {x['id']: x for x in records(clean_path)}
    ids = full['ids']
    groups = sorted({r['group'] for r in refs.values()})
    require(list(refs) == ids and len(groups) == 54, 'Frozen visual groups differ')
    support = read(evidence / 'full/support.json')
    require(support == {k: mask_record(m) for k, m in masks(cfg).items()}, 'Support certificate differs')
    specs = list(cases(cfg))
    require(read(evidence / 'full/cases.json') == specs, 'Case manifest differs')
    by_case = {s['id']: 'identity' if s['kind'] == 'identity' else s['position'] for s in specs}
    by_case.update({f'replace.{alias}.{i}': 'center' for alias in ('cross', 'concat') for i in range(4)})
    n = len(groups)
    weights = np.random.default_rng(cfg['bootstrap_seed']).multinomial(
        n, np.full(n, 1 / n), size=cfg['bootstrap_replicates'])
    summary, count_data = {}, {}
    out = evidence / 'analysis'
    out.mkdir(exist_ok=True)
    for name, region in by_case.items():
        pair_rows = records(evidence / f'full/pairs-{name}.jsonl')
        require([r['id'] for r in pair_rows] == ids, 'Unmatched full pair order')
        for r in pair_rows:
            audit_pair(r, refs[r['id']], region, name, support[region]['true_count'])
        values, group_rows, count_data[name] = collect_case(pair_rows, refs, groups, weights)
        write_json(out / f'groups-{name}.json', group_rows)
        summary[name] = values
    primary = count_data['primary']
    for name in by_case:
        if not name.startswith('replace.'):
            continue
        changed = count_data[name]
        require(np.array_equal(changed[:, 0], primary[:, 0]), 'Replacement outside support differs')
        diff = changed - primary
        delta = weights @ diff
        counts = diff.sum(0)
        summary[name]['contrast_vs_primary'] = {
            k: interval(ratio(counts[i], primary[:, 0].sum()), ratio(delta[:, i], weights @ primary[:, 0]))
            for i, k in enumerate(COUNTS) if i}
    for spec in specs:
        name = spec['id']
        rows = records(evidence / f'full/response-{name}.jsonl')
        require([r['id'] for r in rows] == ids and all(r['case'] == name and r['group'] == refs[r['id']]['group']
                                                     for r in rows), 'Unmatched full response identity')
        values, by_group = responses(rows, refs, groups, weights)
        if name == 'identity':
            require(all(v['zero_response_images'] == len(ids) and v['outside_response_ratio']['point'] is None
                        for v in values), 'Identity fusion responses are nonzero')
        write_json(out / f'responses-{name}.json', values)
        # Compact records preserve the group-level means' sums and denominators.
        (out / f'group-responses-{name}.jsonl').write_text(
            ''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in by_group), encoding='utf-8', newline='\n')
    result = {'audit': 'PASS', 'images': len(ids), 'groups': n, 'bootstrap_seed': cfg['bootstrap_seed'],
              'replicates': cfg['bootstrap_replicates'], 'unit': 'paired whole visual proxy group',
              'scope': 'conditional on this trained model, frozen visual grouping, and tested conditions',
              'cases': summary}
    write_json(out / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    r = analyze(args.evidence)
    print(json.dumps({k: v for k, v in r.items() if k != 'cases'}, indent=2))
    print(json.dumps(r['cases']['primary']['outside'], indent=2))


if __name__ == '__main__':
    main()
