"""Paired inference over the complete repair cohort."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from experiments.r1.assets import sha256
from experiments.r1c.data import write_json
from experiments.r1g.partition import EVIDENCE, ROOT, read, references, require
from experiments.r2a.analyze import ratio
from experiments.r2d.analysis import bootstrap_weights, iou
from experiments.r2e.analysis import inference
from experiments.r2e.paths import mobius
from experiments.r2g.journal import checksum, encode
from experiments.r2g.replay import scopes
from experiments.r2h.metrics import FIELDS
from experiments.r2h.run import ARMS, CONFIG, validate_gate


def rows_for(folder, item, ids, fp, assignments):
    file = folder / item['records']
    ident = item['identity']
    arm, seed = item['arm'], item['seed']
    require(sha256(file) == item['sha256'] and ident == {
        'phase': 'full', 'seed': seed, 'arm': arm, 'ids': ids,
        'checkpoint_sha256': fp['checkpoints'][f'{seed}/{arm}'],
        'fingerprint_sha256': checksum(encode(fp))}, 'Cohort identity differs')
    expected = {'natural_corrupt', 'mismatch/6',
                *(f'{scope}/{m}' for scope in scopes(arm) for m in range(7))}
    order = sorted(ids)
    donor = dict(zip(order, order[1:] + order[:1], strict=True))
    layout = sha256(folder / f'{seed}-{arm}-layout.json')
    rows = []
    with gzip.open(file, 'rt', encoding='utf-8') as stream:
        for line in stream:
            wrapped = json.loads(line)
            r = wrapped['record']
            require(checksum(encode(r)) == wrapped['sha256'] and r['identity_sha256'] == checksum(encode(ident))
                    and r['layout_sha256'] == layout and r['group'] == assignments[r['id']]
                    and r['donor']['id'] == donor[r['id']] and r['cleanup']
                    and set(r['outcomes']) == expected and set(r['prediction_sha256']) == expected | {'clean'}
                    and r['identity_count'] == len(scopes(arm)), 'Row coverage differs')
            scans = 54 if arm == 'ssm' else 42
            for key, count in (('readout', 2 * scans), ('prefix', 2 * scans), ('hybrid', 8 * scans)):
                require(r['numeric'][key]['count'] == count, 'Numerical coverage differs')
            for m in [r['mismatch_numeric'], *(q['maxima'] for q in r['numeric'].values())]:
                require(m['max_mixed_ratio'] <= 1 and m['rms_relative'] <= 2e-5
                        and m['peak_relative'] <= 3e-5, 'Numerical gate failed')
            rows.append(r)
    require([r['id'] for r in rows] == ids and item['count'] == len(ids), 'Image coverage differs')
    return rows


def grouped(rows, groups, weights):
    g = [groups.index(r['group']) for r in rows]

    def pool(values):
        x = np.asarray(values, dtype=np.float64)
        matrix = np.zeros((len(groups), x.shape[1]))
        np.add.at(matrix, g, x)
        return weights @ matrix

    def measures(values, prefix):
        totals = pool([[x[k] for k in FIELDS] for x in values])
        out = {prefix + k + '_rate': ratio(totals[:, j], totals[:, 0])
               for j, k in enumerate(FIELDS) if j}
        out[prefix + 'recovered_minus_introduced_rate'] = ratio(totals[:, 2] - totals[:, 3], totals[:, 0])
        out[prefix + 'recovered_minus_introduced_pixels'] = totals[:, 2] - totals[:, 3]
        out[prefix + 'recovery_fraction'] = ratio(totals[:, 2], totals[:, 1])
        for k in ('induced', 'recovered', 'introduced', 'corrected'):
            out[prefix + k + '_pixels'] = totals[:, FIELDS.index(k)]
        return out

    def performance(matrices):
        x = iou(pool([np.ravel(m) for m in matrices]).reshape(-1, 9, 9))
        out = {f'iou_{i}': x[:, i] for i in range(9)}
        for label, v in [('miou', x), ('fg_miou', x[:, 1:])]:
            out[label] = ratio(np.nansum(v, axis=1), np.isfinite(v).sum(1))
        return out

    clean = performance([r['clean_matrix'] for r in rows])
    result = {'clean': clean}
    natural = performance([r['outcomes']['natural_corrupt']['matrix'] for r in rows])
    for condition in rows[0]['outcomes']:
        v = [r['outcomes'][condition] for r in rows]
        out = measures([r['outside'] for r in v], '')
        for band in v[0]['bands']:
            out.update(measures([r['bands'][band] for r in v], f'distance/{band}/'))
        for i in range(len(v[0]['classes'])):
            out.update(measures([r['classes'][i] for r in v], f'class/{i}/'))
        p = performance([r['matrix'] for r in v])
        out.update(p)
        out.update({f'delta_{k}': p[k] - natural[k] for k in p})
        result[condition] = out
    return result


def comparisons(result, arm):
    out = dict(result)
    pairs = [('encoder/h-minus-c', 'encoder/6', 'encoder/5'),
             ('encoder/h-minus-d', 'encoder/6', 'encoder/3'),
             ('encoder-minus-decoder/h', 'encoder/6', 'decoder/6'),
             ('paired-minus-mismatch/h', 'encoder/6', 'mismatch/6')]
    if arm == 'ssm':
        pairs.append(('fusion-minus-shared/h', 'fusion/6', 'shared/6'))
    focus = ('new_change_rate', 'net_change_rate', 'recovery_fraction', 'introduced_rate',
             'recovered_rate', 'delta_miou', 'delta_fg_miou')
    for name, a, b in pairs:
        out[name] = {k: result[a][k] - result[b][k] for k in focus}
    for scope in scopes(arm):
        masks = [result['natural_corrupt']] + [result[f'{scope}/{7 ^ i}'] for i in range(1, 8)]
        for mask in (3, 5, 6, 7):
            out[f'{scope}/interaction-{mask}'] = {k: mobius([m[k] for m in masks], mask) for k in focus}
    return out


def summarize(points, draws, weights):
    if np.isfinite(points).all():
        return inference(points, draws, weights)
    from experiments.r2d.analysis import summarize as bootstrap_summary

    result = bootstrap_summary(points, draws, weights)
    result.update(seed_t95=None, seed_directions={'positive': int((points > 0).sum()),
                  'negative': int((points < 0).sum()), 'zero': int((points == 0).sum()),
                  'missing': int((~np.isfinite(points)).sum())})
    return result


def analyze(evidence):
    cfg = read(CONFIG)
    folder = evidence / 'full'
    cert = read(folder / 'certificate.json')
    fp = cert['fingerprints']
    ids = read(EVIDENCE / 'split.json')['validation_ids']
    pilot = read(ROOT / 'configs/R2a_exp0.json')['pilot_ids']
    cohorts = {(s, a) for s in cfg['seeds'] for a in ARMS}
    require(cert['phase'] == 'r2h-full' and cert['status'] == 'PASS' and cert['ids'] == ids
            and cert['official_test_pixels_read'] == 0 and len(cert['summary']) == len(cohorts)
            and {(r['seed'], r['arm']) for r in cert['summary']} == cohorts
            and fp['protocol_sha256'] == sha256(CONFIG)
            and cert['pilot_gate_sha256'] == validate_gate(evidence / 'pilot', fp, pilot, 3407),
            'Full certificate differs')
    graph, _ = references()
    assignment = {i: g[0] for g in graph['groups'] for i in g if i in ids}
    groups = sorted(set(assignment.values()))
    require(len(groups) == 54 and set(assignment) == set(ids), 'Frozen group map differs')
    sw, gw = bootstrap_weights(len(cfg['seeds']), len(groups), cfg['draws'], cfg['bootstrap_seed'])
    chunks = []
    audit = {}
    for item in cert['summary']:
        rows = rows_for(folder, item, ids, fp, assignment)
        key = (item['seed'], item['arm'])
        chunks.append((key, rows))
        audit[f'{key[0]}/{key[1]}'] = {
            k: max(r['numeric'][k]['maxima']['max_mixed_ratio'] for r in rows)
            for k in ('readout', 'prefix', 'hybrid')}
        audit[f'{key[0]}/{key[1]}']['mismatch'] = max(r['mismatch_numeric']['max_mixed_ratio'] for r in rows)
    effects = {}
    # Keep all draws for one condition at a time. Replicate weights are identical
    # across arms, conditions and seeds.
    weights = np.vstack((np.ones((1, len(groups))), gw))
    for arm in ARMS:
        by_seed = {s: rows for (s, a), rows in chunks if a == arm}
        cohort_rows = [by_seed[s] for s in cfg['seeds']]
        conditions = list(cohort_rows[0][0]['outcomes'])
        out = effects[arm] = {}
        for condition in ['clean', *conditions]:
            selected = {'natural_corrupt', condition} - {'clean'}
            per_seed = []
            for rows in cohort_rows:
                subset = [{**r, 'outcomes': {k: r['outcomes'][k] for k in selected}} for r in rows]
                per_seed.append(grouped(subset, groups, weights)[condition])
            out[condition] = {}
            for k in per_seed[0]:
                x = np.stack([r[k] for r in per_seed])
                out[condition][k] = summarize(x[:, 0], x[:, 1:], sw)
        # Scope and factorial contrasts need only ROI and global metrics.
        per_seed = []
        for rows in cohort_rows:
            subset = [{**r, 'outcomes': {k: {**v, 'bands': {}, 'classes': []}
                                         for k, v in r['outcomes'].items()}} for r in rows]
            reduced = comparisons(grouped(subset, groups, weights), arm)
            per_seed.append({k: v for k, v in reduced.items() if k not in out})
        for condition in per_seed[0]:
            out[condition] = {}
            for k in per_seed[0][condition]:
                x = np.stack([r[condition][k] for r in per_seed])
                out[condition][k] = summarize(x[:, 0], x[:, 1:], sw)
        print(f'Analyzed {arm}', flush=True)
    write_json(evidence / 'paired-statistics.json', {'status': 'PASS', 'seeds': cfg['seeds'], 'draws': len(sw),
               'groups': groups, 'primary': cfg['primary'], 'numeric': audit, 'effects': effects})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', required=True, type=Path)
    analyze(parser.parse_args().evidence)
