"""Analyze the complete, hash-verified R2g cohort with paired resampling."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from experiments.r1.assets import sha256
from experiments.r1c.data import write_json
from experiments.r1g.partition import read, require
from experiments.r2d.analysis import bootstrap_weights
from experiments.r2e.analysis import contrasts, inference, metrics
from experiments.r2g.journal import checksum, encode
from experiments.r2g.replay import scopes
from experiments.r2g.run import ARMS, CONFIG, validate_gate


def scope_contrasts(values, arm):
    result = contrasts(values)
    pairs = [('encoder-minus-decoder', 'encoder', 'decoder')]
    if arm == 'ssm':
        pairs.append(('fusion-minus-shared', 'fusion', 'shared'))
    for name, a, b in pairs:
        for mask in range(1, 8):
            left, right = values[f'{a}/{mask}'], values[f'{b}/{mask}']
            result[f'{name}/{mask}'] = {metric: left[metric] - right[metric] for metric in left}
    return result


def read_rows(folder, item, ids, fingerprint):
    file = folder / item['records']
    identity = item['identity']
    require(sha256(file) == item['sha256'] and identity['ids'] == ids
            and identity['phase'] == 'full' and identity['seed'] == item['seed']
            and identity['arm'] == item['arm']
            and identity['checkpoint_sha256'] == fingerprint['checkpoints'][f"{item['seed']}/{item['arm']}"]
            and identity['fingerprint_sha256'] == checksum(encode(fingerprint)), 'Full record identity differs')
    identity_sha = checksum(encode(identity))
    expected = {'natural_corrupt', *(f'{name}/{m}' for name in scopes(item['arm']) for m in range(1, 8))}
    scans = 54 if item['arm'] == 'ssm' else 42
    with gzip.open(file, 'rt', encoding='utf-8') as handle:
        rows = []
        for line in handle:
            wrapper = json.loads(line)
            row = wrapper['record']
            require(checksum(encode(row)) == wrapper['sha256'] and row['identity_sha256'] == identity_sha
                    and row['cleanup'] and row['identity_count'] == len(scopes(item['arm']))
                    and set(row['outcomes']) == expected and set(row['prediction_sha256']) == expected | {'clean'},
                    'Full record audit or condition coverage differs')
            for name, count in (('readout', 2 * scans), ('prefix', 2 * scans), ('hybrid', 8 * scans)):
                stats = row['numeric'][name]
                require(stats['count'] == count and stats['maxima']['max_mixed_ratio'] <= 1
                        and stats['maxima']['rms_relative'] <= 2e-5 and stats['maxima']['peak_relative'] <= 3e-5,
                        'Full numerical audit differs')
            rows.append(row)
    require([row['id'] for row in rows] == ids and item['count'] == len(ids), 'Full image cohort differs')
    return rows


def analyze(evidence, output):
    from experiments.r1g.partition import EVIDENCE, ROOT

    cfg = read(CONFIG)
    folder = evidence / 'full'
    cert = read(folder / 'certificate.json')
    seeds = cfg['seeds']
    keys = {(s, a) for s in seeds for a in ARMS}
    ids = read(EVIDENCE / 'split.json')['validation_ids']
    pilot_ids = read(ROOT / 'configs/R2a_exp0.json')['pilot_ids']
    fp = cert['fingerprints']
    require(cert['phase'] == 'r2g-full' and cert['status'] == 'PASS' and cert['ids'] == ids
            and cert['official_test_pixels_read'] == 0 and len(cert['summary']) == len(keys)
            and {(r['seed'], r['arm']) for r in cert['summary']} == keys
            and cert['pilot_gate_sha256'] == validate_gate(evidence / 'pilot', fp, pilot_ids, cfg['pilot_seed']),
            'Full results are incomplete or uncertified')
    seed_weights, group_weights = bootstrap_weights(len(seeds), cfg['groups'], cfg['bootstrap_draws'],
                                                     cfg['bootstrap_seed'])
    weights = np.vstack((np.ones((1, cfg['groups'])), group_weights))
    stats, audit, assignments = {}, {}, None
    for item in cert['summary']:
        seed, arm = item['seed'], item['arm']
        rows = read_rows(folder, item, ids, fp)
        groups = sorted({row['group'] for row in rows})
        mapping = {row['id']: row['group'] for row in rows}
        require(len(groups) == cfg['groups'] and (assignments is None or assignments == mapping),
                'Visual group assignments differ')
        assignments = mapping
        values, _ = metrics(rows, weights, groups)
        stats[seed, arm] = scope_contrasts(values, arm)
        audit[f'{seed}/{arm}'] = {
            kind: {'count': sum(row['numeric'][kind]['count'] for row in rows),
                   'maxima': {field: max(row['numeric'][kind]['maxima'][field] for row in rows)
                              for field in rows[0]['numeric'][kind]['maxima']}}
            for kind in ('readout', 'prefix', 'hybrid')}
        print(f'Analyzed {seed}/{arm}', flush=True)
    results = {}
    for arm in ARMS:
        out = results[arm] = {}
        for condition, table in stats[seeds[0], arm].items():
            out[condition] = {}
            for metric in table:
                values = np.stack([stats[s, arm][condition][metric] for s in seeds])
                out[condition][metric] = inference(values[:, 0], values[:, 1:], seed_weights)
    write_json(output, {'status': 'PASS', 'full_certificate_sha256': sha256(folder / 'certificate.json'),
                        'seeds': seeds, 'groups': groups, 'draws': len(seed_weights),
                        'primary': cfg['primary_comparisons'], 'numeric': audit, 'effects': results})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    analyze(args.evidence, args.output)


if __name__ == '__main__':
    main()

