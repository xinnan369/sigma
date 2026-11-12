"""Reconcile the certified pilot files and summarize numerical maxima."""
import argparse
from collections import Counter
from pathlib import Path

from experiments.r1.assets import sha256
from experiments.r1c.data import write_json
from experiments.r1g.partition import ROOT, read, require
from experiments.r2a.run import records


def audit(evidence):
    folder = evidence / 'pilot'
    cert = read(folder / 'certificate.json')
    cfg = read(ROOT / 'configs/R2a_exp0.json')
    require(cert['phase'] == 'r2f-pilot' and cert['status'] == 'PASS' and cert['ids'] == cfg['pilot_ids']
            and cert['test_pixels_read'] == 0 and len(cert['fingerprints']['checkpoints']) == 15,
            'Incomplete original-cohort pilot')
    require(all(sha256(ROOT / 'experiments/r2f' / name) == digest for name, digest in
                cert['fingerprints']['r2f_implementation_sha256'].items()), 'Executed implementation differs')
    result = {}
    for item in cert['summary']:
        seed, arm = item['seed'], item['arm']
        file = folder / f'{seed}-{arm}.jsonl'
        require(sha256(file) == item['records_sha256'], 'Pilot record digest differs')
        rows = records(file)
        layout = read(folder / f'{seed}-{arm}-layout.json')
        scopes = Counter(r['scope'] for r in layout)
        expected = {'encoder': 30, 'decoder': 12}
        if arm == 'ssm':
            expected.update({f'cross.{i}': 2 for i in range(4)})
            expected.update({f'concat.{i}': 1 for i in range(4)})
        require(scopes == expected and [r['id'] for r in rows] == cfg['pilot_ids'], 'Pilot coverage differs')
        keys = [r['call'] for r in layout]
        maximum = {k: {} for k in ('native_readout', 'native_prefix', 'hybrid')}
        records_by_kind = Counter()
        for row in rows:
            require([r['call'] for r in row['numeric']] == keys
                    and len(row['outcomes']) == 1 + 7 * (len(scopes) + 1), 'Incomplete numerical or replay outputs')
            for entry in row['numeric']:
                native = entry['reference']['native']
                hybrid = entry['reference']['hybrid']
                require(len(native) == 2 and len(hybrid) == 8, 'Incomplete endpoint or hybrid comparisons')
                for kind, values in (('native_readout', [v['readout'] for v in native]),
                                     ('native_prefix', [v['chunk_end'] for v in native]), ('hybrid', hybrid)):
                    for v in values:
                        require(v['max_mixed_ratio'] <= 1 and v['rms_relative'] <= 2e-5
                                and v['peak_relative'] <= 3e-5, 'Numerical gate failed')
                        records_by_kind[kind] += 1
                        for field, value in v.items():
                            maximum[kind][field] = max(maximum[kind].get(field, 0.), value)
        result[arm] = {'images': len(rows), 'scope_counts': dict(scopes), 'comparison_counts': dict(records_by_kind),
                       'maxima': maximum, 'seconds': item['seconds'], 'peak_allocated': item['peak_allocated']}
    require(set(result) == {'ssm', 'cnn', 'attention'}, 'Architecture cohort differs')
    write_json(evidence / 'pilot-audit.json', {'status': 'PASS', 'arms': result})
    print(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    audit(parser.parse_args().evidence)


if __name__ == '__main__':
    main()
