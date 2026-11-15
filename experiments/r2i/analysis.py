"""Report every arm and condition with equal within-sample pairing."""
import json

import numpy as np

from experiments.r2a.analyze import interval, ratio
from experiments.r2a.metrics import scores


def summarize(rows, cfg):
    groups = sorted({r['group'] for r in rows})
    if len(groups) != 54:
        raise ValueError('Frozen visual group count differs')
    draws = np.random.default_rng(cfg['bootstrap_seed']).multinomial(
        len(groups), np.full(len(groups), 1 / len(groups)), size=cfg['bootstrap_draws'])
    output = {}
    for case in cfg['validation']:
        selected = [r for r in rows if r['case'] == case]
        if len(selected) != 157:
            raise ValueError('Frozen validation count differs')
        result = scores(np.sum([r['matrix'] for r in selected], axis=0))
        if 'outside' in selected[0]:
            counts = np.zeros((len(groups), 4), dtype=np.int64)
            for row in selected:
                c = row['outside']
                counts[groups.index(row['group'])] += [c[k] for k in ('pixels', 'new', 'corrected', 'net')]
            totals = counts.sum(0)
            sampled = draws @ counts
            result['outside_counts'] = dict(zip(('pixels', 'new', 'corrected', 'net'), totals.tolist()))
            result['net_rate'] = interval(ratio(totals[3], totals[0]), ratio(sampled[:, 3], sampled[:, 0]))
        output[case] = result
    return output


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('records', type=Path)
    parser.add_argument('protocol', type=Path)
    args = parser.parse_args()
    with args.records.open() as stream:
        print(json.dumps(summarize([json.loads(line) for line in stream],
                                   json.loads(args.protocol.read_text())), indent=2))
