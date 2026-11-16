"""Pair whole training seeds and shared visual groups across all contrasts."""
import argparse
from pathlib import Path

import numpy as np

from experiments.r1c.data import write_json
from experiments.r1g.partition import read, require
from experiments.r2d.analysis import bootstrap_weights, summarize
from experiments.r2i_s2.independent_audit import jac

COHORT = (3407, 3408, 3409)


def estimate(points_and_groups, seed):
    points = points_and_groups[:, 0]
    result = summarize(points, points_and_groups[:, 1:], seed)
    require(np.isfinite(points).all(), 'Nonfinite seed point')
    result['seed_directions'] = ['negative' if p < 0 else 'positive' if p > 0 else 'zero' for p in points]
    return result


def paired_stats(audits):
    group_order = audits[0]['group_order']
    require(tuple(a['seed'] for a in audits) == COHORT
            and all(a['status'] == 'PASS' and a['groups'] == 54 and a['group_order'] == group_order for a in audits),
            'Independent audit cohort differs')
    seeds, groups = bootstrap_weights(3, 54, 10000, 3407)
    weights = np.vstack((np.ones((1,54)), groups))
    output = {}
    for track in ('clean','augmented'):
        data = [a['tracks'][track] for a in audits]
        comparisons = {}
        for name in data[0]['contrasts']:
            arm, ref = name.split('-minus-')
            out = {}
            for case in data[0]['contrasts'][name]:
                rows = [t['contrasts'][name][case] for t in data]
                pixels = np.stack([weights @ np.asarray(r['group_counts']) for r in rows])
                denom = pixels[:,:,0]
                require(np.all(denom > 0) and np.array_equal(denom[0], denom[1])
                        and np.array_equal(denom[0], denom[2]), 'Paired denominators differ')
                out[case] = {k:estimate(pixels[:,:,j]/denom,seeds)
                             for j,k in enumerate(('new','corrected','net','disagree'),1)}
                out[case]['exploratory_did'] = estimate(
                    np.stack([weights@np.asarray(r['group_did']) for r in rows])/denom,seeds)
            ni = {}
            for key, start in (('overall',0),('foreground',1)):
                diffs = []
                for t in data:
                    matrices = t['group_clean_confusions']
                    a,b = [jac((weights@np.asarray(matrices[n]).reshape(54,81)).reshape(-1,9,9))
                           for n in (arm,ref)]
                    diffs.append(np.nanmean(a[:,start:],axis=1) - np.nanmean(b[:,start:],axis=1))
                ni[key] = estimate(np.stack(diffs),seeds)
                ni[key]['noninferior_at_0_5_pp'] = ni[key]['ci95'][0] > -.005
            out['clean_noninferiority'] = ni
            comparisons[name] = out
        output[track] = comparisons
    return {'status':'PASS','seeds':list(COHORT),'visual_groups':group_order,'draws':10000,
            'units':'ratios; across-seed equal-weight mean and paired crossed resampling',
            'tracks':output}


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--audit',type=Path,nargs=3,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    write_json(a.output,paired_stats([read(path) for path in a.audit]))
