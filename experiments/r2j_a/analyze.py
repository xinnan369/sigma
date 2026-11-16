"""Analyze immutable S1/S2 pairs without accessing pixels or running a model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'reports/evidence/r2i-s2'
SEEDS = (3407, 3408)
TRACKS = ('clean', 'augmented')
COMPARISONS = ('gated-minus-baseline', 'control-minus-baseline', 'gated-minus-control')
CONDITIONS = ('clean', 'primary', 'random_occlusion', 'gaussian')
CLASSES = ('unlabeled/background', 'car', 'person', 'bike', 'curve', 'car stop', 'guardrail', 'color cone', 'bump')
FIELDS = ('pixels', 'new', 'corrected', 'net', 'disagree')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def vector(counts):
    x = np.array([counts[k] for k in FIELDS], dtype=np.int64)
    n, new, corrected, net, disagree = x
    require(n >= disagree >= new + corrected >= 0 and min(new, corrected) >= 0
            and net == new - corrected, 'Contradictory paired counts')
    return x


def interval(table, weights):
    sums = weights @ table
    require(sums[0,0] > 0, 'Stratum has no valid ROI-outside pixels')
    valid = sums[1:,0] > 0
    rate = sums[0,3] / sums[0,0]
    ci = np.percentile(sums[1:,3][valid] / sums[1:,0][valid], [2.5, 97.5]) if valid.any() else None
    return {'counts': dict(zip(FIELDS,sums[0].astype(int).tolist())), 'rate_pp': 100*rate,
            'ci95_pp': (100*ci).tolist() if ci is not None else None, 'valid_draws': int(valid.sum()),
            'groups_with_pixels': int(np.count_nonzero(table[:,0])),
            'groups_negative': int(np.count_nonzero(table[:,3]<0)),
            'groups_positive': int(np.count_nonzero(table[:,3]>0))}


def stratify(rows, ids, groups, assignment, weights):
    require(len(rows) == len(ids) and {r['id'] for r in rows} == set(ids), 'Stratum roster differs')
    cells = {}
    lookup = {g: j for j,g in enumerate(groups)}

    def add(key, group, v):
        cells.setdefault(key,np.zeros((len(groups),5), dtype=np.int64))[group] += v

    for r in rows:
        i = r['id']
        require(r['group'] == assignment[i] and i[-1] in ('D','N'), 'Visual group or day/night differs')
        j = lookup[r['group']]
        whole = vector(r['outside'])
        per_class = [vector(c) for c in r['by_true_class_outside']]
        require(len(per_class) == len(CLASSES) and np.array_equal(sum(per_class),whole), 'Class counts do not sum')
        add('all',j,whole)
        add('time/'+i[-1],j,whole)
        for name,v in zip(CLASSES,per_class,strict=True):
            add('class/'+name,j,v)
        bands = {name:vector(v) for name,v in r['by_distance'].items()}
        if bands:
            require(np.array_equal(sum(bands.values()),whole), 'Distance counts do not sum')
        for name,v in bands.items():
            add('distance/'+name,j,v)
    result = {key:interval(v,weights) for key,v in cells.items() if v[:,0].sum()}
    result['visual_groups'] = {group:dict(zip(FIELDS,v.tolist())) for group,v in zip(groups,cells['all'],strict=True)}
    return result


def overall_miou(matrix):
    diag = np.diag(matrix)
    union = matrix.sum(0) + matrix.sum(1) - diag
    require(np.any(union > 0), 'Empty clean confusion matrix')
    return float(np.mean(diag[union > 0] / union[union > 0]))


def pair_file(seed, track, comparison):
    folder = ROOT/'reports/evidence/r2i-s1' if seed==3407 else ARCHIVE/'seed3408'
    return folder/f'{track}-{comparison}.jsonl'


def analyze():
    split = read(ROOT/'reports/evidence/r1g/split.json')
    graph = read(ROOT/'reports/evidence/r1f/scene-audit.json')
    ids = split['validation_ids']
    assignment = {i:g[0] for g in graph['groups'] for i in g if i in ids}
    groups = sorted(set(assignment.values()))
    require(len(ids)==157 and len(groups)==54 and len(assignment)==157, 'Frozen cohort differs')
    n = len(groups)
    draws = np.random.default_rng(3407).multinomial(n,np.full(n,1/n),size=10000)
    weights = np.vstack([np.ones(n,dtype=np.int64),draws])
    result, hashes = {}, {}
    for seed in SEEDS:
        summary_path = ARCHIVE/f'seed{seed}-summary.json'
        audit_path = ARCHIVE/f'audit-{seed}.json'
        summary, audit = read(summary_path), read(audit_path)
        for path in (summary_path,audit_path):
            hashes[path.relative_to(ROOT).as_posix()] = sha(path)
        require(summary['status']=='PASS' and audit['status']=='PASS' and summary['seed']==seed
                and audit['group_order']==groups and summary['test_pixels_read']==0, 'Source audit differs')
        result[seed] = {}
        for track in TRACKS:
            result[seed][track] = {}
            for comp in COMPARISONS:
                p = pair_file(seed,track,comp)
                hashes[p.relative_to(ROOT).as_posix()] = sha(p)
                rows = [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()]
                require(len(rows)==628 and {r['case'] for r in rows}==set(CONDITIONS), 'Condition cohort differs')
                cases = {}
                for condition in CONDITIONS:
                    z = stratify([r for r in rows if r['case']==condition],ids,groups,assignment,weights)
                    old = audit['tracks'][track]['contrasts'][comp][condition]
                    published = summary['tracks'][track]['comparisons'][comp][condition]
                    require([z['visual_groups'][g] for g in groups] ==
                            [dict(zip(FIELDS,c)) for c in old['group_counts']]
                            and abs(z['all']['rate_pp'] - 100*published['net_rate']['point']) < 1e-10,
                            'Archived group counts or net rate differ')
                    cases[condition] = z
                primary = audit['tracks'][track]['contrasts'][comp]['primary']
                counts = weights @ np.asarray(primary['group_counts'])
                did_draws = 100 * (weights @ np.asarray(primary['group_did'])) / counts[:,0]
                did = [float(did_draws[0]), *np.percentile(did_draws[1:],[2.5,97.5]).tolist()]
                require(np.allclose(did,100*np.asarray(primary['exploratory_corrupt_minus_clean_difference']),
                                    rtol=0,atol=1e-10), 'Archived DID differs')
                matrices = audit['tracks'][track]['group_clean_confusions']
                candidate, reference = comp.split('-minus-')
                clean_delta = 100 * (overall_miou(np.sum(matrices[candidate],axis=0))
                                     - overall_miou(np.sum(matrices[reference],axis=0)))
                require(np.isclose(clean_delta,100*summary['tracks'][track]['comparisons'][comp]
                                   ['clean_noninferiority']['overall']['point'],rtol=0,atol=1e-10),
                        'Clean confusion matrices differ from archived mIoU')
                c = cases['primary']['all']['rate_pp']
                result[seed][track][comp] = {'conditions':cases, 'center_did_pp':did,
                    'center_same_roi_clean_delta_pp':c-did[0],
                    'clean_miou_delta_pp':clean_delta}
    for path in (ROOT/'reports/evidence/r1g/split.json',ROOT/'reports/evidence/r1f/scene-audit.json'):
        hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    return {'status':'PASS','group_order':groups,'draws':10000,'seed':3407,'source_sha256':hashes,'results':result}


def checkpoint_scores(domains):
    import torch

    torch.set_num_threads(4)
    output = {}
    for seed, root in zip(SEEDS,domains,strict=True):
        summary = read(ARCHIVE/f'seed{seed}-summary.json')
        output[seed] = {}
        for track in TRACKS:
            for arm in ('gated','control'):
                path = root/f'{track}-{arm}'/'local.pt'
                expected = summary['tracks'][track]['cells'][arm]
                require(sha(path)==expected['checkpoint_sha256'], 'Checkpoint authenticity check failed')
                state = torch.load(path,map_location='cpu',weights_only=False)
                require(state['signature']==expected['signature'], 'Checkpoint signature differs')
                gate = {name:t for name,t in state['model'].items() if 'r2i_gate.' in name}
                flat = [float(v) for name in gate for v in state['model'][name].flatten().tolist()]
                require(len(flat)==16 and np.isfinite(flat).all(), 'Gate coefficients differ')
                weights = np.asarray([t.flatten().tolist() for k,t in gate.items() if k.endswith('.weight')])
                output[seed][f'{track}/{arm}'] = {
                    'positive_by_feature': (weights>0).sum(0).tolist(),
                    'negative_by_feature': (weights<0).sum(0).tolist(),
                    'absolute_range_by_feature': [np.abs(weights).min(0).tolist(), np.abs(weights).max(0).tolist()],
                    'checkpoint_sha256':expected['checkpoint_sha256']}
                del state
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint-roots',type=Path,nargs=2)
    args=p.parse_args()
    result=analyze()
    if args.checkpoint_roots:
        result['gate_checkpoints']=checkpoint_scores(args.checkpoint_roots)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('PASS: frozen strata, published group totals and post-hoc summaries')
