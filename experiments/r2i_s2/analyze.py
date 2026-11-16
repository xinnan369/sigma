"""Independently reconcile final checkpoints, validation pixels and paired outcomes."""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from experiments.r1.assets import sha256
from experiments.r1c.data import write_json
from experiments.r1g.partition import EVIDENCE, ROOT, read, references, require, selected_files
from experiments.r1g.train import write_rows
from experiments.r2a.analyze import counts_vector, interval, ratio
from experiments.r2a.metrics import confusion, digest, distance_bands, pair, scores
from experiments.r2a.run import records
from experiments.r2d.analysis import iou
from experiments.r2i.augment import position, rng
from experiments.r2i_s2.driver import verify_history
from experiments.r2i_s2.protocol import STAGE, require_seed


def clean_contrast(rows, ref, groups, weights, foreground):
    def sampled(index):
        matrix = np.zeros((len(groups), 81))
        for row in index:
            matrix[groups.index(row['group'])] += np.ravel(row['matrix'])
        values = iou((weights @ matrix).reshape(-1, 9, 9))
        if foreground:
            values = values[:, 1:]
        return ratio(np.nansum(values, axis=1), np.isfinite(values).sum(1))
    delta = sampled(rows) - sampled(ref)
    result = interval(delta[0], delta[1:])
    result['noninferiority'] = result['ci95'][0] > -.005
    return result


def analyze(root, output, seed):
    require_seed(seed)
    cfg = read(ROOT / 'configs/R2i_protocol.json')
    r2a = read(ROOT / 'configs/R2a_exp0.json')
    split = read(EVIDENCE / 'split.json')
    graph, _ = references()
    ids = split['validation_ids']
    groups_by_id = {i: g[0] for g in graph['groups'] for i in g}
    groups = sorted({groups_by_id[i] for i in ids})
    require(len(ids) == 157 and len(groups) == 54, 'Holdout cohort differs')
    expected_assets = {r['path']: r['sha256'] for r in selected_files(split)}
    validation = {}
    for sample in ids:
        path = f'Label/{sample}.png'
        require(sha256(root/path) == expected_assets[path], 'Validation label digest differs')
        gt = cv2.imread(str(root/path), cv2.IMREAD_GRAYSCALE)
        y,x = position(rng(cfg['random_occlusion_seed'], 0, sample), *gt.shape, 64)
        masks = {}
        for case in cfg['validation']:
            mask = np.zeros_like(gt, dtype=bool)
            if case == 'random_occlusion':
                mask[y:y+64, x:x+64] = True
            elif case != 'clean':
                y0,x0,y1,x1 = r2a['positions_y0_x0_y1_x1']['center']
                mask[y0:y1,x0:x1] = True
            masks[case] = (mask, distance_bands(mask, r2a['distance_upper_bounds_px']))
        validation[sample] = gt, masks
    count_draws = np.random.default_rng(cfg['bootstrap_seed']).multinomial(
        len(groups), np.full(len(groups), 1 / len(groups)), size=cfg['bootstrap_draws'])
    weights = np.vstack((np.ones((1,len(groups))), count_draws))
    result = {}
    common = set()
    for track in cfg['tracks']:
        cells, rows, predictions, schedules = {}, {}, {}, []
        for arm in cfg['arms']:
            folder = output / f'{track}-{arm}'
            cell = read(folder / 'complete.json')
            sig = cell['signature']
            require(cell['status']=='PASS' and cell['updates']==785 and sha256(folder/'local.pt')==cell['checkpoint_sha256']
                    and sig['seed']==seed and sig['stage_sha256']==sha256(STAGE) and sig['track']==track and sig['arm']==arm
                    and sig['protocol_sha256']==sha256(ROOT/'configs/R2i_protocol.json')
                    and all(sha256(ROOT/name)==value for name,value in sig['sources'].items()),
                    'Final checkpoint or source fingerprint differs')
            history = records(folder / 'training.jsonl')
            verify_history(history, split['train_ids'], seed, complete=True)
            require(all(x['gradient_tensor_count']==cell['coverage'] and np.isfinite(x['gradient_l2'])
                        for x in history), 'Gradient audit failed')
            schedules.append([(x['sample_ids'], x['augmentation'], x['valid_pixel_counts']) for x in history])
            common.add(cell['common_initial_sha256'])
            cells[arm] = cell
            rows[arm] = {(r['id'], r['case']): r for r in records(folder/'validation.jsonl')}
            require(set(rows[arm]) == {(i,c) for i in ids for c in cfg['validation']}, 'Evaluation coverage differs')
            with np.load(folder/'predictions.npz') as archive:
                predictions[arm] = {k: archive['/'.join(k)] for k in rows[arm]}
            for (i,c), row in rows[arm].items():
                gt, masks = validation[i]
                pred = predictions[arm][i,c]
                require(row['group']==groups_by_id[i] and row['prediction_sha256']==digest(pred)
                        and row['roi_sha256']==digest(masks[c][0])
                        and row['matrix']==confusion(gt,pred).tolist(), 'Prediction audit failed')
        require(schedules[0]==schedules[1]==schedules[2], 'Schedules or augmentation differ')
        performance, comparisons = {}, {}
        for arm in cfg['arms']:
            performance[arm] = {c:scores(np.sum([rows[arm][i,c]['matrix'] for i in ids],axis=0))
                                for c in cfg['validation']}
        for arm, ref in [('gated','baseline'),('control','baseline'),('gated','control')]:
            out, paired = {}, []
            for c in cfg['validation']:
                table = np.zeros((len(groups),5))
                for i in ids:
                    gt, masks = validation[i]
                    a,b = predictions[ref][i,c],predictions[arm][i,c]
                    row = pair(a,b,gt,*masks[c],scores(confusion(gt,a)))
                    group = groups_by_id[i]
                    table[groups.index(group)] += counts_vector(row['outside'])
                    paired.append({'id':i,'case':c,'group':group,**row})
                sampled = weights @ table
                rate = ratio(sampled[:,3],sampled[:,0])
                out[c] = {'counts': table.sum(0).astype(int).tolist(),
                          'net_rate': interval(rate[0],rate[1:])}
            out['clean_noninferiority'] = {
                k:clean_contrast([rows[arm][i,'clean'] for i in ids],[rows[ref][i,'clean'] for i in ids],
                                 groups,weights,k=='foreground') for k in ('overall','foreground')}
            comparisons[f'{arm}-minus-{ref}'] = out
            write_rows(output/f'{track}-{arm}-minus-{ref}.jsonl',paired)
        result[track] = {'cells':cells,'performance':performance,'comparisons':comparisons}
    require(len(common)==1, 'Common initialization differs')
    write_json(output/'summary.json',{'status':'PASS','seed':seed,'visual_groups':groups,
                                     'draws':len(count_draws),'tracks':result,'test_pixels_read':0})


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=(3408,3409),required=True)
    args=p.parse_args()
    analyze(args.root,args.output / f'seed{args.seed}',args.seed)
