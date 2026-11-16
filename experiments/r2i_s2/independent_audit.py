"""Read-only independent R2i-S1 audit. No imports from the formal driver or metrics."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CASES = ('clean', 'primary', 'random_occlusion', 'gaussian')
ARMS = ('baseline', 'gated', 'control')
FIELDS = ('pixels', 'new', 'corrected', 'net', 'disagree')


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def rows(p):
    return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()]


def check(flag, message):
    if not flag:
        raise ValueError(message)


def file_hash(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def pixel_hash(x):
    return hashlib.sha256(x.tobytes()).hexdigest()


def jac(matrix):
    diag = np.diagonal(matrix, axis1=-2, axis2=-1)
    union = matrix.sum(-2) + matrix.sum(-1) - diag
    return np.divide(diag, union, out=np.full(diag.shape, np.nan), where=union > 0)


def roi(sample, case, shape):
    if case == 'random_occlusion':
        key = hashlib.sha256(f'3407/0/{sample}'.encode()).digest()
        g = np.random.Generator(np.random.PCG64(int.from_bytes(key[:16], 'big')))
        y, x = int(g.integers(shape[0]-63)), int(g.integers(shape[1]-63))
    else:
        y, x = 208, 288
    mask = np.zeros(shape, bool)
    if case != 'clean':
        mask[y:y+64, x:x+64] = True
    return mask


def paired(a, b, target, region):
    ref, cand = a != target, b != target
    add = np.count_nonzero(region & ~ref & cand)
    corrected = np.count_nonzero(region & ref & ~cand)
    return np.array([np.count_nonzero(region), add, corrected, add-corrected,
                     np.count_nonzero(region & (a != b))], dtype=np.int64)


def interval(v):
    return [float(v[0]), *np.percentile(v[1:], [2.5, 97.5]).tolist()]


def reconcile_metric(result, original):
    check(np.allclose(result, [original['point'], *original['ci95']], atol=1e-13, rtol=0),
          'Published interval differs from independent calculation')


def audit_train(run, train, cell, arm, track, seed):
    history = rows(run/'training.jsonl')
    gen = torch.Generator().manual_seed(seed)
    check(len(history) == 785 and cell['status'] == 'PASS', 'Update budget differs')
    coverage = 668 if arm == 'baseline' else 676
    epoch_summary = []
    for epoch in range(1, 6):
        order = torch.randperm(len(train), generator=gen).tolist()
        chunk = history[(epoch-1)*157:epoch*157]
        pixels = 0
        for j,r in enumerate(chunk):
            end = min((j+1)*8, 1254)
            ids = [train[i] for i in order[j*8:end]]
            n = len(ids)
            valid = np.asarray(r['valid_pixel_counts'])
            loss = np.asarray(r['losses'])
            check(r['sample_ids']==ids and r['cursor']==end and r['epoch']==epoch
                  and r['update']==(epoch-1)*157+j+1 and r['microbatches']==n
                  and len(valid)==len(loss)==len(r['augmentation'])==n
                  and np.all(valid>0) and np.isfinite(loss).all()
                  and np.isclose(loss@valid,r['pixel_loss_sum'])
                  and np.isclose((loss@valid)/sum(valid),r['mean_pixel_loss'])
                  and sum(valid)==sum(r['class_pixels'])
                  and r['gradient_tensor_count']==coverage and np.isfinite(r['gradient_l2'])
                  and r['lr']==6e-5, 'Training accounting differs')
            for sample, box in zip(ids,r['augmentation'],strict=True):
                key = hashlib.sha256(f'{seed}/{epoch}/{sample}'.encode()).digest()
                draw = np.random.Generator(np.random.PCG64(int.from_bytes(key[:16],'big')))
                chosen = draw.random() < .5 and track == 'augmented'
                check(chosen==(box is not None), 'Augmentation Bernoulli differs')
                if chosen:
                    check(box==[int(draw.integers(417)),int(draw.integers(577)),64],
                          'Augmentation position differs')
            pixels += int(sum(valid))
        epoch_summary.append({'epoch':epoch,'updates':len(chunk),'records':sum(r['microbatches'] for r in chunk),
                              'tail_pixels':chunk[-1]['valid_pixel_counts'],'valid_pixels':pixels})
    p=run/'local.pt'
    check(file_hash(p)==cell['checkpoint_sha256'], 'Checkpoint SHA256 differs')
    ck=torch.load(p,map_location='cpu',weights_only=False)
    state=ck['optimizer']['state']
    check(ck['progress']['updates']==785 and ck['scheduler']['last_epoch']==785
          and len(state)==coverage==cell['coverage'] and all(v['step'].item()==785 for v in state.values())
          and ck['sampler']['epoch']==5 and ck['sampler']['cursor']==1254
          and ck['sampler']['order']==order and torch.equal(ck['sampler']['generator'],gen.get_state())
          and ck['signature']==cell['signature'] and set(ck['rng'])=={'python','numpy','torch','cuda'}
          and bool(ck['rng']['cuda']), 'Saved optimizer, sampler, scheduler or RNG differs')
    for name, value in cell['signature']['sources'].items():
        check(file_hash(ROOT/name)==value, f'Execution source differs: {name}')
    check(file_hash(ROOT/'configs/R2i_protocol.json')==cell['signature']['protocol_sha256'], 'Protocol differs')
    model=hashlib.sha256()
    for name, t in ck['model'].items():
        check(not t.is_floating_point() or torch.isfinite(t).all().item(), 'Nonfinite model checkpoint')
        model.update(name.encode())
        model.update(t.contiguous().numpy().tobytes())
    check(model.hexdigest()==cell['model_sha256'], 'Model digest differs')
    return epoch_summary


def main(run, data, output, seed):
    torch.set_num_threads(4)
    frozen=read(ROOT/'reports/evidence/r1g/split.json')
    graph=read(ROOT/'reports/evidence/r1f/scene-audit.json')
    ids=frozen['validation_ids']
    assignment={i:g[0] for g in graph['groups'] for i in g}
    groups=sorted({assignment[i] for i in ids})
    check(len(ids)==157 and len(groups)==54, 'Frozen cohort differs')
    group_idx={i:groups.index(assignment[i]) for i in ids}
    manifests={r['path']:r['sha256'] for r in read(ROOT/'reports/evidence/r1c/manifest.json')['files']}
    labels={}
    for i in ids:
        key=f'Label/{i}.png'
        check(file_hash(data/key)==manifests[key],'Label bytes differ')
        labels[i]=cv2.imread(str(data/key),cv2.IMREAD_GRAYSCALE)
    masks={(i,c):roi(i,c,labels[i].shape) for i in ids for c in CASES}
    weights=np.vstack((np.ones((1,54)),np.random.Generator(np.random.PCG64(3407)).multinomial(
        54,np.full(54,1/54),size=10000)))
    reference=read(run/'summary.json')
    check(reference['status']=='PASS' and reference['seed']==seed and reference['test_pixels_read']==0,
          'Run summary differs')
    output_values={}
    for track in ('clean','augmented'):
        track_ref=reference['tracks'][track]
        by_arm, prediction, train_audit = {}, {}, {}
        for arm in ARMS:
            folder=run/f'{track}-{arm}'
            cell=read(folder/'complete.json')
            train_audit[arm]=audit_train(folder,frozen['train_ids'],cell,arm,track,seed)
            all_rows=rows(folder/'validation.jsonl')
            check(len(all_rows)==628 and [(r['id'],r['case']) for r in all_rows]==[(i,c) for i in ids for c in CASES],
                  'Prediction records are incomplete or misordered')
            table={c:np.zeros((54,9,9),dtype=np.int64) for c in CASES}
            preds={}
            with np.load(folder/'predictions.npz') as z:
                check(set(z.files)=={f'{i}/{c}' for i in ids for c in CASES},'Prediction archive keys differ')
                for r in all_rows:
                    i,c=r['id'],r['case']
                    g=labels[i]
                    p=z[f'{i}/{c}']
                    valid=g<9
                    h=np.bincount(9*g[valid].astype(int)+p[valid],minlength=81).reshape(9,9)
                    check(p.shape==g.shape and p.max()<9 and pixel_hash(p)==r['prediction_sha256']
                          and pixel_hash(masks[i,c])==r['roi_sha256'] and assignment[i]==r['group']
                          and np.array_equal(h,r['matrix']),'Per-image prediction audit failed')
                    table[c][group_idx[i]]+=h
                    preds[i,c]=p
            for c in CASES:
                j=jac(table[c].sum(0))
                published=track_ref['performance'][arm][c]
                check(np.allclose(j,published['iou_0_to_8']) and np.isclose(np.nanmean(j),published['miou'])
                      and np.isclose(np.nanmean(j[1:]),published['foreground_miou']),'IoU differs')
            by_arm[arm]=table
            prediction[arm]=preds
        contrasts={}
        for arm,ref in (('gated','baseline'),('gated','control'),('control','baseline')):
            contrast=f'{arm}-minus-{ref}'
            prior=track_ref['comparisons'][contrast]
            archived=rows(run/f'{track}-{contrast}.jsonl')
            check(len(archived)==628,'Paired archive length differs')
            items={(r['id'],r['case']):r for r in archived}
            check(set(items)==set(masks),'Paired archive keys differ')
            result={}
            for c in CASES:
                groups_count=np.zeros((54,5),dtype=np.int64)
                changes=np.zeros(54,dtype=np.int64)
                within={a:np.zeros((54,5),dtype=np.int64) for a in ARMS} if ref=='baseline' else {}
                for i in ids:
                    gt=labels[i]
                    region=(gt<9)&~masks[i,c]
                    a,b=prediction[ref][i,c],prediction[arm][i,c]
                    check(items[i,c]['group']==assignment[i],'Paired group differs')
                    counts=paired(a,b,gt,region)
                    check(counts.tolist()==[items[i,c]['outside'][k] for k in FIELDS],'Paired counts differ')
                    group=group_idx[i]
                    groups_count[group]+=counts
                    changes[group]+=counts[3]-paired(prediction[ref][i,'clean'],
                                                     prediction[arm][i,'clean'],gt,region)[3]
                    for name in within:
                        within[name][group]+=paired(prediction[name][i,'clean'],prediction[name][i,c],gt,region)
                draws=weights@groups_count
                total=groups_count.sum(0)
                est=interval(draws[:,3]/draws[:,0])
                check(total.tolist()==prior[c]['counts'],'Summary pixel counts differ')
                reconcile_metric(est,prior[c]['net_rate'])
                result[c]={'counts':dict(zip(FIELDS,total.tolist())),
                           'rates':dict(zip(FIELDS[1:],(total[1:]/total[0]).tolist())),
                           'net_rate_point_low_high':est,
                           'group_counts':groups_count.tolist(), 'group_did':changes.tolist(),
                           'exploratory_corrupt_minus_clean_difference':interval((weights@changes)/draws[:,0])}
                if within:
                    result[c]['exploratory_within_model']={name:dict(zip(FIELDS,m.sum(0).tolist()))
                                                            for name,m in within.items()}
            cand=jac((weights@by_arm[arm]['clean'].reshape(54,81)).reshape(-1,9,9))
            base=jac((weights@by_arm[ref]['clean'].reshape(54,81)).reshape(-1,9,9))
            for k,start in (('overall',0),('foreground',1)):
                est=interval(np.nanmean(cand[:,start:],axis=1)-np.nanmean(base[:,start:],axis=1))
                old=prior['clean_noninferiority'][k]
                reconcile_metric(est,old)
                check((est[1]>-.005)==old['noninferiority'],'Clean noninferiority differs')
            contrasts[contrast]=result
        output_values[track]={'training':train_audit,'contrasts':contrasts,
            'group_clean_confusions':{a:by_arm[a]['clean'].tolist() for a in ARMS}}
        print(track, 'PASS', flush=True)
    output.write_text(json.dumps({'status':'PASS','seed':seed,'groups':54,'group_order':groups,'draws':10000,
                      'tracks':output_values},indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for option in ('run','data','output'):
        parser.add_argument('--'+option,type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=(3407,3408),required=True)
    args=parser.parse_args()
    main(args.run,args.data,args.output,args.seed)
