"""Independent read-only validation of the frozen protocol and saved outputs."""
import argparse
import ast
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def check(ok,msg):
    if not ok:
        raise ValueError(msg)


def hash_rng(*keys):
    d=hashlib.sha256('/'.join(str(k) for k in keys).encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(d[:16],'big')))


def ap_and_brier(hist):
    # Weighted precision at the end of each tie block, weighted by recall steps.
    positive=hist[...,1,::-1]
    negative=hist[...,0,::-1]
    both=positive+negative
    accumulated=np.add.accumulate(both,axis=-1)
    tp=np.add.accumulate(positive,axis=-1)
    with np.errstate(divide='ignore',invalid='ignore'):
        precision=np.divide(tp,accumulated,out=np.zeros_like(tp),where=accumulated>0)
        ap=np.einsum('ij,ij->i',precision,positive)/positive.sum(-1)
        q=np.linspace(1,0,256)
        brier=(negative@(q*q)+positive@((1-q)**2))/both.sum(-1)
    return ap,brier


def verify_calibration(local, fit, cfg):
    z=local/'calibration.npz'
    check(sha(z)==fit['calibration_input_sha256'],'Calibration arrays differ')
    result={}
    with np.load(z,allow_pickle=False) as archive:
        y=archive['target']
        h=archive['healthy']
        check(len(y)==sum(r['grid_points'] for r in fit['calibration_log'])
              and int(y.sum())==sum(r['positives'] for r in fit['calibration_log'])
              and not y[h].any(),'Calibration label accounting differs')
        for a,c in fit['calibration'].items():
            candidates={}
            for name,(s,b) in c['candidates'].items():
                x=archive[a]*s+b
                p=np.rint(255/(1+np.exp(-x))).astype(np.uint8)
                candidates[name]=(float(np.mean((p/255-y)**2)),p)
            chosen=min(candidates,key=lambda k:candidates[k][0])
            check(chosen==c['chosen'] and all(np.isclose(v[0],c['candidate_brier'][k],atol=1e-13,rtol=0)
                  for k,v in candidates.items()),'Calibration selection differs')
            p=candidates[chosen][1][h].astype(np.int64)
            threshold=c['threshold_bin']
            false_rate=float(np.mean(p>=threshold))
            check(false_rate<=cfg['health_target_fpr'] and
                  (threshold==0 or np.mean(p>=threshold-1)>cfg['health_target_fpr'])
                  and np.isclose(false_rate,c['calibration_healthy_fpr'],atol=1e-13,rtol=0),
                  'Calibration threshold differs')
            result[a]=false_rate
    return result


def audit(repo,local,evidence):
    start=time.perf_counter()
    protocol=repo/'configs/R2k_Q1_protocol.json'
    cfg=read(protocol)
    split=read(repo/'reports/evidence/r1g/split.json')
    graph=read(repo/'reports/evidence/r1f/scene-audit.json')
    graph_lookup={i:g[0] for g in graph['groups'] for i in g}
    roles=cfg['partitions']
    names=set()
    groups=set()
    for role,section in roles.items():
        selected=set(section['images'])
        group=set(section['groups'])
        check(not names&selected and not groups&group and group==set(section['images'].values())
              and all(graph_lookup[i]==g for i,g in section['images'].items()),f'{role} leaks groups')
        names|=selected
        groups|=group
    check(set(split['train_ids'])==names|{i+'_flip' for i in names} and not names&set(split['validation_ids'])
          and not groups&{graph_lookup[i] for i in split['validation_ids']},'Parent train-only membership differs')
    freeze=read(evidence/'calibration-freeze.json')
    fit=read(local/'frozen-fit.json')
    report=read(evidence/'holdout-results.json')
    check(fit['signature']['protocol_sha256']==sha(protocol) and sha(local/'frozen-fit.json')==freeze['fit_sha256']
          ==report['fingerprints']['fit'] and sha(local/'holdout.npz')==report['fingerprints']['predictions']
          and sha(local/'holdout-index.json')==report['fingerprints']['index']
          and all(sha(repo/'experiments/r2k_q1'/p)==d for p,d in fit['signature']['source_sha256'].items()),
          'Evidence hash differs')
    for log,role in ((fit['fit_log']['rows'],'fit'),(fit['calibration_log'],'calibration')):
        observed={(r['id'],r['family'],r['tier']) for r in log}
        expected={(i,c,t) for i in roles[role]['images'] for c,t in
                  (('healthy',0),('occlusion',0),('gaussian',0),('gaussian',1))}
        check(len(log)==len(expected) and observed==expected and all(
            r['group']==roles[role]['images'][r['id']] for r in log),'Fit/calibration family or roster differs')
    check(all(len(h['weights'])==7 and len(h['mean'])==6 for h in fit['heads'].values()),
          'Parameter budgets differ')
    # Audit that the inference function exposes a single image and no corruption metadata.
    source=ast.parse((repo/'experiments/r2k_q1/model.py').read_text(encoding='utf-8'))
    node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='predict')
    check([a.arg for a in node.args.args]==['image','head','calibration']
          and {n.id for n in ast.walk(node) if isinstance(n,ast.Name)}
          =={'probabilities','logits','describe','image','head','calibration'},'Inference contract differs')
    calibration=verify_calibration(local,fit,cfg)
    rows=read(local/'holdout-index.json')
    holdout=roles['holdout']
    expected=[(i,c,t) for i in holdout['images'] for c,t in (('healthy',0),('blur',0),('blur',1))]
    check([(r['id'],r['family'],r['tier']) for r in rows]==expected,'Holdout roster or family differs')
    group_names=holdout['groups']
    arms=('thermal','rgb','shuffle')
    grouped={key:{a:np.zeros((len(group_names),2,256),np.int64) for a in arms}
             for key in ('healthy','blur')}
    healthy_rgb={}
    with np.load(local/'holdout.npz',allow_pickle=False) as arr:
        check(all(arr[a].shape==(len(rows),4800) and arr[a].dtype==np.uint8 for a in arms),
              'Probability grid dimensions or range differ')
        for j,r in enumerate(rows):
            i=r['id']
            check(r['group']==holdout['images'][i],'Holdout group differs')
            mask=np.zeros((60,80),bool)
            bins=arr['rgb'][j]
            if r['family']=='blur':
                gen=hash_rng(3407,i,'blur',r['tier'])
                y,x=int(gen.integers(12,373)),int(gen.integers(12,533))
                yy,xx=np.ogrid[4:480:8,4:640:8]
                mask=(yy>=y)&(yy<y+96)&(xx>=x)&(xx<x+96)
                check(np.array_equal(bins,healthy_rgb[i]),'RGB saw synthetic corruption metadata')
            else:
                healthy_rgb[i]=bins
            target=mask.ravel()
            check(np.array_equal(target,arr['target'][j]),'Frozen ROI differs')
            g=group_names.index(r['group'])
            for a in arms:
                for label in (0,1):
                    grouped[r['family']][a][g,label]+=np.bincount(arr[a][j][target==label],minlength=256)
    n=len(group_names)
    w=np.vstack((np.ones(n),hash_rng(3407,'bootstrap').multinomial(n,[1/n]*n,size=10000)))
    thresholds={a:fit['calibration'][a]['threshold_bin'] for a in arms}
    cached={}
    for a in arms:
        blur=(w@grouped['blur'][a].reshape(n,512)).reshape(-1,2,256)
        healthy=(w@grouped['healthy'][a].reshape(n,512)).reshape(-1,2,256)
        ap,brier=ap_and_brier(blur)
        recall=blur[:,1,thresholds[a]:].sum(-1)/blur[:,1].sum(-1)
        fpr=healthy[:,0,thresholds[a]:].sum(-1)/healthy[:,0].sum(-1)
        cached[a]={'ap':ap,'brier':brier,'recall':recall,'fpr':fpr}

    def match(x, original):
        check(np.allclose([x[0],*np.percentile(x[1:],[2.5,97.5])],
                          [original['point'],*original['ci95']],atol=5e-13,rtol=0),'Bootstrap differs')

    for a,m in cached.items():
        for k,x in m.items():
            key='healthy' if k=='fpr' else 'blur'
            match(x,report['strata'][key][a][k])
            if a=='thermal':
                match(x-cached['rgb'][k],report['strata'][key]['thermal_minus_rgb'][k])
    return {'status':'PASS','read_only':True,'fit_calibration_family_isolation':True,
            'train_heldout_group_isolation':True,'parameters_each':7,'verified_holdout_rows':len(rows),
            'calibration_fpr':calibration,'bootstrap_replicates_verified':10000,
            'metrics':['AUPRC','Brier','recall','healthy FPR','paired differences'],
            'seconds':time.perf_counter()-start,
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'exit':0}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ('repo','local','evidence','output'):
        p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args()
    result=audit(a.repo,a.local,a.evidence)
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(result['status'],flush=True)
