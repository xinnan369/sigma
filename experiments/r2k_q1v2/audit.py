"""Check the local freeze and reproduce calibration metrics without the model runner."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def check(ok,message):
    if not ok:
        raise ValueError(message)


def audit(repo, local, evidence):
    path=repo/'configs/R2k_Q1v2_protocol.json'
    protocol=read(path)
    old=read(repo/protocol['parent'])
    p=read(local/'frozen-parameters.json')
    report=read(evidence/'fit-calibration.json')
    check(sha(path)==report['protocol_sha256']==p['protocol_sha256']
          and sha(repo/protocol['parent'])==protocol['parent_sha256']
          and sha(local/'frozen-parameters.json')==report['frozen_parameters_sha256']
          and sha(local/'calibration.npz')==report['calibration_arrays_sha256']
          and p['source_sha256']==report['source_sha256']
          and all(sha(repo/name)==v for name,v in p['source_sha256'].items()),'Evidence hash differs')
    groups=[set(old['partitions'][k]['groups']) for k in ('fit','calibration','holdout')]
    ids=[set(old['partitions'][k]['images']) for k in ('fit','calibration','holdout')]
    check(all(len(set.union(*x))==sum(map(len,x)) for x in (groups,ids)), 'Partition leaks')
    families=[('healthy',0),('occlusion',0),('gaussian',0),('gaussian',1)]
    for role in ('fit','calibration'):
        rows=report[role+'_index']
        roster=old['partitions'][role]['images']
        check([(r['id'],r['family'],r['tier']) for r in rows]==[(i,f,t) for i in roster for f,t in families]
              and all(roster[r['id']]==r['group'] for r in rows), 'Exposure roles or family differ')
    check(report['holdout_image_reads']==report['blur_views']==0,'Unexpected evaluation exposure')
    src=ast.parse((repo/'experiments/r2k_q1v2/run.py').read_text(encoding='utf-8'))
    predict=next(n for n in src.body if isinstance(n,ast.FunctionDef) and n.name=='predict')
    check(tuple(x.arg for x in predict.args.args)==('image','arm','cfg','head','calibration')
          and {n.id for n in ast.walk(predict) if isinstance(n,ast.Name)}
          =={'f','identifiable','features','image','arm','cfg','head','calibration','probabilities','logits'},
          'Inference interface differs')
    rows=report['calibration_index']
    with np.load(local/'calibration.npz',allow_pickle=False) as a:
        y,h=a['target'],a['healthy']
        check(len(y)==sum(r['points'] for r in rows) and int(y.sum())==sum(r['positive'] for r in rows)
              and not y[h].any(), 'Calibration mask/accounting differs')
        # The RGB score is constant across paired view types for each original.
        rgb=a['rgb'].reshape(len(rows)//4,4,4800)
        check(np.all(rgb==rgb[:,:1]), 'RGB paired inputs differed')
        for arm,data in p['heads'].items():
            v=data['weights']
            check(len(v)==7 and np.isfinite(v).all() and len(data['scale'])==6
                  and np.all(np.asarray(data['scale'])>0), 'Head dimension or scale differs')
            rec=p['calibration'][arm]
            prob={}
            for k,(w,b) in rec['choices'].items():
                prob[k]=np.rint(255/(1+np.exp(-(w*a[arm]+b)))).astype(np.uint8)
            errors={k:np.mean((x/255-y)**2) for k,x in prob.items()}
            winner=min(errors,key=errors.get)
            x=prob[winner]
            t=rec['threshold_bin']
            healthy=x[h].astype(int)
            check(winner==rec['method'] and np.array_equal(x,a['bins_'+arm])
                  and np.mean(healthy>=t)<=.05 and (t==0 or np.mean(healthy>=t-1)>.05),
                  'Calibration method or threshold differs')
            table=np.zeros((2,256),np.int64)
            np.add.at(table,(y.astype(int),x),1)
            pos=table[1,::-1]
            tp=np.cumsum(pos)
            ns=np.cumsum(table[:,::-1].sum(0))
            ap=np.sum(np.divide(tp,ns,out=np.zeros(256),where=ns>0)*pos)/sum(pos)
            check(np.isclose(ap,report['metrics'][arm]['ap'],atol=1e-12,rtol=0)
                  and np.isclose(errors[winner],report['metrics'][arm]['brier'],atol=1e-12,rtol=0)
                  and report['metrics'][arm]['evaluated_points']==len(y),'Metric or coverage differs')
    return {'status':'PASS','scope':'train-side engineering only','images_fit':len(ids[0]),
            'images_calibration':len(ids[1]),'groups_fit':len(groups[0]),'groups_calibration':len(groups[1]),
            'holdout_or_blur_reads':0,'paired_rgb_verified':True,'checked_heads':len(p['heads']),
            'calibration_ap_brier_and_threshold_recomputed':True,'frozen_parameters_sha256':report['frozen_parameters_sha256']}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('repo','local','evidence','output'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    r=audit(a.repo,a.local,a.evidence)
    a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    print(r['status'],flush=True)
