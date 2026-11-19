"""Read-only candidate issuance. Never installs a receipt or starts training."""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from .authorization import REQUIRED, require_authorization
from .checkpoint import digest, hashfile, write_json
from .coverage import audit, environment
from .data import DevelopmentData, SPLIT_SHA256
from .model import SEEDS
from .train import ROOT, batches, reproducible, runtime, signature
from .upstream import load_sigma

BASE='e1d0d462f84791f0b888fdae8dab25629b2aee9f'
D0=ROOT/'reports/evidence/r3-d0/split.json'
PROTOCOL='docs/R3_D1_DETECTION_PROTOCOL.md'
EXPECTED={'fit':(9926,28547,28540,7,2,[90027,150737,27]),
          'validation':(2087,5536,5530,6,0,[5272,41060,0])}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def data_audit(root,out):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    data=DevelopmentData(root,D0)
    rosters={role:data.ids(role) for role in EXPECTED}
    require(not set(rosters['fit'])&set(rosters['validation']),'Overlapping development roles')
    evidence={}
    for role,ids in rosters.items():
        require(len(set(ids))==len(ids)==EXPECTED[role][0],f'Unexpected {role} roster')
        valid_boxes=0
        for ident in ids:
            rgb,thermal,boxes,_=data.read_raw(ident,role)
            require(rgb.shape==(1024,1280,3) and thermal.shape==(1024,1280)
                    and rgb.dtype==thermal.dtype==np.uint8,f'Invalid paired image: {ident}')
            valid_boxes+=len(boxes)
        require(valid_boxes==EXPECTED[role][1],f'Unexpected {role} GT count')
        evidence[role]=dict(pairs=len(ids),verified_jpeg_files=2*len(ids),verified_xml_files=len(ids),
                            valid_gt=valid_boxes,roster_sha256=digest(ids))
        print(role,evidence[role],flush=True)
    audit(data,out)
    seen=read(out/'gt-coverage.json')
    verify_coverage(seen)
    write_json(out/'data-integrity.json',dict(status='PASS',split_sha256=hashfile(D0),
               official_test_reads=0,quarantine_reads=0,real_llvip_optimizer_updates=0,
               roles=evidence,environment=environment(),
               coverage_sha256=hashfile(out/'gt-coverage.json'),
               per_image_sha256=hashfile(out/'gt-per-image.jsonl')))
    print('PASS development-only bytes, paired decode, XML and FCOS matches',flush=True)


def verify_coverage(doc):
    require(doc.get('status')=='MEASURED_UNCHANGED_PROTOCOL'
            and doc.get('split_sha256')==SPLIT_SHA256 and doc.get('official_test_reads')==0
            and doc.get('image_pixel_reads')==0 and set(doc['roles'])==set(EXPECTED),
            'Invalid coverage scope')
    for role,t in EXPECTED.items():
        a=doc['roles'][role]['totals']
        got=tuple(a[k] for k in ('images','gt','matched','unmatched','zero_positive_images'))
        require(got+([a[f'positive_stride_{s}'] for s in (8,16,32)],)==t,
                f'FCOS coverage changed: {role}')


def verify_data(out):
    p=out/'data-integrity.json'
    d=read(p)
    require(d.get('status')=='PASS' and d.get('split_sha256')==SPLIT_SHA256
            and all(d.get(k)==0 for k in ('official_test_reads','quarantine_reads',
                                         'real_llvip_optimizer_updates'))
            and set(d['roles'])==set(EXPECTED),'Invalid data evidence scope')
    split=read(D0)
    for role,(n,gt,*_) in EXPECTED.items():
        require(d['roles'][role]==dict(pairs=n,verified_jpeg_files=2*n,verified_xml_files=n,
                valid_gt=gt,roster_sha256=digest(split[role])),f'Invalid evidence for {role}')
    for key,name in [('coverage_sha256','gt-coverage.json'),('per_image_sha256','gt-per-image.jsonl')]:
        require(d[key]==hashfile(out/name),f'Evidence digest differs: {name}')
    verify_coverage(read(out/'gt-coverage.json'))
    with (out/'gt-per-image.jsonl').open(encoding='utf-8') as stream:
        for role,expected in EXPECTED.items():
            total=[0]*8
            for ident in split[role]:
                row=json.loads(next(stream))
                require((row['id'],row['role'])==(ident,role),'FCOS audit ordering differs')
                counts=[row['positive_by_stride'][str(s)] for s in (8,16,32)]
                total=[x+y for x,y in zip(total,[1,row['gt'],row['matched'],len(row['unmatched']),
                                                int(not any(counts)),*counts],strict=True)]
            require(total==[*expected[:5],*expected[5]],'FCOS per-image totals differ')
        require(next(stream,None) is None,'Unexpected extra coverage records')
    return hashfile(p)


def verify_resume(r,base_sig):
    require(r.get('status')=='PASS_SYNTHETIC_ONLY' and r.get('backend')=='deterministic'
            and r.get('official_test_pixel_reads')==0 and r.get('real_llvip_optimizer_updates')==0
            and r.get('formal_training_started') is False,'Invalid synthetic recovery scope')
    for k in ('environment','upstream','source_sha256'):
        require(r[k]==base_sig[k],f'Synthetic recovery fingerprint differs: {k}')
    require(set(r['arms'])==set('ABCD'),'Missing arm recovery evidence')
    for arm,row in r['arms'].items():
        require(row.get('status')=='BITWISE_EQUAL' and row.get('update')==2
                and all(row.get(k) is True for k in ('bn_buffers','optimizer','all_rng','event_log_prefix')),
                f'Failed recovery evidence: {arm}')
        workers=row['workers']
        require([x.get('phase') for x in workers]==['continuous','interrupt','resume']
                and [x.get('updates') for x in workers]==[2,1,2]
                and workers[-1].get('restored_before_update_exact') is True,
                f'Invalid cross-process evidence: {arm}')
        calls=[(18,72),(9,36),(9,36)] if arm=='D' else [(0,0)]*3
        require(all(x.get('arm')==arm and x.get('native_calls')==dict(fwd=f,bwd=b)
                    for x,(f,b) in zip(workers,calls,strict=True)),f'Invalid native scan calls: {arm}')


def issue(upstream,out):
    require(hashfile(D0)==SPLIT_SHA256,'Frozen split hash differs')
    from .review_registry import TRUSTED_RECEIPTS
    require(TRUSTED_RECEIPTS==frozenset(),'Independent trust registry must remain empty')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Requires the audited single GPU')
    reproducible(3407,True)
    _,_,src=load_sigma(upstream)
    env=runtime(True)
    rpath=out/'resume.json'
    verify_resume(read(rpath),signature(3407,'D',src,env))
    datasha=verify_data(out)
    executions={}
    roster=tuple(read(D0)['fit'])
    sampled={seed:[digest(batches(seed,e,roster)) for e in range(10)] for seed in SEEDS}
    for arm in 'ABCD':
        for seed in SEEDS:
            s=signature(seed,arm,src,env)
            require(s==signature(seed,arm,src,env),'Signature is not repeatable')
            require(s['arm']==arm and s['seed']==seed and s['epochs']==10 and s['batch']==4
                    and s['updates_per_epoch']==2482 and s['deterministic_backend']
                    =='sigma-r3-exclusive-block-pairwise-v1','Frozen run contract differs')
            executions[f'{arm}/{seed}']=dict(signature=s,sha256=digest(s),fit_orders=sampled[seed])
    roots={v['sha256'] for v in executions.values()}
    require(len(roots)==20,'Run signatures collide')
    pp=hashfile(ROOT/PROTOCOL)
    write_json(out/'run-signatures.json',dict(baseline=BASE,runs=executions))
    path=out/'authorization-candidate.json'
    require(not path.exists(),'Candidate already exists')
    receipt=dict(schema='r3-d1b-r1-authorization-v1',baseline=BASE,split_sha256=SPLIT_SHA256,
                 disposition='CANDIDATE_NOT_TRUSTED',gates=dict.fromkeys(REQUIRED,'PASS'),
                 run_signatures={k:v['sha256'] for k,v in executions.items()},
                 data_integrity_sha256=datasha,protocol_sha256=pp,
                 gate_evidence={**{a:dict(file='resume.json',sha256=hashfile(rpath),record=f'arms/{a}')
                                   for a in 'ABCD'},
                     'data_integrity':dict(file='data-integrity.json',sha256=datasha),
                     'protocol_signature':dict(file='run-signatures.json',
                         sha256=hashfile(out/'run-signatures.json'),protocol_sha256=pp)})
    write_json(path,receipt)
    for arm in 'ABCD':
        try:
            require_authorization(path,arm,3407)
        except ValueError as exc:
            require('unreviewed authorization' in str(exc),str(exc))
        else:
            raise RuntimeError('Candidate unexpectedly trusted')
    print('PASS 20 reproducible signatures and 6 evidence-backed gates; all arms still denied',flush=True)
    print('Candidate SHA256:',hashfile(path),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('data','issue'))
    p.add_argument('--root',type=Path)
    p.add_argument('--upstream',type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=='data':
        if a.root is None:
            p.error('data requires --root')
        data_audit(a.root,a.output)
    else:
        if a.upstream is None:
            p.error('issue requires --upstream')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        issue(a.upstream,a.output)
