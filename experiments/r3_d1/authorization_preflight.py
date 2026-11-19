"""Read-only candidate issuance. Never installs a receipt or starts training."""
import argparse
import hashlib
import re
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

BASE='e7f782258be07e3d9718fa19190fe47705e074fe'
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


ACTIVE_MARKER='<!-- BEGIN D1B-R1 ACTIVE CONTRACT -->'
END_MARKER='<!-- END D1B-R1 ACTIVE CONTRACT -->'
ATOL,RTOL=2e-5,2e-4
BACKEND='sigma-r3-exclusive-block-pairwise-v1'
ACTIVE_CONTRACT=dict(revision='D1B-R1',status='ACTIVE',backend=BACKEND,
    supersedes='D_NATIVE_BACKWARD_BLOCKED',superseded_rule_active=False,
    forward='PINNED_NATIVE_UNCHANGED',
    backward='EXCLUSIVE_SAMPLE_CHANNEL_SLOTS_FIXED_PAIRWISE_REDUCTION',
    comparison=dict(scope='NATIVE_VS_DETERMINISTIC_ONLY',atol=ATOL,rtol=RTOL),
    repeat_backward='BITWISE',checkpoint_resume='BITWISE',
    formal_training='INDEPENDENT_AUTHORIZATION_REQUIRED')


def unique_keys(pairs):
    d=dict(pairs)
    require(len(d)==len(pairs),'Duplicate protocol field')
    return d


def verify_protocol(text):
    require(text.count(ACTIVE_MARKER)==text.count(END_MARKER)==1,
            'Expected one active D1B-R1 contract')
    body,tail=text.split(ACTIVE_MARKER)
    segment,remainder=tail.split(END_MARKER)
    region=re.fullmatch(r'\s*```json\s*([\s\S]+?)\s*```\s*',segment)
    require(region is not None and not remainder.strip(),'Invalid or nonfinal active contract')
    contract=json.loads(region[1],object_pairs_hook=unique_keys)
    require(digest(contract)==digest(ACTIVE_CONTRACT),'Active backend or acceptance standard differs')
    declaration='旧 D 臂 BLOCKED 条款已由本修订正式取代，不再是现行规则。'
    require(body.count(declaration)==1,'Missing explicit old-rule supersession')
    heading='## D1B-R1 确定性后端（现行修订）'
    require(body.count(heading)==1,'Missing unique active amendment heading')
    prior,amendment=body.split(heading)
    for marker in ('sigma-r3-exclusive-block-pairwise-v1','atol=2e-5','rtol=2e-4',
                   '仍须逐位一致','独立验收登记的可信回执'):
        require(marker in amendment,f'Missing normative amendment declaration: {marker}')
    # Refuse the obsolete current-rule sentence and competing blocking declarations.
    remaining=prior+'\n'+amendment.replace(declaration,'')
    conflict=r'显式阻止\s*D\s*的正式运行|(?:D\s*臂?|arm\s+D)[^。\n]{0,35}(?:BLOCKED|仍禁止|仍阻断|不得启动|禁止正式)|旧\s*D[^。\n]{0,35}(?:继续有效|仍有效)'
    require(not re.search(conflict,remaining,re.I),'Obsolete D block remains in active prose')
    return contract


def verify_numerics(n,base_sig):
    require(n.get('status')=='PASS' and n.get('version')==BACKEND
            and n.get('tolerance')==dict(atol=ATOL,rtol=RTOL,replays=4)
            and n.get('official_test_reads')==n.get('real_llvip_updates')==0,
            'Invalid numerical evidence scope or tolerances')
    for k in ('environment','upstream','source_sha256'):
        require(n[k]==base_sig[k],f'Numerical fingerprint differs: {k}')
    expected={(b,d,hw*hw*f,g) for b in (4,2) for hw in (80,40,20,7)
              for d,g,f in ((128,1,1),(256,2,2))}
    rows=n['raw']
    require(len(rows)==16 and {(*r['shape'],r['groups']) for r in rows}==expected,
            'Numerical case matrix differs')
    require(all(all(r.get(k) is True for k in ('numerical_pass','forward_bitwise',
                'scan_buffer_bitwise','deterministic_bitwise'))
                and len(r['candidate_vs_native'])==5
                and all(set(a)=={'du','ddelta','dA','dB','dC','dD','ddelta_bias'}
                        and all(v['close'] is True and 0<=v['mixed_ratio']<=1
                                for v in a.values()) for a in r['candidate_vs_native'])
                for r in rows),'Raw numerical gate failed')
    for batch in (4,2):
        r=n['detector'][f'D/{batch}']
        require(r['detector_forward_bitwise'] is True and r['deterministic_repeat_bitwise'] is True
                and set(r['inputs'])=={'rgb','thermal'} and bool(r['gradients'])
                and all(v['close'] is True and 0<=v['mixed_ratio']<=1 for k in ('inputs','gradients') for v in r[k].values()),
                'Detector numerical gate failed')


def independent_digest(value):
    # Separate from the runner digest implementation; includes the terminal LF.
    wire=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    return hashlib.sha256((wire+'\n').encode('utf-8')).hexdigest()


def independent_orders(roster):
    require(len(roster)==len(set(roster))==9926,'Unexpected Fit roster')
    result={}
    for seed in SEEDS:
        result[seed]=[]
        for epoch in range(10):
            rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,epoch])))
            ids=[roster[int(i)] for i in rng.permutation(9926)]
            window=[ids[k:k+4] for k in range(0,9926,4)]
            require(len(window)==2482 and len(window[-1])==2,'Independent batch shape differs')
            result[seed].append(independent_digest(window))
    require(len({h for rows in result.values() for h in rows})==50,'Epoch digests not distinct')
    return result


def independent_signature(seed,arm,src,env):
    """Bind the approved v1 contract independently of train.signature()."""
    files={p.relative_to(ROOT).as_posix():hashfile(p)
           for p in (ROOT/'experiments/r3_d1').rglob('*.py')
           if p.name!='review_registry.py'}
    files.update({rel:hashfile(ROOT/rel) for rel in (PROTOCOL,'experiments/r3_d0/llvip.py')})
    return dict(schema='r3-d1b-v1',seed=seed,arm=arm,cuda=True,split_sha256=SPLIT_SHA256,
                upstream=src,environment=env,deterministic_backend=BACKEND,precision='fp32',
                optimizer='AdamW',lr=.0003,betas=[.9,.999],eps=1e-8,conv_decay=.01,
                other_decay=0.,scheduler=None,epochs=10,batch=4,updates_per_epoch=2482,
                fit=9926,validation=2087,cnn_widths=[158,370,18],source_sha256=files)


def verify_persisted(path,roster,src,env):
    doc=read(path)
    require(doc.get('baseline')==BASE,'Signature ledger baseline differs')
    runs=doc['runs']
    require(set(runs)=={f'{a}/{s}' for a in 'ABCD' for s in SEEDS},'Wrong run matrix')
    orders=independent_orders(roster)
    hashes={}
    for arm in 'ABCD':
        for seed in SEEDS:
            key=f'{arm}/{seed}'
            row=runs[key]
            fresh=independent_signature(seed,arm,src,env)
            require(row['signature']==fresh,f'Persisted signature fields differ: {key}')
            check=independent_digest(fresh)
            require(check==row['sha256']==digest(fresh),f'Persisted signature SHA256 differs: {key}')
            require(row['fit_orders']==orders[seed],f'Independent epoch order differs: {key}')
            hashes[key]=check
    require(len(set(hashes.values()))==20,'Run signatures collide')
    return hashes,orders


def issue(upstream,out):
    require(hashfile(D0)==SPLIT_SHA256,'Frozen split hash differs')
    from .review_registry import TRUSTED_RECEIPTS
    require(TRUSTED_RECEIPTS==frozenset(),'Independent trust registry must remain empty')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Requires the audited single GPU')
    reproducible(3407,True)
    _,_,src=load_sigma(upstream)
    env=runtime(True)
    protocol=ROOT/PROTOCOL
    pp=hashfile(protocol)
    contract=verify_protocol(protocol.read_text(encoding='utf-8'))
    npath=out/'numerics.json'
    base_sig=signature(3407,'D',src,env)
    verify_numerics(read(npath),base_sig)
    rpath=out/'resume.json'
    verify_resume(read(rpath),base_sig)
    datasha=verify_data(out)
    executions={}
    roster=tuple(read(D0)['fit'])
    sampled={seed:[digest(batches(seed,e,roster)) for e in range(10)] for seed in SEEDS}
    for arm in 'ABCD':
        for seed in SEEDS:
            s=signature(seed,arm,src,env)
            require(s['arm']==arm and s['seed']==seed and s['epochs']==10 and s['batch']==4
                    and s['updates_per_epoch']==2482 and s['deterministic_backend']
                    =='sigma-r3-exclusive-block-pairwise-v1','Frozen run contract differs')
            executions[f'{arm}/{seed}']=dict(signature=s,sha256=digest(s),fit_orders=sampled[seed])
    path=out/'authorization-candidate.json'
    require(not path.exists(),'Candidate already exists')
    ledger=out/'run-signatures.json'
    write_json(ledger,dict(baseline=BASE,runs=executions))
    hashes,orders=verify_persisted(ledger,roster,src,env)
    require(hashfile(protocol)==pp,'Protocol changed during issuance')
    verification=out/'signature-verification.json'
    write_json(verification,dict(status='PASS',signature_count=len(hashes),
        independently_recomputed_epoch_order_count=sum(map(len,orders.values())),
        persisted_ledger_sha256=hashfile(ledger),active_contract=contract,
        protocol_sha256=pp,numerics_sha256=hashfile(npath)))
    receipt=dict(schema='r3-d1b-r1-authorization-v1',baseline=BASE,split_sha256=SPLIT_SHA256,
                 disposition='CANDIDATE_NOT_TRUSTED',gates=dict.fromkeys(REQUIRED,'PASS'),
                 run_signatures=hashes,
                 data_integrity_sha256=datasha,protocol_sha256=pp,
                 gate_evidence={**{a:dict(file='resume.json',sha256=hashfile(rpath),record=f'arms/{a}')
                                   for a in 'ABCD'},
                     'data_integrity':dict(file='data-integrity.json',sha256=datasha),
                     'protocol_signature':dict(file='run-signatures.json',
                         sha256=hashfile(ledger),protocol_sha256=pp,
                         verification_file=verification.name,verification_sha256=hashfile(verification),
                         numerics_file=npath.name,numerics_sha256=hashfile(npath))})
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
