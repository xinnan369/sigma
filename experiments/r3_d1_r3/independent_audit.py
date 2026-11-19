"""Independent R3 evidence reader. Does not import the issuer or training modules."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

HEAD='3ce9a30c1521df6143cec8d371d5d0b6be579434'
CANDIDATE='eb002f9b3e4c82bdb6bbab6bb205f8ff5f88c0ddb13cc03b47d0796c9550effa'
PARENT='e7f782258be07e3d9718fa19190fe47705e074fe'
ROOT=Path(__file__).resolve().parents[2]
OLD='reports/evidence/r3-d1b-r2r/'
SPLIT='reports/evidence/r3-d0/split.json'
SPLIT_HASH='bd1090cb2ad4696db3ba2f8e984788204c433a7e0cab656073c5e2b67e53e3d4'
PROTOCOL='docs/R3_D1_DETECTION_PROTOCOL.md'
REGISTRY='experiments/r3_d1/review_registry.py'
VERSION='sigma-r3-exclusive-block-pairwise-v1'
FIELDS={'du','ddelta','dA','dB','dC','dD','ddelta_bias'}
SEEDS=range(3407,3412)


def must(ok,msg):
    if not ok:
        raise ValueError(msg)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(obj):
    return sha((json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode())


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args])


def blob(path):
    return git('show',f'{HEAD}:{path}')


def protocol(raw):
    s=raw.decode()
    marker='<!-- BEGIN D1B-R1 ACTIVE CONTRACT -->'
    must(s.count(marker)==1,'Missing or duplicate active contract')
    body,block=s.split(marker)
    j=re.fullmatch(r'\s*```json\s*(.*?)\s*```\s*<!-- END D1B-R1 ACTIVE CONTRACT -->\s*',
                   block,flags=re.S)
    must(j is not None,'Malformed end of active contract')
    value=json.loads(j[1])
    expected={'revision':'D1B-R1','status':'ACTIVE','backend':VERSION,
              'supersedes':'D_NATIVE_BACKWARD_BLOCKED','superseded_rule_active':False,
              'forward':'PINNED_NATIVE_UNCHANGED',
              'backward':'EXCLUSIVE_SAMPLE_CHANNEL_SLOTS_FIXED_PAIRWISE_REDUCTION',
              'comparison':{'scope':'NATIVE_VS_DETERMINISTIC_ONLY','atol':2e-5,'rtol':2e-4},
              'repeat_backward':'BITWISE','checkpoint_resume':'BITWISE',
              'formal_training':'INDEPENDENT_AUTHORIZATION_REQUIRED'}
    must(digest(value)==digest(expected),'JSON contract differs from audit expectation')
    old_clause='旧 D 臂 BLOCKED 条款已由本修订正式取代，不再是现行规则。'
    must(body.count(old_clause)==1,'Missing unambiguous supersession')
    before,after=body.split('## D1B-R1 确定性后端（现行修订）')
    for text in (VERSION,'保持原生 Sigma 前向','按样本拆分','每通道','固定配对二叉树',
                 'atol=2e-5','rtol=2e-4','仍须逐位一致','独立验收登记的可信回执'):
        must(text in after,f'Missing active prose: {text}')
    must(not re.search(r'显式阻止\s*D\s*的正式运行|(?:D\s*臂?|arm\s+D)[^。\n]{0,35}'
                       r'(?:BLOCKED|仍禁止|仍阻断|不得启动|禁止正式)',
                       before+after.replace(old_clause,''),re.I),'Conflicting old D rule')
    return value


def sources(signed,expected_blobs):
    named={p for p in git('ls-tree','-r','--name-only',HEAD,'experiments/r3_d1').decode().splitlines()
           if p.endswith('.py') and p!=REGISTRY}|{PROTOCOL,'experiments/r3_d0/llvip.py'}
    must(set(signed)==set(expected_blobs)==named,'Wrong signed source set')
    for p in named:
        file=(ROOT/p).read_bytes()
        original=blob(p)
        must(sha(file)==signed[p] and sha(original)==expected_blobs[p]
             and file.replace(b'\r\n',b'\n')==original,f'Source drift: {p}')


def signatures(c,ledger,split,env,upstream,signed):
    runs=ledger['runs']
    keys={f'{arm}/{seed}' for arm in 'ABCD' for seed in SEEDS}
    must(ledger['baseline']==PARENT and set(runs)==set(c['run_signatures'])==keys,'Wrong signature matrix')
    orders={}
    fit=split['fit']
    for seed in SEEDS:
        orders[seed]=[]
        for epoch in range(10):
            ix=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,epoch]))).permutation(len(fit))
            must(len(ix)==len(set(ix.tolist()))==9926,'Bad permutation')
            batch=[[fit[int(j)] for j in ix[i:i+4]] for i in range(0,len(ix),4)]
            must(len(batch)==2482 and len(batch[-1])==2,'Bad batch count or tail')
            orders[seed].append(digest(batch))
        for a in 'ABCD':
            key=f'{a}/{seed}'
            expected=dict(schema='r3-d1b-v1',seed=seed,arm=a,cuda=True,split_sha256=SPLIT_HASH,
                          upstream=upstream,environment=env,deterministic_backend=VERSION,
                          precision='fp32',optimizer='AdamW',lr=.0003,betas=[.9,.999],eps=1e-8,
                          conv_decay=.01,other_decay=0.,scheduler=None,epochs=10,batch=4,
                          updates_per_epoch=2482,fit=9926,validation=2087,cnn_widths=[158,370,18],
                          source_sha256=signed)
            row=runs[key]
            must(row['signature']==expected and digest(expected)==row['sha256']==c['run_signatures'][key]
                 and row['fit_orders']==orders[seed],f'Signature or order mismatch: {key}')
    must(len(set(c['run_signatures'].values()))==20
         and len({h for lst in orders.values() for h in lst})==50,'Duplicate signature/digest')
    return orders


def numeric(n):
    must(n['status']=='PASS' and n['version']==VERSION and n['tolerance']==dict(atol=2e-5,rtol=2e-4,replays=4)
         and n['real_llvip_updates']==n['official_test_reads']==0,'Numeric status or scope')
    matrix={(b,d,w*w*f,g) for b in (2,4) for w in (7,20,40,80)
            for d,f,g in ((128,1,1),(256,2,2))}
    must(len(n['raw'])==len(matrix) and {(*r['shape'],r['groups']) for r in n['raw']}==matrix,'Numeric cases incomplete')
    def close(x):
        return x['close'] is True and 0<=x['mixed_ratio']<=1 and 0<=x['max_abs'] and 0<=x['max_relative']
    changed=set()
    raw_ratio=0
    for r in n['raw']:
        must(all(r[x] is True for x in ('deterministic_bitwise','forward_bitwise','scan_buffer_bitwise','numerical_pass')),
             'Raw bitwise check failed')
        native=r['original_replays']
        cand=r['candidate_vs_native']
        must(len(native)==4 and len(cand)==5 and all(set(a)==FIELDS for a in native+cand),'Gradient matrix incomplete')
        changed.update(k for a in native for k,v in a.items() if v['differing_bytes'])
        must(all(close(v) for a in cand for v in a.values()),'Raw comparison failed')
        raw_ratio=max(raw_ratio,*(v['mixed_ratio'] for a in cand for v in a.values()))
    must(changed==FIELDS-{'du','ddelta'},'Native nondeterministic gradient set differs')
    maxima={}
    param_names=None
    for b in (4,2):
        r=n['detector'][f'D/{b}']
        must(r['detector_forward_bitwise'] is True and r['deterministic_repeat_bitwise'] is True,
             'Detector bitwise check failed')
        names=set(r['gradients'])
        if param_names is None:
            param_names=names
        must(names==param_names and len(names)==155 and set(r['inputs'])=={'rgb','thermal'},
             'Incomplete model gradient coverage')
        must(all(close(v) for k in ('gradients','inputs') for v in r[k].values()),'Detector gradient tolerance failed')
        maxima[str(b)]={k:max(v['max_abs'] for v in r[k].values()) for k in ('gradients','inputs')}
    return dict(raw_cases=len(matrix),native_changed=sorted(changed),raw_max_mixed_ratio=raw_ratio,
                detector_max_abs=maxima,parameter_tensors=len(param_names))


def recovery(r):
    must(r['backend']=='deterministic' and r['status']=='PASS_SYNTHETIC_ONLY'
         and r['official_test_pixel_reads']==r['real_llvip_optimizer_updates']==0
         and r['formal_training_started'] is False and set(r['arms'])==set('ABCD'),'Recovery scope failed')
    for a,arm in r['arms'].items():
        must(arm['status']=='BITWISE_EQUAL' and arm['update']==2 and all(
             arm[k] is True for k in ('optimizer','bn_buffers','all_rng','event_log_prefix')),f'Recovery failed {a}')
        w=arm['workers']
        must([(x['phase'],x['updates'],x['arm']) for x in w]==
             [(p,u,a) for p,u in [('continuous',2),('interrupt',1),('resume',2)]]
             and w[-1]['restored_before_update_exact'] is True,f'Cross-process count {a}')
        must([(x['native_calls']['fwd'],x['native_calls']['bwd']) for x in w]==
             ([(18,72),(9,36),(9,36)] if a=='D' else [(0,0)]*3),f'Native calls {a}')
    return list(sorted(r['arms']))


def data(d,coverage,ledger,split):
    must(d['status']=='PASS' and coverage['status']=='MEASURED_UNCHANGED_PROTOCOL'
         and d['split_sha256']==coverage['split_sha256']==SPLIT_HASH and
         all(d[k]==0 for k in ('official_test_reads','quarantine_reads','real_llvip_optimizer_updates'))
         and coverage['official_test_reads']==coverage['image_pixel_reads']==0,'Unsafe data evidence')
    must(d['per_image_sha256']==sha(ledger),'Coverage ledger digest differs')
    totals={k:Counter() for k in ('fit','validation')}
    ids={k:[] for k in totals}
    for line in ledger.splitlines():
        r=json.loads(line)
        role=r['role']
        ids[role].append(r['id'])
        m=len(r['unmatched'])
        points=r['positive_by_stride']
        must(r['gt']==r['matched']+m and set(points)=={'8','16','32'}
             and all(v>=0 for v in points.values()),'Inconsistent per-image GT or stride count')
        totals[role].update(images=1,gt=r['gt'],matched=r['matched'],unmatched=m,
                            zero_positive_images=int(sum(points.values())==0))
        totals[role].update({'positive_stride_'+k:v for k,v in points.items()})
        totals[role].update(x['reason'] for x in r['unmatched'])
    for role,expected in [('fit',(9926,28547,28540)),('validation',(2087,5536,5530))]:
        n,gt,m=expected
        must(ids[role]==split[role] and len(set(ids[role]))==n,'Roster drift')
        must(d['roles'][role]==dict(pairs=n,verified_jpeg_files=2*n,verified_xml_files=n,
             valid_gt=gt,roster_sha256=digest(ids[role])),'Data read counts differ')
        must(dict(totals[role])==coverage['roles'][role]['totals']
             and (totals[role]['gt'],totals[role]['matched'])==(gt,m),'FCOS totals differ')
    return {k:dict(v) for k,v in totals.items()}


def exit_code(path):
    m=re.search(rb'Exit code: (\d+)\s*$',path.read_bytes())
    must(m is not None,f'Missing exit code: {path.name}')
    return int(m[1])


def audit(fresh):
    must(fresh.resolve().is_dir() and not fresh.resolve().is_relative_to(ROOT),'Require external rerun directory')
    must(git('merge-base',HEAD,'HEAD').decode().strip()==HEAD,'Audited commit not in checkout ancestry')
    pre=lambda name:blob(OLD+name)
    load=lambda name:json.loads(pre(name))
    run=lambda name:json.loads((fresh/name).read_bytes())
    registry=blob(REGISTRY)
    must(registry==(ROOT/REGISTRY).read_bytes().replace(b'\r\n',b'\n')
         and registry.rstrip().endswith(b'TRUSTED_RECEIPTS=frozenset()'),'Trust registry changed')
    verification=load('verification.json')
    names={s.rsplit('/',1)[-1] for s in git('ls-tree','-r','--name-only',HEAD,OLD).decode().splitlines()}
    must(names-{'verification.json'}==set(verification['files_sha256']),'Unlisted R2R evidence')
    for name,h in verification['files_sha256'].items():
        must(sha(pre(name))==h,f'Old evidence hash fails: {name}')
    must(sha(pre('authorization-candidate.json'))==CANDIDATE,'Unexpected candidate hash')
    c=load('authorization-candidate.json')
    gates={'A','B','C','D','data_integrity','protocol_signature'}
    must(c['schema']=='r3-d1b-r1-authorization-v1' and c['baseline']==PARENT
         and c['disposition']=='CANDIDATE_NOT_TRUSTED' and c['split_sha256']==SPLIT_HASH
         and c['gates']==dict.fromkeys(gates,'PASS') and set(c['gate_evidence'])==gates,'Candidate fields differ')
    expected_refs={**{a:('resume.json',f'arms/{a}') for a in 'ABCD'},
                   'data_integrity':('data-integrity.json',None),
                   'protocol_signature':('run-signatures.json',None)}
    for k,v in c['gate_evidence'].items():
        must((v['file'],v.get('record'))==expected_refs[k],f'Wrong gate reference: {k}')
        for prefix in ('','numerics_','verification_'):
            if prefix+'file' in v:
                must(sha(pre(v[prefix+'file']))==v[prefix+'sha256'],f'Broken gate evidence: {k}')
    must(sha(pre('data-integrity.json'))==c['data_integrity_sha256'],'Data root hash differs')
    proof=load('signature-verification.json')
    must(proof['status']=='PASS' and proof['signature_count']==20
         and proof['independently_recomputed_epoch_order_count']==50
         and proof['protocol_sha256']==c['protocol_sha256']
         and proof['persisted_ledger_sha256']==sha(pre('run-signatures.json'))
         and proof['numerics_sha256']==sha(pre('numerics.json')),'Verification receipt differs')
    checked=protocol(blob(PROTOCOL))
    must(checked==proof['active_contract'] and sha((ROOT/PROTOCOL).read_bytes())==c['protocol_sha256'],
         'Protocol digest or contract differs')
    split_blob=blob(SPLIT)
    must(sha(split_blob)==sha((ROOT/SPLIT).read_bytes())==SPLIT_HASH,'Split digest differs')
    split=json.loads(split_blob)
    must(not (set(split['fit']) & set(split['validation'])) and
         not (set(split['fit'])|set(split['validation']))&set(split['quarantine']),'Development roles overlap')
    nr,rr=load('numerics.json'),load('resume.json')
    now_nr,now_rr=run('numerics.json'),run('resume.json')
    signed=nr['source_sha256']
    sources(signed,verification['source_blobs_sha256_lf'])
    for k in ('environment','upstream','source_sha256'):
        must(nr[k]==rr[k]==now_nr[k]==now_rr[k],f'Fresh runtime fingerprint differs: {k}')
    must(nr['environment']['gpu']['name']=='NVIDIA GeForce RTX 3090','Not audited RTX 3090')
    src=nr['upstream']
    must(src['commit']=='98ba3c6938d29169bbd27f91519084bb462561df'
         and sha(Path(src['extension_file']).read_bytes())==src['extension_sha256']
         and sha(Path(src['wrapper_file']).read_bytes())==src['wrapper_sha256'],'Local upstream binary/wrapper drift')
    relative='models/encoders/vmamba.py'
    path=Path(src['source_file'])
    upstream=path.parents[2]
    must(path==upstream/relative and subprocess.check_output(
        ['git','-C',str(upstream),'rev-parse','HEAD']).decode().strip()==src['commit'],
        'Runtime Sigma checkout commit differs')
    expected_upstream={relative,'models/encoders/dual_vmamba.py',
        'models/encoders/selective_scan/selective_scan/selective_scan_interface.py'}
    must(set(src['source_sha256'])==expected_upstream,'Unexpected upstream source scope')
    for rel,h in src['source_sha256'].items():
        raw=(upstream/rel).read_bytes()
        orig=subprocess.check_output(['git','-C',str(upstream),'show',f"{src['commit']}:{rel}"])
        must(sha(raw)==h and raw.replace(b'\r\n',b'\n')==orig,f'Upstream source differs: {rel}')
    orders=signatures(c,load('run-signatures.json'),split,nr['environment'],src,signed)
    numerical={'historical':numeric(nr),'rerun':numeric(now_nr)}
    restored=recovery(now_rr)
    recovery(rr)
    must(now_rr['arms']==rr['arms'],'Recovery worker evidence differs')
    old_data,new_data=load('data-integrity.json'),run('data-integrity.json')
    old_coverage,new_coverage=load('gt-coverage.json'),run('gt-coverage.json')
    ledger=verification['duplicate_per_image_ledger']
    old_ledger=blob(ledger['path'])
    new_ledger=(fresh/'gt-per-image.jsonl').read_bytes()
    must(old_ledger==new_ledger and sha(new_ledger)==ledger['sha256'],'Per-image ledger differs')
    must(old_data['coverage_sha256']==sha(pre('gt-coverage.json')) and
         new_data['coverage_sha256']==sha((fresh/'gt-coverage.json').read_bytes()),'Coverage SHA mismatch')
    totals=data(new_data,new_coverage,new_ledger,split)
    must(totals==data(old_data,old_coverage,old_ledger,split),'Coverage drift')
    codes={p.stem:exit_code(p) for p in fresh.glob('*.log')}
    must({'cpu','compileall','numerics','resume','data'}<=codes.keys(),'Missing run log')
    status='PASS' if all(v==0 for v in codes.values()) else 'HOLD'
    return dict(status=status,head=HEAD,candidate_sha256=CANDIDATE,signature_count=20,
                independently_rebuilt_orders=sum(map(len,orders.values())),active_contract=checked,
                numerical=numerical,resume_arms=restored,development_totals=totals,
                coverage_ledger_sha256=sha(new_ledger),exit_codes=codes,
                fresh_file_sha256={p.name:sha(p.read_bytes()) for p in sorted(fresh.iterdir()) if p.is_file()},
                runtime=nr['environment'],registry_empty=True,authorized=False,
                official_test_reads=0,real_llvip_updates=0)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--fresh',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        result=audit(a.fresh)
    except (FileNotFoundError,NotADirectoryError) as exc:
        raise SystemExit(f'BLOCKED: {exc}') from exc
    with a.output.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write('\n')
    print(result['status'], 'candidate remains untrusted')
    raise SystemExit(0 if result['status']=='PASS' else 1)
