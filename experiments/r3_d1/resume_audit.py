"""Bounded cross-process audit using only synthetic tensors; never opens LLVIP."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import numpy as np
import torch

from .checkpoint import Checkpoints, Events, rng_state, run_lock, write_json
from .model import Detector
from .deterministic_scan import installed
from .plan import WIDTHS, epoch_indices, optimizer
from .train import external, optimizer_cursor, reproducible, runtime, signature, update
from .upstream import load_sigma


def compare(a,b,key='state'):
    if type(a)!=type(b):
        raise AssertionError(f'{key}: type differs')
    if isinstance(a,torch.Tensor):
        if not (a.shape==b.shape and a.dtype==b.dtype and
                torch.equal(a.reshape(-1).contiguous().view(torch.uint8),
                            b.reshape(-1).contiguous().view(torch.uint8))):
            raise AssertionError(f'{key}: tensor differs bitwise')
    elif isinstance(a,np.ndarray):
        if a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes():
            raise AssertionError(f'{key}: NumPy state differs')
    elif isinstance(a,dict):
        if a.keys()!=b.keys():
            raise AssertionError(f'{key}: keys differ')
        for k in a:
            compare(a[k],b[k],f'{key}/{k}')
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):
            raise AssertionError(f'{key}: length differs')
        for i,(x,y) in enumerate(zip(a,b)):
            compare(x,y,f'{key}/{i}')
    elif a!=b:
        raise AssertionError(f'{key}: values differ')


def isolate_native_backward(original, args, kwargs):
    """Replay identical extension arguments; compare returned gradients only."""
    def invoke():
        # Preserve layouts rather than cloning strides; the extension does not mutate
        # saved inputs. Synchronize each replay before comparing outputs.
        outputs=original(*args,**kwargs)
        torch.cuda.synchronize()
        return outputs
    ref=invoke()
    fields=('du','ddelta','dA','dB','dC','dD','ddelta_bias')
    records={}
    for _ in range(3):
        actual=invoke()
        for label,x,y in zip(fields,ref[:7],actual[:7],strict=True):
            if x is not None and y is not None:
                d=(x-y).abs()
                if torch.count_nonzero(d):
                    n=float(d.max())
                    v=records.setdefault(label,dict(max_abs=0.,different_elements=0))
                    v['max_abs']=max(v['max_abs'],n)
                    v['different_elements']=max(v['different_elements'],int(torch.count_nonzero(d)))
    return dict(input_shape=list(args[0].shape),repeated_replays=3,differences=records)


def worker(args):
    reproducible(3407,True)
    upstream,core,fingerprints=load_sigma(args.upstream)
    env=runtime(True)
    sig=signature(3407,args.arm,fingerprints,env)
    sig['schema']='r3-d1b-r1-synthetic-only-v1'
    sig['audit_backend']=args.backend
    m=Detector(args.arm,widths=WIDTHS,upstream=upstream).cuda().train()
    opt=optimizer(m)
    calls=dict(fwd=0,bwd=0)
    probe=None
    restored_exact=None
    scan_states=[]

    def trace(name,original):
        def traced(*a,**k):
            nonlocal probe
            calls[name]+=1
            if args.arm=='D' and args.phase=='continuous' and name=='bwd' and probe is None and args.backend=='native':
                probe=isolate_native_backward(original,a,k)
            value=original(*a,**k)
            if name=='fwd':
                import hashlib
                scan_states.append(hashlib.sha256(value[1].detach().cpu().contiguous().numpy().tobytes()).hexdigest())
            return value
        return traced

    order=[v.tolist() for v in epoch_indices(3407,0)[:2]]
    from .checkpoint import digest
    def progress(update):
        return dict(update=update,order=order,order_sha256=digest(order))
    original={k:getattr(core,k) for k in calls}
    for k,v in original.items():
        setattr(core,k,trace(k,v))
    try:
        with (installed(core) if args.backend=='deterministic' and args.arm=='D'
              else nullcontext()),run_lock(args.work),Events(args.work/'events.jsonl',resume=args.phase=='resume') as events:
            ck=Checkpoints(args.work)
            p=progress(0)
            if args.phase=='resume':
                def check(p):
                    if p!=progress(1):
                        raise AssertionError('Synthetic cursor differs')
                p=ck.restore(m,opt,sig,events,check,lambda p:optimizer_cursor(opt,m,p['update']))
                saved=snapshot(args.work)
                compare(dict(saved['model']),{k:v.detach().cpu() for k,v in m.state_dict().items()},'restored_model')
                cpu_opt={**opt.state_dict(),'state':{
                    i:{k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in row.items()}
                    for i,row in opt.state_dict()['state'].items()}}
                compare(saved['optimizer'],cpu_opt,'restored_optimizer')
                compare(saved['rng'],rng_state(True),'restored_rng')
                restored_exact=True
            end=1 if args.phase=='interrupt' else 2
            while p['update']<end:
                # All four RNG sources are consumed to catch incomplete RNG restoration.
                bias=(random.random()+float(np.random.random())+float(torch.rand(())))/30
                x=torch.rand(4,3,640,640,device='cuda')+bias
                y=torch.rand(4,3,640,640,device='cuda')-bias
                gt=[torch.tensor([[12,90,44,130],[140,200,230,294],[300,140,460,350]],
                                 dtype=torch.float32,device='cuda')]*4
                scan_states.clear()
                result=update(m,opt,x if args.arm!='B' else None,y if args.arm!='A' else None,gt)
                p=progress(p['update']+1)
                events.append(dict(update=p['update'],synthetic_indices=order[p['update']-1],scan_state_sha256=list(scan_states),**result))
            optimizer_cursor(opt,m,end)
            ck.commit(m,opt,sig,p,events)
            if args.phase=='interrupt':
                events.append({'uncommitted_tail':True})
    finally:
        for k,v in original.items():
            setattr(core,k,v)
    scans=9*(2 if args.phase=='continuous' else 1) if args.arm=='D' else 0
    expected_bwd=scans*(4 if args.backend=='deterministic' else 1)
    if calls!=dict(fwd=scans,bwd=expected_bwd):
        raise AssertionError(f'Unexpected native calls {calls}')
    return dict(arm=args.arm,phase=args.phase,updates=p['update'],native_calls=calls,native_backward_probe=probe,restored_before_update_exact=restored_exact)


def snapshot(root):
    ck=Checkpoints(root)
    entry=ck._read()
    return torch.load(ck._path(entry['last']),map_location='cpu',weights_only=False)


def audit(upstream,output,backend="native"):
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise SystemExit('BLOCKED: requires one CUDA device')
    with tempfile.TemporaryDirectory(prefix='sigma-d1b-synthetic-') as t:
        work=external(t)
        result={'status':'PASS_SYNTHETIC_ONLY','arms':{},'official_test_pixel_reads':0,
                'real_llvip_optimizer_updates':0,'formal_training_started':False,'backend':backend}
        for arm in 'ABCD':
            row={'workers':[]}
            result['arms'][arm]=row
            for phase in ('continuous','interrupt','resume'):
                d=work/(arm+('-continuous' if phase=='continuous' else '-resumed'))
                cmd=[sys.executable,'-m',__spec__.name,'--upstream',str(upstream),
                     '--work',str(d),'--phase',phase,'--arm',arm,'--backend',backend]
                proc=subprocess.run(cmd,text=True,capture_output=True)
                if proc.stderr:
                    print(proc.stderr,end='',file=sys.stderr)
                if proc.returncode:
                    row.update(status='FAIL',failed_phase=phase,exit_code=proc.returncode)
                    result['status']='FAIL'
                    if output:
                        write_json(output,result)
                    raise RuntimeError(f'Worker failed: {arm}/{phase}')
                row['workers'].append(json.loads(proc.stdout.splitlines()[-1]))
            c,r=(snapshot(work/(arm+s)) for s in ('-continuous','-resumed'))
            try:
                compare(c,r)
            except AssertionError as exc:
                row.update(status='FAIL',difference=str(exc))
                result['status']='FAIL'
                kinds={'parameters':dict(c['model']),'bn_buffers':{},'optimizer':c['optimizer'],
                       'rng':c['rng'],'progress':c['progress'],'event_log':c['event_log']}
                other={'parameters':dict(r['model']),'bn_buffers':{},'optimizer':r['optimizer'],
                       'rng':r['rng'],'progress':r['progress'],'event_log':r['event_log']}
                for item in (kinds,other):
                    for k in list(item['parameters']):
                        if k.endswith(('running_mean','running_var','num_batches_tracked')):
                            item['bn_buffers'][k]=item['parameters'].pop(k)
                row['final_checks']={}
                for k in kinds:
                    try:
                        compare(kinds[k],other[k],k)
                    except AssertionError as mismatch:
                        row['final_checks'][k]=str(mismatch)
                    else:
                        row['final_checks'][k]='BITWISE_EQUAL'
                result['environment']=r['signature']['environment']
                result['upstream']=r['signature']['upstream']
                result['source_sha256']=r['signature']['source_sha256']
                if output:
                    write_json(output,result)
                raise
            row.update(status='BITWISE_EQUAL',update=2,bn_buffers=True,optimizer=True,
                       all_rng=True,event_log_prefix=True)
            result['environment']=r['signature']['environment']
            result['upstream']=r['signature']['upstream']
            result['source_sha256']=r['signature']['source_sha256']
            print(arm,'PASS exact model, BN, optimizer, RNG and committed event bytes',flush=True)
        if output:
            write_json(output,result)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--work',type=Path)
    p.add_argument('--phase',choices=('continuous','interrupt','resume'))
    p.add_argument('--backend',choices=('native','deterministic'),default='native')
    p.add_argument('--arm',choices=list('ABCD'))
    a=p.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    print(json.dumps(worker(a) if a.phase else audit(a.upstream,a.output,a.backend),sort_keys=True))
