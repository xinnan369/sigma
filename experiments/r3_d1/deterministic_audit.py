"""Fixed synthetic matrix and resource audit for native/isolated-backward Sigma."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch

from .checkpoint import write_json
from .deterministic_scan import VERSION, backward, installed
from .model import Detector
from .fcos import losses
from .plan import WIDTHS, optimizer
from .resume_audit import compare
from .train import reproducible, runtime, signature
from .upstream import load_sigma

ATOL=2e-5
RTOL=2e-4
REPEATS=4
FIELDS=('du','ddelta','dA','dB','dC','dD','ddelta_bias')


def differences(a,b):
    d=(a-b).abs()
    # Relative error is reported both with and without a near-zero floor.
    eps=torch.finfo(a.dtype).tiny
    raw=(d.double()/a.abs().double().clamp_min(eps)).max().item()
    denom=a.abs()+ATOL/RTOL
    return dict(max_abs=d.max().item(),max_relative=raw,
                max_relative_stabilized=(d/denom).max().item(),
                mixed_ratio=(d/(ATOL+RTOL*a.abs())).max().item(),
                differing_bytes=int((a.reshape(-1).contiguous().view(torch.uint8)!=
                                      b.reshape(-1).contiguous().view(torch.uint8)).sum()),
                close=bool(torch.allclose(a,b,atol=ATOL,rtol=RTOL)))


def multi_diff(a,b,names):
    return {k:differences(x,y) for k,x,y in zip(names,a,b,strict=True)}


def raw_case(core,batch,dim,groups,length):
    device='cuda'
    def rand(*shape):
        return (torch.rand(*shape,device=device)-.5)*.4
    u,dt,dout=(rand(batch,dim,length) for _ in range(3))
    a=-torch.exp(rand(dim,4))
    b,c=(rand(batch,groups,4,length) for _ in range(2))
    d,bias=(rand(dim) for _ in range(2))
    native_fwd,saved=core.fwd(u,dt,a,b,c,d,bias,True,1)
    with installed(core):
        candidate_fwd,candidate_saved=core.fwd(u,dt,a,b,c,d,bias,True,1)
    compare(native_fwd,candidate_fwd,'forward')
    compare(saved,candidate_saved,'saved scan state')
    args=(u,dt,a,b,c,d,bias,dout,saved,True,1)
    native=core.bwd(*args)
    deterministic=backward(core.bwd,*args)
    against=[multi_diff(native,deterministic,FIELDS)]
    replay=[]
    for _ in range(REPEATS):
        new_native=core.bwd(*args)
        replay.append(multi_diff(native,new_native,FIELDS))
        against.append(multi_diff(new_native,deterministic,FIELDS))
        compare(deterministic,backward(core.bwd,*args),'all backward gradients')
    return dict(shape=[batch,dim,length],groups=groups,original_replays=replay,
                candidate_vs_native=against,deterministic_bitwise=True,
                forward_bitwise=True,scan_buffer_bitwise=True,
                numerical_pass=all(v['close'] for row in against for v in row.values()))


def tick():
    torch.cuda.synchronize()
    return time.perf_counter()


def integrated(vm,core,arm,batch,det,inputs,repeats=3):
    """Same initialization and data on both paths; no dataset or real update."""
    m=Detector(arm,widths=WIDTHS,upstream=vm).cuda().train()
    opt=optimizer(m)
    original_state={k:v.clone() for k,v in m.state_dict().items()}
    ref=None
    timings=[]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gt=[torch.tensor([[12,90,44,130],[140,200,230,294],[300,140,460,350]],
                     dtype=torch.float32,device='cuda')]*batch
    with (installed(core) if det else nullcontext()):
        for i in range(repeats+1):
            m.load_state_dict(original_state)
            opt.state.clear()
            opt.zero_grad(set_to_none=True)
            x={k:v.detach().clone().requires_grad_() for k,v in inputs.items()
               if (k!='rgb' or arm!='B') and (k!='thermal' or arm!='A')}
            start=tick()
            preds=m(**x)
            f=tick()
            objective=sum(losses(preds,gt).values())
            loss_end=tick()
            objective.backward()
            b=tick()
            opt.step()
            s=tick()
            vals=[p.grad for p in m.parameters()]+[p.grad for p in x.values()]
            if any(v is None or not torch.isfinite(v).all() for v in vals):
                raise RuntimeError('Nonfinite or missing gradient')
            current=dict(predictions=[p.detach() for p in preds],
                         gradients={k:p.grad.detach() for k,p in m.named_parameters()},
                         inputs={k:p.grad.detach() for k,p in x.items()})
            if ref is None:
                ref=current
            elif det:
                compare(ref,current,f'{arm}/{batch}/detector')
            if i:
                timings.append([f-start,loss_end-f,b-loss_end,s-b,s-start])
    report=dict(zip(('forward_ms','loss_ms','backward_ms','adamw_ms','update_ms'),
                    [1000*statistics.median(c) for c in zip(*timings)]))
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    report['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    report['baseline_allocated_note']='includes inputs, one reference gradient snapshot and cloned initial state'
    return ref,report


def run(upstream,out):
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('BLOCKED: exactly one CUDA device required')
    reproducible(3407,True)
    vm,core,source=load_sigma(upstream)
    env=runtime(True)
    from .upstream import sha
    from .deterministic_scan import __file__ as adapter
    tool=lambda c:subprocess.run(c,text=True,capture_output=True).stdout.strip()
    result=dict(version=VERSION,adapter_sha256=sha(adapter),upstream=source,environment=env,
                source_sha256=signature(3407,'D',source,env)['source_sha256'],
                toolchain=dict(nvcc_installed=tool(['nvcc','--version']),host_cxx=tool(['c++','--version']),
                               note='No new CUDA binary built; pinned extension reused, tree adds use PyTorch CUDA.'),
                tolerance=dict(atol=ATOL,rtol=RTOL,replays=REPEATS),raw=[],detector={},
                official_test_reads=0,real_llvip_updates=0,status='RUNNING')
    try:
        for batch in (4,2):
            for n in (80,40,20,7):
                for dim,groups,factor in ((128,1,1),(256,2,2)):
                    print('raw',batch,n,dim,flush=True)
                    result['raw'].append(raw_case(core,batch,dim,groups,n*n*factor))
            inputs={k:torch.rand(batch,3,640,640,device='cuda') for k in ('rgb','thermal')}
            for arm in 'ABCD':
                key=f'{arm}/{batch}'
                print('model',key,flush=True)
                original,t0=integrated(vm,core,arm,batch,False,inputs)
                if arm=='D':
                    new,t1=integrated(vm,core,arm,batch,True,inputs)
                    compare(original['predictions'],new['predictions'],'all detector forwards')
                    metrics={key:{k:differences(x,new[key][k]) for k,x in original[key].items()}
                             for key in ('gradients','inputs')}
                    result['detector'][key]=dict(native=t0,deterministic=t1,
                       detector_forward_bitwise=True,deterministic_repeat_bitwise=True,**metrics)
                else:
                    result['detector'][key]=dict(native=t0)
        raw_ok=all(r['numerical_pass'] for r in result['raw'])
        model_ok=all(m['close'] for b in (4,2) for key in ('gradients','inputs')
                     for m in result['detector'][f'D/{b}'][key].values())
        result['status']='PASS' if raw_ok and model_ok else 'FAIL_NUMERICAL'
        result['five_seed_compute_hours']={arm:50*sum(
            result['detector'][f'{arm}/{b}']['deterministic' if arm=='D' else 'native']['update_ms']
            *steps/1000 for b,steps in ((4,2481),(2,1)))/3600 for arm in 'ABCD'}
        result['five_seed_estimate_scope']='synthetic FCOS forward, assignment/loss, backward and AdamW; no data I/O, validation or checkpoints'
    except Exception as exc:
        result.update(status='FAIL',failure=repr(exc))
        raise
    finally:
        write_json(out,result)
    if result['status']!='PASS':
        raise RuntimeError(result['status'])
    print('PASS numerical and repeat audits',flush=True)


if __name__=='__main__':
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    p=argparse.ArgumentParser()
    p.add_argument('--upstream',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    run(a.upstream,a.output)
