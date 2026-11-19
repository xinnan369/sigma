"""Bounded synthetic CUDA preflight. No data path, checkpoint, or optimizer step."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import torch

from experiments.r3_d0.llvip import transform

from .budget import complete_count, match
from .fcos import decode, losses
from .model import Detector, SEEDS
from .upstream import load_sigma, sha
from .data import normalize
from .plan import WIDTHS


def fingerprint(state):
    digest=hashlib.sha256()
    for k,v in sorted(state.items()):
        digest.update(f'{k}:{v.dtype}:{tuple(v.shape)}\n'.encode())
        digest.update(v.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def profile(model, inputs, core=None):
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    result={}
    rng=np.random.default_rng(3407)
    raw_rgb=rng.integers(0,256,size=(1024,1280,3),dtype=np.uint8)
    raw_t=rng.integers(0,256,size=(1024,1280),dtype=np.uint8)
    for mode in ('model','model_and_cpu_decode','preprocess_model_and_decode'):
        samples=[]
        with torch.inference_mode():
            for i in range(120):
                torch.cuda.synchronize()
                start=time.perf_counter()
                args=inputs
                if mode=='preprocess_model_and_decode':
                    r,t,_=transform(raw_rgb,raw_t,[])
                    args={k:normalize(r if k=='rgb' else t).unsqueeze(0).cuda() for k in inputs}
                output=model(**args)
                if mode!='model':
                    decode(output)
                del output
                torch.cuda.synchronize()
                if i>=20:
                    samples.append((time.perf_counter()-start)*1000)
        q=statistics.quantiles(samples,n=100,method='inclusive')
        result[mode]=dict(median_ms=statistics.median(samples),p95_ms=q[94],
                          warmup=20,measurements=100)
    result['peak_inference_allocated_bytes']=torch.cuda.max_memory_allocated()
    model.train()
    samples=[]
    snapshots=[]
    torch.cuda.reset_peak_memory_stats()
    for batch in (4,2):
        x={k:v.repeat(batch,1,1,1) for k,v in inputs.items()}
        gt=[torch.tensor([[12,90,44,130],[140,200,230,294],[300,140,460,350]],
                         dtype=torch.float32,device='cuda')]*batch
        for i in range(5):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start=time.perf_counter()
            scores=model(**x)
            if any(not torch.isfinite(p).all() for p in scores):
                raise RuntimeError('Nonfinite FCOS output')
            parts=losses(scores,gt)
            total=sum(parts.values())
            total.backward()
            torch.cuda.synchronize()
            if i:
                samples.append((batch,time.perf_counter()-start))
            if i==4:
                bad=[k for k,p in model.named_parameters() if p.grad is None
                     or not torch.isfinite(p.grad).all()]
                if bad or not torch.isfinite(total) or not all(float(v)>0 for v in parts.values()):
                    raise RuntimeError(f'Loss/gradient check failed: {bad}')
                active={}
                if core is not None:
                    for name in ('cross','concat'):
                        active[name]=[sum(bool(torch.count_nonzero(p.grad)) for p in mod.parameters())
                                      for mod in getattr(model,name)]
                        if any(not a for a in active[name]):
                            raise RuntimeError(f'Inactive fusion branch: {name}')
                snapshots.append(dict(batch=batch,loss={k:float(v.detach()) for k,v in parts.items()},
                                      nonzero_gradient_tensors_per_fusion_scale=active))
            del scores,total,parts
        del x,gt
    mem=torch.cuda.max_memory_allocated()
    adam_bytes=2*sum(p.numel()*p.element_size() for p in model.parameters())
    med={b:statistics.median(s for k,s in samples if k==b) for b in (4,2)}
    result.update(synthetic_forward_backward=snapshots,
                  training_step_median_seconds=med,peak_training_allocated_bytes=mem,
                  unallocated_adam_state_bytes=adam_bytes,
                  five_seed_compute_only_hours=5*10*(2481*med[4]+med[2])/3600)
    model.zero_grad(set_to_none=True)
    return result


def run(upstream, output):
    if not torch.cuda.is_available():
        raise RuntimeError('BLOCKED: CUDA unavailable')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(3407)
    vm,core,source=load_sigma(upstream)
    models={}
    receipts={}
    device=torch.device('cuda:0')
    x,y=(torch.rand(1,3,640,640,device=device) for _ in range(2))
    paired=dict(rgb=x,thermal=y)
    def build_and_count(arm,inputs,widths=None):
        m=Detector(arm,widths=widths,upstream=vm).to(device).eval()
        models[arm]=m
        receipts[arm]=complete_count(m,inputs,core if arm=='D' else None)
        receipts[arm]['initial_state_sha256']=fingerprint(m.state_dict())
        print(arm,'count',receipts[arm]['parameters'],receipts[arm]['mac'],flush=True)
    build_and_count('D',paired)
    build_and_count('A',dict(rgb=x))
    build_and_count('B',dict(thermal=y))
    p=receipts['A']['parameters']+receipts['B']['parameters']-sum(v.numel() for v in models['A'].head.parameters())
    common=receipts['A']['mac']+receipts['B']['mac']-(64*64*9+64*6)*8400
    pick=match(receipts['D'],p,common)
    audit=dict(upstream=source,runtime=dict(python=platform.python_version(),torch=torch.__version__,
               cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
               gpu=torch.cuda.get_device_name(),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
               timm=version('timm'),einops=version('einops'),cpu_threads=torch.get_num_threads()),
               counts=receipts,search=pick,source_sha256={f.name:sha(f) for f in sorted(Path(__file__).parent.glob('*.py'))},
               official_test_reads=0,optimizer_updates=0,seeds=list(SEEDS))
    def save():
        output.write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    if pick is None:
        audit['status']='BLOCKED_BUDGET'
        save()
        raise RuntimeError('No simultaneous parameter/MAC match')
    if tuple(pick['widths'])!=WIDTHS:
        raise RuntimeError('Width search disagrees with frozen widths')
    build_and_count('C',paired,pick['widths'])
    C,D=receipts['C'],receipts['D']
    if (C['parameters'],C['mac'])!=(pick['parameters'],pick['mac']):
        raise RuntimeError('Instantiated CNN differs from search arithmetic')
    diffs={k:abs(C[k]/D[k]-1) for k in ('parameters','mac')}
    assert diffs['parameters']<=.01 and diffs['mac']<=.05
    for seed in SEEDS:
        c=Detector('C',seed=seed,widths=pick['widths']).state_dict()
        d=Detector('D',seed=seed,upstream=vm).state_dict()
        for k in c:
            if k.startswith(('rgb.','thermal.','head.')) and not torch.equal(c[k],d[k]):
                raise RuntimeError(f'Common initialization differs for {seed}/{k}')
    audit.update(budget_difference=diffs,common_init_equal_seeds=list(SEEDS),profiles={})
    audit['status']='COUNTS_PASS_PROFILE_PENDING'
    save()
    for arm,inp in [('A',dict(rgb=x)),('B',dict(thermal=y)),('C',paired),('D',paired)]:
        print('profiling',arm,flush=True)
        # Only the model under measurement occupies the device.
        for key,m in models.items():
            if key!=arm:
                m.cpu()
        models[arm].to(device)
        original=core.bwd
        backward=[]
        def trace_backward(u,*args,**kwargs):
            backward.append(list(u.shape))
            return original(u,*args,**kwargs)
        if arm=='D':
            core.bwd=trace_backward
        try:
            audit['profiles'][arm]=profile(models[arm],inp,core if arm=='D' else None)
        finally:
            if arm=='D':
                core.bwd=original
        if arm=='D':
            expected=[r['shape'][1:] for r in D['scan_calls']]
            for batch,offset in [(4,0),(2,45)]:
                for i in range(5):
                    observed=backward[offset+9*i:offset+9*(i+1)]
                    if sorted(observed)!=sorted([[batch,*shape] for shape in expected]):
                        raise RuntimeError('Selective Scan backward topology differs')
            if len(backward)!=90:
                raise RuntimeError('Unexpected Selective Scan backward count')
            audit['profiles'][arm]['native_cuda_backward_calls']=len(backward)
    audit['five_seed_all_arms_compute_only_hours']=sum(
        p['five_seed_compute_only_hours'] for p in audit['profiles'].values())
    audit['status']='PASS_SYNTHETIC_ONLY'
    save()
    print('PASS synthetic GPU A/B/C/D; no optimizer updates',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.upstream,args.output)
