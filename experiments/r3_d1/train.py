"""Explicitly authorized single-arm LLVIP runner. Importing never starts a run."""
import argparse
import math
import os
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import time

import cv2
import numpy as np
import torch

from .authorization import require_authorization
from .deterministic_scan import VERSION, installed
from .checkpoint import Checkpoints, Events, digest, hashfile, run_lock, write_json
from .data import DevelopmentData, SPLIT_SHA256, flipped
from .evaluate import improves, validate
from .model import Detector, SEEDS
from .plan import EPOCHS, FIT_COUNT, WIDTHS, epoch_indices, optimizer, training_forward
from .preflight import fingerprint
from .upstream import load_sigma

ROOT=Path(__file__).resolve().parents[2]
UPDATES=2482
VALIDATION_COUNT=2087
INTERVAL=100


def external(path):
    dest=Path(path).resolve()
    if dest.is_relative_to(ROOT):
        raise ValueError('Output must be outside repository')
    return dest


def reproducible(seed,cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if cuda:
        torch.cuda.manual_seed_all(seed)


def runtime(cuda):
    result=dict(python=platform.python_version(),platform=platform.platform(),
                torch=str(torch.__version__),numpy=np.__version__,opencv=cv2.__version__,
                pycocotools=version('pycocotools'),cuda_build=torch.version.cuda,
                deterministic=torch.are_deterministic_algorithms_enabled(),
                cudnn=torch.backends.cudnn.version(),cpu_threads=torch.get_num_threads(),
                cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS',
                                                           'OPENBLAS_NUM_THREADS')})
    if cuda:
        p=torch.cuda.get_device_properties(0)
        result['gpu']=dict(name=p.name,capability=[p.major,p.minor],bytes=p.total_memory,
                           visible_devices=torch.cuda.device_count(),
                           driver=driver())
    return result


def driver():
    import subprocess
    return subprocess.check_output(['nvidia-smi','--query-gpu=driver_version',
                                    '--format=csv,noheader'],text=True).strip()


def signature(seed,arm,source,env):
    # The independently reviewed allowlist must not hash the receipt it authorizes.
    files=[p for p in Path(__file__).parent.rglob('*.py') if p.name!='review_registry.py']+[
        ROOT/'docs/R3_D1_DETECTION_PROTOCOL.md',ROOT/'experiments/r3_d0/llvip.py']
    return dict(schema='r3-d1b-v1',seed=seed,arm=arm,cuda=True,split_sha256=SPLIT_SHA256,
                upstream=source,environment=env,deterministic_backend=VERSION,precision='fp32',optimizer='AdamW',
                lr=.0003,betas=[.9,.999],eps=1e-8,conv_decay=.01,other_decay=0.,
                scheduler=None,epochs=EPOCHS,batch=4,updates_per_epoch=UPDATES,
                fit=FIT_COUNT,validation=VALIDATION_COUNT,cnn_widths=list(WIDTHS),
                source_sha256={p.relative_to(ROOT).as_posix():hashfile(p) for p in sorted(files)})


def batches(seed,epoch,roster):
    if not 0<=epoch<EPOCHS:
        return []
    indexes=epoch_indices(seed,epoch)
    if len(roster)!=FIT_COUNT or len(indexes)!=UPDATES:
        raise ValueError('Frozen Fit size or update count changed')
    groups=[[roster[i] for i in ix] for ix in indexes]
    if [len(b) for b in groups]!=[4]*(UPDATES-1)+[2]:
        raise ValueError('Frozen batch shape changed')
    used=[i for batch in groups for i in batch]
    if len(set(used))!=FIT_COUNT or set(used)!=set(roster):
        raise ValueError('Fit order repeats or misses samples')
    return groups


def new_progress(seed,arm,epoch,roster,history=None,best=None):
    order=batches(seed,epoch,roster)
    return dict(seed=seed,arm=arm,epoch=epoch,cursor=0,update=epoch*UPDATES,
                order=order,order_sha256=digest(order),history=[] if history is None else history,
                best=best,loss_sum=dict(cls=0.,box=0.,center=0.),images=0)


def verify_progress(p,seed,arm,roster):
    e,c,u=p['epoch'],p['cursor'],p['update']
    if (p['seed']!=seed or p['arm']!=arm or not 0<=e<=EPOCHS
            or not 0<=c<=(UPDATES if e<EPOCHS else 0) or u!=e*UPDATES+c):
        raise ValueError('Invalid training cursor')
    order=batches(seed,e,roster)
    if p['order']!=order or p['order_sha256']!=digest(order):
        raise ValueError('Sample order differs')
    h=p['history']
    if [v['epoch'] for v in h]!=list(range(e)) or p['images']!=sum(map(len,order[:c])):
        raise ValueError('Invalid epoch history or sample count')
    if set(p['loss_sum'])!={'cls','box','center'} or any(not math.isfinite(v) for v in p['loss_sum'].values()):
        raise ValueError('Invalid committed loss sums')
    best=None
    for row in h:
        if improves(row,row['epoch'],best):
            best={k:row[k] for k in ('epoch','ap50','ap50_95')}
    if p['best']!=best:
        raise ValueError('Best history differs')


def optimizer_cursor(opt,model,update):
    state=opt.state
    # All parameters receive a gradient, including empty-GT batches.
    if len(state)!=(len(list(model.parameters())) if update else 0):
        raise ValueError('AdamW state coverage differs')
    if any(float(state[p]['step'])!=update for p in state):
        raise ValueError('AdamW step differs from global update')
    for g,decay in zip(opt.param_groups,(0.,.01),strict=True):
        if (g['lr'],g['betas'],g['eps'],g['weight_decay'])!=(.0003,(.9,.999),1e-8,decay):
            raise ValueError('AdamW hyperparameters differ')


def update(model,opt,rgb,thermal,boxes):
    opt.zero_grad(set_to_none=True)
    parts=training_forward(model,rgb,thermal,boxes)
    total=sum(parts.values())
    if not torch.isfinite(total):
        raise ValueError('Nonfinite training loss')
    total.backward()
    grad=[p.grad for p in model.parameters()]
    if any(g is None for g in grad):
        raise ValueError('Missing trainable gradient')
    grad_norm=float(torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(g.detach()) for g in grad])))
    if not math.isfinite(grad_norm):
        raise ValueError('Nonfinite gradient norm')
    opt.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite updated parameter')
    opt.zero_grad(set_to_none=True)
    return dict(loss={k:float(v.detach()) for k,v in parts.items()},
                gradient_l2=grad_norm,lr=opt.param_groups[0]['lr'])


def train(args):
    if not args.authorize_formal_training:
        raise ValueError('Formal run requires explicit authorization flag')
    require_authorization(args.authorization,args.arm,args.seed)
    out=external(args.output)/args.arm/str(args.seed)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('Expose exactly one CUDA GPU')
    reproducible(args.seed,True)
    env=runtime(True)
    if env['cublas_workspace'] not in (':4096:8',':16:8'):
        raise ValueError('Set CUBLAS_WORKSPACE_CONFIG before starting')
    vm,core,source=load_sigma(args.upstream)
    sig=signature(args.seed,args.arm,source,env)
    require_authorization(args.authorization,args.arm,args.seed,sig)
    data=DevelopmentData(args.root,args.split)
    roster=data.ids('fit')
    if len(data.ids('validation'))!=VALIDATION_COUNT:
        raise ValueError('Frozen validation count differs')
    with run_lock(out), installed(core):
        ledger=out/'checkpoint-ledger.json'
        if args.resume!=ledger.exists() or (not args.resume and any(
                p.name!='run.lock' for p in out.iterdir())):
            raise ValueError('Explicit resume requires a committed checkpoint; fresh output must be empty')
        device=torch.device('cuda:0')
        m=Detector(args.arm,seed=args.seed,widths=WIDTHS,upstream=vm).to(device)
        opt=optimizer(m)
        init=dict(signature_sha256=digest(sig),model_state_sha256=fingerprint(m.state_dict()))
        with Events(out/'events.jsonl',resume=args.resume) as events:
            ck=Checkpoints(out)
            if args.resume:
                if (json.loads((out/'initialization.json').read_bytes())!=init or
                        json.loads((out/'runtime.json').read_bytes())!=env):
                    raise ValueError('Run metadata differs')
                p=ck.restore(m,opt,sig,events,lambda p:verify_progress(p,args.seed,args.arm,roster),
                             lambda p:optimizer_cursor(opt,m,p['update']))
                write_json(out/'resume-audit.json',dict(status='RESTORED',epoch=p['epoch'],
                           cursor=p['cursor'],update=p['update'],signature_sha256=digest(sig)))
            else:
                p=new_progress(args.seed,args.arm,0,roster)
                write_json(out/'initialization.json',init)
                write_json(out/'runtime.json',env)
                write_json(out/'resume-audit.json',dict(status='NOT_RESUMED'))
            write_json(out/'epochs.json',p['history'])
            while p['epoch']<EPOCHS:
                e=p['epoch']
                m.train()
                while p['cursor']<len(p['order']):
                    t=time.perf_counter()
                    ids=p['order'][p['cursor']]
                    rows=[data.read_pair(i,'fit',seed=args.seed,epoch=e) for i in ids]
                    r,h,b=zip(*rows,strict=True)
                    torch.cuda.reset_peak_memory_stats()
                    metrics=update(m,opt,torch.stack(r).to(device) if args.arm!='B' else None,
                                   torch.stack(h).to(device) if args.arm!='A' else None,
                                   [v.to(device) for v in b])
                    torch.cuda.synchronize()
                    p['cursor']+=1
                    p['update']+=1
                    p['images']+=len(ids)
                    for k,v in metrics['loss'].items():
                        p['loss_sum'][k]+=v*len(ids)
                    events.append(dict(kind='update',epoch=e,batch_cursor=p['cursor'],
                                      update=p['update'],ids=ids,
                                      flips=[flipped(i,'fit',args.seed,e) for i in ids],
                                      seconds=time.perf_counter()-t,
                                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),**metrics))
                    if p['update']%INTERVAL==0:
                        ck.commit(m,opt,sig,p,events)
                # A last-batch checkpoint can have cursor==2482; resume then validates it.
                t=time.perf_counter()
                val=validate(m,data,device)
                row=dict(epoch=e,update=p['update'],validation_seconds=time.perf_counter()-t,
                         training_loss={k:v/p['images'] for k,v in p['loss_sum'].items()},**val)
                better=improves(val,e,p['best'])
                best={k:row[k] for k in ('epoch','ap50','ap50_95')} if better else p['best']
                events.append(dict(kind='epoch',**row))
                p=new_progress(args.seed,args.arm,e+1,roster,p['history']+[row],best)
                ck.commit(m,opt,sig,p,events,best=better)
                write_json(out/'epochs.json',p['history'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('root','split','upstream','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--arm',choices=list('ABCD'),required=True)
    parser.add_argument('--authorization',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--authorize-formal-training',action='store_true')
    train(parser.parse_args())
