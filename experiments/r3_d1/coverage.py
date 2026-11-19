"""Read-only FCOS coverage and input receipts; no test paths or images in coverage."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform

import cv2
import numpy as np
import torch

from .data import (DevelopmentData, SPLIT_SHA256, flipped, prepare_pair,
                   transform_boxes)
from .fcos import assignments, grid
from .model import STRIDES

BINS=(0,8,16,32,64,128,float('inf'))


def sha(b):
    return hashlib.sha256(b).hexdigest()


def points_by_level():
    return [grid(torch.empty(n,n,device='cpu'),s) for n,s in zip((80,40,20),STRIDES)]


def coverage(boxes, points):
    wins=torch.zeros(len(boxes),dtype=torch.long)
    eligible=torch.zeros_like(wins)
    levels=[]
    for level,xy in enumerate(points):
        index,possible=assignments(xy,boxes,level,audit=True)
        chosen=index[index>=0]
        wins+=torch.bincount(chosen,minlength=len(boxes))
        eligible+=possible
        levels.append(len(chosen))
    return wins,eligible,levels


def audit(data,out):
    points=points_by_level()
    stats={}
    with (out/'gt-per-image.jsonl').open('x',encoding='utf-8',newline='\n') as file:
        for role in ('fit','validation'):
            totals=Counter()
            size=Counter()
            misses=Counter()
            missed_sizes=[]
            examples=[]
            for i in data.ids(role):
                src=data.read_boxes(i,role)
                current=flipped(i,role)
                b=transform_boxes(src,current)
                wins,eligible,levels=coverage(b,points)
                if role=='fit':
                    # Verify that future synchronized flip schedules do not alter coverage.
                    other=coverage(transform_boxes(src,not current),points)
                    if not torch.equal(wins,other[0]) or not torch.equal(eligible,other[1]) or levels!=other[2]:
                        raise RuntimeError(f'Unexpected flip dependence: {i}')
                wh=b[:,2:]-b[:,:2]
                sizes=wh.prod(1).sqrt()
                missed=[]
                for j,(n,e,scale) in enumerate(zip(wins.tolist(),eligible.tolist(),sizes.tolist(),strict=True)):
                    index=next(k for k in range(len(BINS)-1) if BINS[k]<=scale<BINS[k+1])
                    key=f'[{BINS[index]},{BINS[index+1]})'
                    size[key]+=1
                    if n==0:
                        misses[key]+=1
                        missed_sizes.append([*wh[j].tolist(),scale])
                        detail=dict(gt_index=j,xyxy=b[j].tolist(),sqrt_area=scale,
                                    reason='lost_tie_or_overlap' if e else 'no_eligible_point')
                        missed.append(detail)
                        totals[detail['reason']]+=1
                        if len(examples)<12:
                            examples.append(dict(id=i,**detail))
                num=len(src)
                totals.update(images=1,gt=num,matched=num-len(missed),unmatched=len(missed),
                              zero_positive_images=int(not any(levels)))
                per_stride=dict(zip(map(str,STRIDES),levels))
                for k,n in per_stride.items():
                    totals[f'positive_stride_{k}']+=n
                file.write(json.dumps(dict(id=i,role=role,flip_seed3407_epoch0=current,gt=num,
                                   matched=num-len(missed),positive_by_stride=per_stride,
                                   unmatched=missed),separators=(',',':'))+'\n')
            def q(axis):
                return np.percentile(np.array(missed_sizes)[:,axis],[0,25,50,75,100]).tolist() if missed_sizes else []
            stats[role]=dict(totals=dict(totals),by_sqrt_area={
                k:dict(gt=n,matched=n-misses[k],unmatched=misses[k],match_rate=(n-misses[k])/n)
                for k,n in size.items()},unmatched_size_quantiles={
                'percentiles':[0,25,50,75,100],'width':q(0),'height':q(1),'sqrt_area':q(2)},
                representative_unmatched=examples)
            print(role,dict(totals),flush=True)
    write(out/'gt-coverage.json',dict(status='MEASURED_UNCHANGED_PROTOCOL',
          split_sha256=SPLIT_SHA256,geometry='train seed 3407 epoch 0; both flips verified per Fit GT; Validation no flip',
          official_test_reads=0,image_pixel_reads=0,environment=environment(),roles=stats))


def environment():
    return dict(python=platform.python_version(),torch=torch.__version__,
                numpy=np.__version__,opencv=cv2.__version__,cpu_threads=torch.get_num_threads())


def fingerprint(x):
    a=x.detach().cpu().contiguous().numpy() if isinstance(x,torch.Tensor) else np.ascontiguousarray(x)
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=sha(a.tobytes(order='C')))


def reading(data,out):
    rows=[]
    for role,i in [('fit','010001'),('validation','020001')]:
        r,t,b,_=data.read_raw(i,role)
        flip=flipped(i,role)
        x,y,z=prepare_pair(r,t,b,flip=flip)
        assert torch.equal(z,transform_boxes(b,flip))
        xx,yy,zz=data.read_pair(i,role)
        assert all(torch.equal(a,b) for a,b in ((x,xx),(y,yy),(z,zz)))
        rows.append(dict(role=role,id=i,file_sha256=data.hashes(i,role),flip_seed3407_epoch0=flip,
                         raw=dict(rgb=fingerprint(r),thermal=fingerprint(t)),gt=len(b),
                         transformed_boxes=z.tolist(),
                         normalized=dict(rgb=fingerprint(x),thermal=fingerprint(y),boxes=fingerprint(z))))
        print(role,i,len(b),'boxes PASS',flush=True)
    write(out/'read-receipt.json',dict(split_sha256=SPLIT_SHA256,official_test_reads=0,
                                      environment=environment(),rows=rows))


def write(path,obj):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(obj,f,indent=2,allow_nan=False)
        f.write('\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('coverage','read'))
    for arg in ('root','split','output'):
        p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    data=DevelopmentData(a.root,a.split)
    (audit if a.mode=='coverage' else reading)(data,a.output)
