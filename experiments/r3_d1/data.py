"""Fit/Validation-only data adapter with frozen synchronous geometry."""
import hashlib

import torch

from experiments.r3_d0.llvip import read_development_pair, transform

SPLIT_SHA256='bd1090cb2ad4696db3ba2f8e984788204c433a7e0cab656073c5e2b67e53e3d4'
MEAN=(.485,.456,.406)
STD=(.229,.224,.225)


def normalize(a):
    x=torch.from_numpy(a.copy()).float()
    x=x.permute(2,0,1) if x.ndim==3 else x.unsqueeze(0).repeat(3,1,1)
    return (x.div(255)-x.new_tensor(MEAN)[:,None,None])/x.new_tensor(STD)[:,None,None]


def read_pair(root,split_path,image_id,role,*,seed=3407,epoch=0):
    import json
    raw=split_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SPLIT_SHA256:
        raise ValueError('Frozen D0 split changed')
    rgb,t,boxes,_=read_development_pair(root,image_id,role,json.loads(raw))
    flip=role=='fit' and hashlib.sha256(f'{seed}/{epoch}/{image_id}/flip'.encode()).digest()[0]<128
    rgb,t,boxes=transform(rgb,t,boxes,flip)
    return normalize(rgb),normalize(t),torch.from_numpy(boxes)
