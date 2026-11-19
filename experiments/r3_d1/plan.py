"""Frozen training rules and paired batch integrator; no executable training job."""
import numpy as np
import torch
from torch import nn

from .fcos import losses
from .model import SEEDS

BATCH=4
EPOCHS=10
FIT_COUNT=9926
WIDTHS=(158,370,18)


def epoch_indices(seed, epoch):
    if seed not in SEEDS or not 0<=epoch<EPOCHS:
        raise ValueError('Outside frozen seed/epoch scope')
    order=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,epoch]))).permutation(FIT_COUNT)
    return [order[i:i+BATCH] for i in range(0,FIT_COUNT,BATCH)]


def optimizer(model):
    conv={id(m.weight) for m in model.modules() if isinstance(m,nn.Conv2d)}
    groups=[[],[]]
    for p in model.parameters():
        if p.requires_grad:
            groups[int(id(p) in conv)].append(p)
    return torch.optim.AdamW([dict(params=g,weight_decay=w) for g,w in zip(groups,(0.,.01))],
                             lr=3e-4,betas=(.9,.999),eps=1e-8)


def training_forward(model,rgb,thermal,boxes):
    inputs={'rgb':rgb if model.arm!='B' else None,
            'thermal':thermal if model.arm!='A' else None}
    return losses(model(**inputs),boxes)


def checkpoint_key(metrics, epoch):
    return (metrics['ap50_95'], metrics['ap50'], -epoch)
