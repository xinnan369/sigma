"""Versioned deterministic FP32 backward built on the pinned CUDA scan core."""
from contextlib import contextmanager

import torch

VERSION='sigma-r3-exclusive-block-pairwise-v1'


def fixed_sum(x):
    """Pair neighbors then carry the last odd entry; never use a shared accumulator."""
    if not len(x):
        raise ValueError('Empty reduction axis')
    while len(x)>1:
        pairs=len(x)//2
        y=x[:2*pairs:2]+x[1:2*pairs:2]
        x=torch.cat((y,x[-1:]),dim=0) if len(x)%2 else y
    return x[0]


def backward(original,u,delta,a,b,c,d,bias,dout,saved,softplus,nrows):
    """Assign every atomic destination to one original (sample, channel) block.

    The forward is untouched. Here each native launch sees B=1 and one B/C group
    per channel, so its atomic output slots have exactly one owning CUDA block.
    Reduce expanded group gradients across channels, then shared parameter
    gradients across samples with a fixed binary tree of CUDA pointwise adds.
    """
    bs,dim,length=u.shape
    groups=b.shape[1]
    state=a.shape[1]
    xs=(u,delta,a,b,c,d,bias,dout,saved)
    if (nrows!=1 or bs<1 or state!=4 or dim%groups or not softplus
            or c.shape!=b.shape or b.shape!=(bs,groups,state,length)
            or any(t is None or not t.is_cuda or t.dtype!=torch.float32 or
                   t.device!=u.device for t in xs)):
        raise ValueError('Deterministic scan supports the pinned detector FP32 ABI only')
    ratio=dim//groups
    du,ddelta,db,dc,da,ds,dbias=([] for _ in range(7))
    for i in range(bs):
        j=slice(i,i+1)
        expanded_b=b[j].repeat_interleave(ratio,dim=1).contiguous()
        expanded_c=c[j].repeat_interleave(ratio,dim=1).contiguous()
        v,w,ap,bp,cp,dp,qp=original(u[j],delta[j],a,expanded_b,expanded_c,d,bias,
                                     dout[j],saved[j],softplus,1)
        du.append(v); ddelta.append(w)
        da.append(ap); ds.append(dp); dbias.append(qp)
        # [groups, channel-within-group, state, time] -> fixed channel tree
        def collapse(t):
            return fixed_sum(t.reshape(groups,ratio,state,length).transpose(0,1)).unsqueeze(0)
        db.append(collapse(bp))
        dc.append(collapse(cp))
        del expanded_b,expanded_c,bp,cp
    return [torch.cat(du),torch.cat(ddelta),fixed_sum(torch.stack(da)),
            torch.cat(db),torch.cat(dc),fixed_sum(torch.stack(ds)),fixed_sum(torch.stack(dbias))]


@contextmanager
def installed(core):
    """Process-local, single-threaded harness; always restore the original callable."""
    original=core.bwd
    def dispatch(*args,**kwargs):
        return backward(original,*args,**kwargs)
    core.bwd=dispatch
    try:
        yield
    finally:
        core.bwd=original
