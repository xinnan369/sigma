"""Check traced full-detector operations; scan counts are semantic, not kernel FLOPs."""
from collections import Counter
from contextlib import contextmanager

import torch
from torch import nn


def parameters(model):
    return sum(p.numel() for p in model.parameters())


@contextmanager
def count_ops(model, core=None):
    count=Counter()
    handles=[]
    def hook(m, args, output):
        if isinstance(m,nn.Conv2d):
            count['conv_mac']+=output.numel()*m.weight[0].numel()
        elif isinstance(m,nn.Linear):
            count['linear_mac']+=output.numel()*m.in_features
        elif isinstance(m,nn.AdaptiveAvgPool2d):
            count['pool_input_elements']+=args[0].numel()
        else:
            count[type(m).__name__+'_elements']+=output.numel()
        if isinstance(m,(nn.Conv2d,nn.Linear)) and m.bias is not None:
            count['bias_add']+=output.numel()
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.Linear,nn.AdaptiveAvgPool2d,nn.BatchNorm2d,
                         nn.LayerNorm,nn.GELU,nn.SiLU,nn.Sigmoid)):
            handles.append(m.register_forward_hook(hook))
    orig=core.fwd if core is not None else None
    scans=[]
    if core is not None:
        def track(u, dt, a, b, c, d, *args):
            assert u.shape==dt.shape and a.shape[0]==u.shape[1] and a.shape[1]==4
            assert d is not None
            batch,channels,steps=u.shape
            states=a.shape[-1]
            values=batch*channels*steps
            updates=values*states
            scans.append(dict(shape=list(u.shape),group_count=b.shape[1],state=states))
            count['scan_state_updates']+=updates
            count['scan_mac']+=2*updates+values  # recurrence, readout and direct skip
            count['scan_exp']+=updates
            count['scan_A_exp']+=channels*states
            count['scan_input_multiply']+=3*updates  # dt*A and dt*B*x
            count['scan_delta_softplus']+=values
            count['scan_delta_bias_add']+=values
            return orig(u,dt,a,b,c,d,*args)
        core.fwd=track
    try:
        yield count,scans
    finally:
        if core is not None:
            core.fwd=orig
        for h in handles:
            h.remove()


def total_mac(c):
    return sum(v for k,v in c.items() if k.endswith('_mac'))


def complete_count(model, inputs, core=None):
    with count_ops(model,core) as (count,scans),torch.no_grad():
        output=model(**inputs)
    if core is not None:
        expected=[]
        for hw in (80,40,20):
            n=hw*hw
            expected.extend([[1,128,n],[1,128,n],[1,256,2*n]])
            # Cross dt products and Concat's two einsums (R=4, N=4, K=2).
            count['functional_projection_mac']+=2*n*128*4+2*2*n*128*(12+4)
            count['fusion_residual_merge_add']+=n*(4*64+4*128)
            count['fusion_gate_multiply']+=n*2*128
        if [r['shape'] for r in scans]!=expected or [r['group_count'] for r in scans]!=[1,1,2]*3:
            raise ValueError(f'Unexpected upstream scan topology: {scans}')
    elif model.arm=='C':
        count['fusion_residual_merge_add']=4*64*sum(n*n for n in (80,40,20))
    if [tuple(x.shape) for x in output]!=[(1,6,n,n) for n in (80,40,20)]:
        raise ValueError('Detector feature grids changed')
    return dict(parameters=parameters(model),mac=total_mac(count),ops=dict(count),scan_calls=scans)


def match(target, common_parameters, common_mac, allowed=range(16,513)):
    """Exhaustive a,b; solve the monotone admissible integer range for c."""
    import math
    pp,mm=target['parameters'],target['mac']
    # Width counts per scale: two 1x1 projections each, one grouped 3x3 each.
    param,opt=470,466
    constant=common_parameters+3*192
    allowed=tuple(allowed)
    if not allowed or allowed!=tuple(range(allowed[0],allowed[-1]+1)):
        raise ValueError('Expected contiguous width range')
    best=None
    for a in allowed:
        for b in allowed:
            start=constant+param*(a+b)
            mac=common_mac+opt*(6400*a+1600*b)
            lo=max(allowed[0],math.ceil((.99*pp-start)/param),
                   math.ceil((.95*mm-mac)/(400*opt)))
            hi=min(allowed[-1],math.floor((1.01*pp-start)/param),
                   math.floor((1.05*mm-mac)/(400*opt)))
            if lo>hi:
                continue
            choice=max(lo,min(hi,round((pp-start)/param)))
            p,m=start+param*choice,mac+400*opt*choice
            key=(abs(p/pp-1),abs(m/mm-1),a,b,choice)
            if best is None or key<best[0]:
                best=(key,dict(widths=[a,b,choice],parameters=p,mac=m))
    return None if best is None else best[1]
