"""Five-stage detector with either Sigma's two fusion blocks or a CNN control."""
import hashlib

import torch
from torch import nn

STRIDES=(8,16,32)
SEEDS=(3407,3408,3409,3410,3411)


def seed_for(seed, name):
    return int.from_bytes(hashlib.sha256(f'{seed}/{name}'.encode()).digest()[:8],'big')


def seeded(seed, name, builder):
    # Construct on CPU, excluding CUDA RNG. Preserve Sigma's native initializer.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed_for(seed,name))
        return builder()


def layer(a,b,stride=1):
    return nn.Sequential(nn.Conv2d(a,b,3,stride,1,bias=False),nn.BatchNorm2d(b),nn.SiLU())


def init_common(module):
    for m in module.modules():
        if isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return module


class Tower(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.ModuleList(layer(a,b,2) for a,b in zip((3,16,32,64,64),(16,32,64,64,64)))

    def forward(self,x):
        out=[]
        for i,l in enumerate(self.layers):
            x=l(x)
            if i>=2:
                out.append(x)
        return out


class CNNBlock(nn.Module):
    """WideCNN topology, matching upstream BHWC interfaces and residuals."""
    def __init__(self,width,cross):
        super().__init__()
        self.cross=cross
        self.mix=nn.Sequential(nn.Conv2d(128,width,1),
                               nn.Conv2d(width,width,3,padding=1,groups=width),
                               nn.GELU(),nn.Conv2d(width,128 if cross else 64,1))

    def forward(self,r,t):
        y=self.mix(torch.cat((r,t),dim=-1).permute(0,3,1,2)).permute(0,2,3,1)
        if self.cross:
            a,b=y.chunk(2,dim=-1)
            return r+a,t+b
        return r+t+y


class Detector(nn.Module):
    def __init__(self,arm,*,seed=3407,widths=None,upstream=None):
        super().__init__()
        if arm not in ('A','B','C','D') or seed not in SEEDS:
            raise ValueError('Unknown arm or unregistered seed')
        self.arm=arm
        self.rgb=seeded(seed,'rgb',lambda:init_common(Tower())) if arm!='B' else None
        self.thermal=seeded(seed,'thermal',lambda:init_common(Tower())) if arm!='A' else None
        self.head=seeded(seed,'head',lambda:init_common(nn.Sequential(layer(64,64),nn.Conv2d(64,6,1))))
        if arm=='C':
            if widths is None or len(widths)!=3 or any(w<1 for w in widths):
                raise ValueError('Three matched CNN widths required')
            self.cross=nn.ModuleList(seeded(seed,f'cross/{i}',lambda:init_common(CNNBlock(w,True)))
                                     for i,w in enumerate(widths))
            self.concat=nn.ModuleList(seeded(seed,f'concat/{i}',lambda:init_common(CNNBlock(w,False)))
                                      for i,w in enumerate(widths))
        elif arm=='D':
            if upstream is None:
                raise ValueError('Pinned Sigma modules required')
            kw=dict(hidden_dim=64,d_state=4,mlp_ratio=0.)
            self.cross=nn.ModuleList(seeded(seed,f'cross/{i}',lambda:upstream.CrossMambaFusionBlock(**kw))
                                     for i in range(3))
            self.concat=nn.ModuleList(seeded(seed,f'concat/{i}',lambda:upstream.ConcatMambaFusionBlock(**kw))
                                      for i in range(3))

    def features(self,rgb=None,thermal=None):
        if (rgb is None)!=(self.rgb is None) or (thermal is None)!=(self.thermal is None):
            raise ValueError('Provide only the modalities used by this arm')
        r=self.rgb(rgb) if self.rgb is not None else None
        t=self.thermal(thermal) if self.thermal is not None else None
        if self.arm in ('A','B'):
            return r if self.arm=='A' else t
        out=[]
        for a,b,c,f in zip(r,t,self.cross,self.concat,strict=True):
            a,b=c(a.permute(0,2,3,1).contiguous(),b.permute(0,2,3,1).contiguous())
            out.append(f(a,b).permute(0,3,1,2).contiguous())
        return out

    def forward(self,rgb=None,thermal=None):
        return [self.head(x) for x in self.features(rgb,thermal)]
