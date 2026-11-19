"""FCOS person target assignment, losses, and deterministic original-grid decoding."""
import torch
from torch.nn import functional as F

from .model import STRIDES


def grid(raw,stride):
    y,x=torch.meshgrid(*[torch.arange(n,device=raw.device,dtype=raw.dtype)
                        for n in raw.shape[-2:]],indexing='ij')
    return torch.stack(((x+.5)*stride,(y+.5)*stride),-1).reshape(-1,2)


def valid(points):
    return ((points[:,0]>=0)&(points[:,0]<640)&(points[:,1]>=64)&(points[:,1]<576))


def assignments(points, boxes, level, *, audit=False):
    """Return winning GT index per location (-1 for background) and eligible counts."""
    if boxes.numel()==0:
        return (torch.full((len(points),),-1,device=points.device,dtype=torch.long),
                torch.zeros(0,device=points.device,dtype=torch.long) if audit else None)
    boxes=boxes.to(points)
    xy0,xy1=boxes[:,:2],boxes[:,2:]
    wh=xy1-xy0
    if not torch.isfinite(boxes).all() or (wh<=0).any():
        raise ValueError('Invalid target box')
    scale=wh.prod(1).sqrt()
    assigned=(scale>= (0,64,128)[level])&(scale< (64,128,float('inf'))[level])
    radius=STRIDES[level]*1.5
    centers=(xy0+xy1)/2
    low=torch.maximum(xy0,centers-radius)
    high=torch.minimum(xy1,centers+radius)
    p=points[:,None,:]
    hit=((p>low)&(p<high)).all(-1)&assigned&valid(points)[:,None]
    sizes=wh.prod(1).expand(len(points),-1).masked_fill(~hit,float('inf'))
    best,index=sizes.min(1)  # tie: first annotation
    return index.masked_fill(~best.isfinite(),-1),hit.sum(0) if audit else None


def targets(points, boxes, level):
    index,_=assignments(points,boxes,level)
    pos=index>=0
    dist=points.new_zeros((len(points),4))
    boxes=boxes.to(points)[index[pos]]
    dist[pos]=torch.cat((points[pos]-boxes[:,:2],boxes[:,2:]-points[pos]),1)
    return pos,dist


def distances(raw,stride):
    return F.softplus(raw[...,1:5])*stride


def xyxy(points,ltrb):
    return torch.cat((points-ltrb[...,:2],points+ltrb[...,2:]),-1)


def giou(a,b):
    wh=lambda tl,br:(br-tl).clamp_min(0).prod(-1)
    union=wh(a[:,:2],a[:,2:])+wh(b[:,:2],b[:,2:])
    intersection=wh(torch.maximum(a[:,:2],b[:,:2]),torch.minimum(a[:,2:],b[:,2:]))
    union=union-intersection
    hull=wh(torch.minimum(a[:,:2],b[:,:2]),torch.maximum(a[:,2:],b[:,2:]))
    return intersection/union.clamp_min(1e-8)-(hull-union)/hull.clamp_min(1e-8)


def losses(predictions, boxes):
    if len(predictions)!=3 or not boxes or any(x.shape[:2]!=(len(boxes),6) for x in predictions):
        raise ValueError('Expected 3 FCOS levels and a nonempty aligned batch')
    totals=dict(cls=0.,box=0.,center=0.)
    points=[grid(p,s) for p,s in zip(predictions,STRIDES)]
    for i,b in enumerate(boxes):
        # Keep each output connected even for an empty image.
        zero=sum(p[i].sum()*0 for p in predictions)
        cls=reg=ct=zero
        n=0
        csum=zero
        for level,(p,s,xy) in enumerate(zip(predictions,STRIDES,points)):
            q=p[i].permute(1,2,0).reshape(-1,6)
            pos,gt=targets(xy,b,level)
            keep=valid(xy)
            logit,y=q[keep,0],pos[keep].to(q)
            prob=logit.sigmoid()
            pt=torch.where(y.bool(),prob,1-prob)
            alpha=torch.where(y.bool(),.25,.75)
            cls=cls+(F.binary_cross_entropy_with_logits(logit,y,reduction='none')*alpha*(1-pt)**2).sum()
            n+=int(pos.sum())
            d=gt[pos]
            if len(d):
                lo=torch.minimum(d[:,:2],d[:,2:])
                hi=torch.maximum(d[:,:2],d[:,2:])
                cent=(lo/hi).prod(1).sqrt()
                reg=reg+((1-giou(xyxy(xy[pos],distances(q[pos],s)),xyxy(xy[pos],d)))*cent).sum()
                csum=csum+cent.sum()
                ct=ct+F.binary_cross_entropy_with_logits(q[pos,5],cent,reduction='sum')
        for key,l in [('cls',cls/max(1,n)),('box',reg/torch.clamp(csum,min=1e-8)),('center',ct/max(1,n))]:
            totals[key]=totals[key]+l/len(boxes)
    return totals


def nms(boxes,scores,threshold=.65,limit=100):
    order=torch.argsort(scores,descending=True,stable=True)
    kept=[]
    area=(boxes[:,2:]-boxes[:,:2]).prod(1)
    while len(order) and len(kept)<limit:
        idx=order[0]
        kept.append(idx)
        rest=order[1:]
        inter=(torch.minimum(boxes[idx,2:],boxes[rest,2:])-
               torch.maximum(boxes[idx,:2],boxes[rest,:2])).clamp_min(0).prod(1)
        order=rest[inter/(area[idx]+area[rest]-inter).clamp_min(1e-8)<=threshold]
    return torch.stack(kept) if kept else order


@torch.no_grad()
def decode(predictions):
    out=[]
    for i in range(predictions[0].shape[0]):
        boxes,scores=[],[]
        for pred,stride in zip(predictions,STRIDES,strict=True):
            # CPU reference NMS avoids repeated GPU synchronization inside greedy NMS.
            p=pred[i:i+1].float().cpu()
            xy=grid(p,stride)
            q=p[0].permute(1,2,0).reshape(-1,6)
            keep=valid(xy)
            xy,q=xy[keep],q[keep]
            b=xyxy(xy,distances(q,stride))
            b[:,1::2]-=64
            b=b*2
            b[:,0::2].clamp_(0,1280)
            b[:,1::2].clamp_(0,1024)
            mask=(b[:,2:]>b[:,:2]).all(1)&torch.isfinite(q).all(1)
            boxes.append(b[mask])
            scores.append((q[mask,0].sigmoid()*q[mask,5].sigmoid()).sqrt())
        b,s=torch.cat(boxes),torch.cat(scores)
        ids=nms(b,s)
        out.append(dict(boxes=b[ids],scores=s[ids]))
    return out
