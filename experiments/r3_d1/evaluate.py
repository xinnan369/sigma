"""Validation only: original-grid GT, frozen decode and pinned COCO evaluator."""
import torch

from . import coco
from .checkpoint import digest
from .fcos import decode
from .plan import checkpoint_key


@torch.inference_mode()
def validate(model,data,device):
    model.eval()
    images,annotations,predictions=[],[],[]
    for ident in data.ids('validation'):
        r,t,_=data.read_pair(ident,'validation')
        boxes=data.read_boxes(ident,'validation')  # original XML coordinates, complete GT
        i=int(ident)
        images.append(dict(id=i,width=1280,height=1024))
        for x,y,u,v in boxes.tolist():
            annotations.append(dict(id=len(annotations)+1,image_id=i,category_id=1,
                                    bbox=[x,y,u-x,v-y],area=(u-x)*(v-y),iscrowd=0))
        pred=model(rgb=r[None].to(device) if model.arm!='B' else None,
                   thermal=t[None].to(device) if model.arm!='A' else None)
        if any(not torch.isfinite(p).all() for p in pred):
            raise ValueError('Nonfinite Validation prediction')
        predictions.extend(coco.records(i,decode(pred)[0]))
    result=coco.evaluate(images,annotations,predictions)
    if any(not 0<=result[k]<=1 for k in ('ap50','ap50_95')):
        raise ValueError('Invalid Validation metrics')
    return dict(**result,predictions_sha256=digest(predictions),predictions=len(predictions),
                images=len(images),gt=len(annotations))


def improves(metrics,epoch,best):
    return best is None or checkpoint_key(metrics,epoch)>checkpoint_key(best,best['epoch'])
