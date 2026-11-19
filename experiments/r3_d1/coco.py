"""Version-locked single-class COCO bbox evaluation on supplied annotations only."""
from importlib.metadata import version

import numpy as np


def evaluate(images, annotations, predictions):
    if version('pycocotools')!='2.0.10':
        raise RuntimeError('Expected pycocotools==2.0.10')
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt=COCO()
    gt.dataset=dict(info={}, images=list(images), annotations=list(annotations),
                    categories=[dict(id=1,name='person')])
    gt.createIndex()
    if predictions:
        # COCO loadRes mutates dictionaries.
        dt=gt.loadRes([dict(p) for p in predictions])
    else:
        dt=COCO()
        dt.dataset=dict(images=list(images),annotations=[],categories=gt.dataset['categories'])
        dt.createIndex()
    if not images or not annotations:
        raise ValueError('COCO AP requires images and positive support')
    runner=COCOeval(gt,dt,'bbox')
    runner.params.catIds=[1]
    runner.params.iouThrs=np.linspace(.5,.95,10)
    runner.params.recThrs=np.linspace(0,1,101)
    runner.params.maxDets=[1,10,100]
    runner.evaluate()
    runner.accumulate()
    runner.summarize()
    return dict(ap50=float(runner.stats[1]),ap50_95=float(runner.stats[0]))


def records(image_id, prediction):
    return [dict(image_id=image_id, category_id=1, score=float(score),
                 bbox=[float(x),float(y),float(u-x),float(v-y)])
            for (x,y,u,v),score in zip(prediction['boxes'],prediction['scores'],strict=True)]
