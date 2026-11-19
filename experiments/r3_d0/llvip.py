"""Local LLVIP detection reader. The official test is not a development split."""
import hashlib
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np

SOURCE_WH = (1280, 1024)
TARGET_WH = (640, 640)


def parse_voc(data, image_id, wh=SOURCE_WH):
    root = ET.fromstring(data)
    if (root.tag != 'annotation' or root.findtext('filename') != f'{image_id}.jpg'
            or tuple(int(root.findtext('size/'+k)) for k in ('width', 'height')) != wh):
        raise ValueError('VOC identity or dimensions differ')
    boxes, flags, invalid = [], [], []
    for idx, obj in enumerate(root.findall('object')):
        if obj.findtext('name') != 'person':
            raise ValueError('Unexpected class')
        box = [int(obj.findtext('bndbox/'+k)) for k in ('xmin','ymin','xmax','ymax')]
        difficult = int(obj.findtext('difficult', '0'))
        if difficult not in (0, 1):
            raise ValueError('Invalid difficult flag')
        x0,y0,x1,y1 = box
        if not (0 <= x0 < x1 <= wh[0] and 0 <= y0 < y1 <= wh[1]):
            invalid.append({'object_index_zero_based':idx, 'xyxy':box})
        else:
            boxes.append(box)
            flags.append(difficult)
    return np.asarray(boxes, dtype=np.float32).reshape(-1,4), flags, invalid


def transform(rgb, thermal, boxes, flip=False):
    if any(a.dtype != np.uint8 or a.shape != shape for a, shape in
           [(rgb,(1024,1280,3)), (thermal,(1024,1280))]):
        raise ValueError('Unexpected paired image grid or dtype')
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1,4).copy()
    if (not np.isfinite(boxes).all() or (boxes[:,:2]<0).any()
            or (boxes[:,2:] > SOURCE_WH).any() or (boxes[:,2:]<=boxes[:,:2]).any()):
        raise ValueError('Invalid source box')
    def image(a):
        a = cv2.resize(a,(640,512),interpolation=cv2.INTER_AREA)
        a = np.pad(a,((64,64),(0,0))+((0,0),)*(a.ndim-2),constant_values=114)
        return np.ascontiguousarray(a[:,::-1] if flip else a)
    boxes *= .5
    boxes[:,1::2] += 64
    if flip:
        boxes[:,[0,2]] = 640-boxes[:,[2,0]]
    return image(rgb),image(thermal),boxes


def source_xyxy(boxes):
    """Invert the evaluation transform and clip to the original image bounds."""
    b=np.asarray(boxes,dtype=np.float32).reshape(-1,4).copy()
    b[:,1::2]-=64
    return np.clip(b*2,0,[1280,1024,1280,1024])


def read_development_pair(root, image_id, role, split):
    if (role not in ('fit','validation') or not re.fullmatch(r'[0-9]{6}', image_id)
            or image_id not in split[role]):
        raise ValueError('ID is not admitted for this development role')
    def raw(rel):
        data=(root/rel).read_bytes()
        if hashlib.sha256(data).hexdigest()!=split['development_sha256'][image_id][rel]:
            raise ValueError(f'Frozen file hash differs: {rel}')
        return data
    flags=cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    rgb,t=(cv2.imdecode(np.frombuffer(raw(f'{m}/train/{image_id}.jpg'),np.uint8),flags)
           for m in ('visible','infrared'))
    if rgb is None or t is None:
        raise ValueError('JPEG decode failed')
    boxes,difficult,invalid=parse_voc(raw(f'Annotations/{image_id}.xml'),image_id)
    if invalid:
        raise ValueError(f'Invalid annotation: {invalid}')
    return rgb[:,:,::-1].copy(),cv2.cvtColor(t,cv2.COLOR_BGR2GRAY),boxes,difficult
