"""Fit/Validation-only data adapter with frozen synchronous geometry."""
import hashlib
import json
import re
from pathlib import Path

import torch

from experiments.r3_d0.llvip import parse_voc, read_development_pair, transform

SPLIT_SHA256='bd1090cb2ad4696db3ba2f8e984788204c433a7e0cab656073c5e2b67e53e3d4'
MEAN=(.485,.456,.406)
STD=(.229,.224,.225)


def normalize(a):
    x=torch.from_numpy(a.copy()).float()
    x=x.permute(2,0,1) if x.ndim==3 else x.unsqueeze(0).repeat(3,1,1)
    return (x.div(255)-x.new_tensor(MEAN)[:,None,None])/x.new_tensor(STD)[:,None,None]


class DevelopmentData:
    """One verified roster per instance; per-file checks still apply on every read."""

    def __init__(self, root, split_path):
        self.root=Path(root)
        raw=Path(split_path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=SPLIT_SHA256:
            raise ValueError('Frozen D0 split changed')
        self._split=json.loads(raw)
        self._ids={role:tuple(self._split[role]) for role in ('fit','validation')}
        self._members={role:frozenset(ids) for role,ids in self._ids.items()}
        # D0 reader uses this container for its membership check as well.
        for role,members in self._members.items():
            self._split[role]=members

    def ids(self, role):
        if role not in self._ids:
            raise ValueError('Only Fit and Validation are allowed')
        return self._ids[role]

    def _require(self, image_id, role):
        if (role not in self._members or not re.fullmatch(r'[0-9]{6}',image_id)
                or image_id not in self._members[role]):
            raise ValueError('ID is not admitted for this development role')

    def read_raw(self, image_id, role):
        self._require(image_id,role)
        return read_development_pair(self.root,image_id,role,self._split)

    def hashes(self, image_id, role):
        self._require(image_id,role)
        return dict(self._split['development_sha256'][image_id])

    def read_boxes(self, image_id, role):
        """Audit path: verify the XML without reading any JPEG."""
        rel=f'Annotations/{image_id}.xml'
        expected=self.hashes(image_id,role)[rel]
        raw=(self.root/rel).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected:
            raise ValueError(f'Frozen file hash differs: {rel}')
        boxes,_,invalid=parse_voc(raw,image_id)
        if invalid:
            raise ValueError(f'Invalid annotation: {invalid}')
        return boxes

    def read_pair(self, image_id, role, *, seed=3407, epoch=0):
        rgb,t,boxes,_=self.read_raw(image_id,role)
        return prepare_pair(rgb,t,boxes,flip=flipped(image_id,role,seed,epoch))


def flipped(image_id, role, seed=3407, epoch=0):
    return role=='fit' and hashlib.sha256(f'{seed}/{epoch}/{image_id}/flip'.encode()).digest()[0]<128


def transform_boxes(boxes, flip=False):
    """Same D0 xyxy geometry without opening or allocating image pixels."""
    b=torch.as_tensor(boxes,dtype=torch.float32).clone()
    b*=.5
    b[:,1::2]+=64
    if flip:
        b[:,[0,2]]=640-b[:,[2,0]]
    return b


def prepare_pair(rgb, thermal, boxes, *, flip=False):
    rgb,t,boxes=transform(rgb,thermal,boxes,flip)
    return normalize(rgb),normalize(t),torch.from_numpy(boxes)
