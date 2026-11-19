import torch
import pytest

from experiments.r3_d1 import coco,fcos
from experiments.r3_d1.data import normalize
from experiments.r3_d1.model import Detector
from experiments.r3_d1.budget import complete_count,match


def test_coco_cases():
    images=[dict(id=1,width=1280,height=1024)]
    gt=[dict(id=1,image_id=1,category_id=1,bbox=[0,0,40,40],area=1600,iscrowd=0)]
    obj=dict(image_id=1,category_id=1,bbox=[0,0,40,40],score=.9)
    for pred,expected in [([obj],1),([dict(obj,category_id=2)],0),([],0),
                          ([dict(obj,bbox=[100,100,40,40])],0),
                          ([obj,dict(obj,score=.8)],1)]:
        r=coco.evaluate(images,gt,pred)
        assert r==pytest.approx(dict(ap50=expected,ap50_95=expected))
    images.append(dict(id=2,width=1280,height=1024))
    gt.append(dict(gt[0],id=2,image_id=2))
    assert coco.evaluate(images,gt,[obj])['ap50']==pytest.approx(51/101)


def test_targets_and_losses():
    points=torch.tensor([[20.,84.],[8.,68.],[32.,84.],[20.,60.]])
    boxes=torch.tensor([[0.,64.,40.,104.],[4.,68.,36.,100.]])
    mask,dist=fcos.targets(points,boxes,0)
    assert mask.tolist()==[True,False,False,False]
    assert dist[0].tolist()==[16,16,16,16]
    assert not fcos.targets(points,boxes,1)[0].any()
    assert fcos.targets(torch.tensor([[40.,100.]]),torch.tensor([[8.,68.,72.,132.]]),1)[0].item()
    p=[torch.zeros(1,6,n,n,requires_grad=True) for n in (80,40,20)]
    parts=fcos.losses(p,[boxes])
    assert all(torch.isfinite(a) and a>0 for a in parts.values())
    sum(parts.values()).backward()
    assert all(torch.isfinite(a.grad).all() for a in p)
    zero=fcos.losses(p,[torch.empty(0,4)])
    assert zero['box']==zero['center']==0


def test_decode_inverse_clip_nms_and_padding():
    a=[torch.full((1,6,n,n),-100.) for n in (80,40,20)]
    a[0][0,:,8,0]=torch.tensor([20.,2.,2.,2.,2.,20.])
    a[0][0,:,0,0]=30 # padding center; must never win
    d=fcos.decode(a)[0]
    w=torch.nn.functional.softplus(torch.tensor(2.)).item()*8
    assert d['boxes'][0].tolist()==pytest.approx([0,0,2*(4+w),2*(4+w)])
    assert len(d['scores'])<=100
    idx=fcos.nms(torch.tensor([[0,0,5,5],[0,0,5,5],[10,10,15,15.]]),torch.tensor([.7,.7,.3]))
    assert idx.tolist()==[0,2]


def test_cpu_models_shapes_common_initialization_and_modality_isolation():
    import numpy as np
    x=normalize(np.full((640,640),114,np.uint8)).unsqueeze(0)
    assert x[0,:,0,0].tolist()==pytest.approx([(114/255-m)/s for m,s in
                                             zip((.485,.456,.406),(.229,.224,.225))])
    a,b,c=Detector('A').eval(),Detector('B').eval(),Detector('C',widths=[32]*3).eval()
    assert all(torch.equal(v,c.state_dict()[k]) for m in (a,b) for k,v in m.state_dict().items())
    for m,inputs in [(a,dict(rgb=x)),(b,dict(thermal=x)),(c,dict(rgb=x,thermal=x))]:
        assert complete_count(m,inputs)['parameters']>0
    with pytest.raises(ValueError):
        a(x,x)
    with pytest.raises(ValueError):
        Detector('D')


def test_width_formula_against_instantiated_cnn():
    x=torch.zeros(1,3,640,640)
    t=complete_count(Detector('C',widths=[16]*3).eval(),dict(rgb=x,thermal=x))
    cp=t['parameters']-3*(470*16+192)
    cm=t['mac']-466*16*8400
    result=match(t,cp,cm,range(16,20))
    assert result==dict(widths=[16]*3,parameters=t['parameters'],mac=t['mac'])


def test_frozen_sampler_optimizer_and_checkpoint_ties():
    import numpy as np
    from experiments.r3_d1.plan import WIDTHS,checkpoint_key,epoch_indices,optimizer,training_forward

    a=epoch_indices(3407,0)
    assert len(a)==2482 and len(a[-1])==2
    assert sorted(np.concatenate(a).tolist())==list(range(9926))
    assert np.array_equal(a[0],epoch_indices(3407,0)[0])
    assert not np.array_equal(a[0],epoch_indices(3411,0)[0])
    assert not np.array_equal(a[0],epoch_indices(3407,1)[0])
    m=Detector('C',widths=WIDTHS)
    groups=optimizer(m).param_groups
    assert {id(p) for g in groups for p in g['params']}=={id(p) for p in m.parameters()}
    assert sum(len(g['params']) for g in groups)==len(list(m.parameters()))
    x=torch.zeros(1,3,640,640)
    parts=training_forward(m,x,x,[torch.tensor([[72.,84.,112.,140.]])])
    sum(parts.values()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert checkpoint_key(dict(ap50_95=.4,ap50=.6),0)>checkpoint_key(dict(ap50_95=.4,ap50=.6),1)


def test_development_data_verifies_and_parses_once_and_refuses_test_before_open(tmp_path,monkeypatch):
    import hashlib
    import json
    from pathlib import Path
    import cv2
    import numpy as np
    from experiments.r3_d1 import data

    ids=['010001','010002']
    roles=dict(fit=ids,validation=[])
    files={}
    ok,jpeg=cv2.imencode('.jpg',np.zeros((1024,1280,3),np.uint8))
    assert ok
    for i in ids:
        files[i]={f'{m}/train/{i}.jpg':jpeg.tobytes() for m in ('visible','infrared')}
        files[i][f'Annotations/{i}.xml']=(f'<annotation><filename>{i}.jpg</filename><size>'
                                         '<width>1280</width><height>1024</height></size></annotation>').encode()
    for rr in files.values():
        for rel,b in rr.items():
            p=tmp_path/rel
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(b)
    src=tmp_path/'split.json'
    raw=json.dumps(dict(**roles,development_sha256={
        i:{p:hashlib.sha256(b).hexdigest() for p,b in rr.items()} for i,rr in files.items()})).encode()
    src.write_bytes(raw)
    monkeypatch.setattr(data,'SPLIT_SHA256',hashlib.sha256(raw).hexdigest())
    open_orig=Path.read_bytes
    loads_orig=json.loads
    calls=[]
    parses=[]
    def read(p):
        calls.append(p)
        return open_orig(p)
    def parse(b,*a,**k):
        parses.append(b)
        return loads_orig(b,*a,**k)
    monkeypatch.setattr(Path,'read_bytes',read)
    monkeypatch.setattr(data.json,'loads',parse)
    ds=data.DevelopmentData(tmp_path,src)
    for i in ids:
        rgb,t,b=ds.read_pair(i,'fit')
        assert rgb.shape==t.shape==(3,640,640) and b.shape==(0,4)
    assert calls.count(src)==1 and parses==[raw] and len(calls)==7
    before=list(calls)
    for reader in (ds.read_pair,ds.read_boxes):
        with pytest.raises(ValueError,match='not admitted'):
            reader('240280','test')
    assert calls==before
    src.write_bytes(raw+b' ')
    with pytest.raises(ValueError,match='Frozen D0 split changed'):
        data.DevelopmentData(tmp_path,src)
    assert parses==[raw]
    # A cached valid manifest still verifies all newly opened file bytes.
    (tmp_path/'Annotations/010002.xml').write_bytes(b'modified')
    with pytest.raises(ValueError,match='Frozen file hash differs'):
        ds.read_pair('010002','fit')
    with pytest.raises(ValueError,match='Frozen file hash differs'):
        ds.read_boxes('010002','fit')


def test_coverage_distinguishes_overlap_losers_from_no_eligible_points():
    from experiments.r3_d1.coverage import coverage, points_by_level
    p=points_by_level()
    b=torch.tensor([[64.,80.,104.,120.],[64.,80.,104.,120.],[0.,65.,1.,66.]])
    wins,eligible,levels=coverage(b,p)
    assert wins.tolist()==[9,0,0] and eligible.tolist()==[9,9,0]
    assert levels==[9,0,0]
    assert fcos.targets(p[0],b,0)[0].sum().item()==levels[0]
    assert coverage(b[:0],p)[2]==[0,0,0]
