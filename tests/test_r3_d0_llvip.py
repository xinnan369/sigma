import numpy as np
import pytest

from experiments.r3_d0.audit import group_split
from experiments.r3_d0.llvip import parse_voc, read_development_pair, source_xyxy, transform


def test_voc_does_not_fix_equal_or_outside_edges():
    def obj(x0, x1):
        return ('<object><name>person</name><difficult>1</difficult><bndbox>'
                f'<xmin>{x0}</xmin><ymin>4</ymin><xmax>{x1}</xmax><ymax>10</ymax>'
                '</bndbox></object>')
    data=('<annotation><filename>000001.jpg</filename><size><width>1280</width>'
          '<height>1024</height></size>'+obj(5,9)+obj(5,5)+obj(1279,1281)+'</annotation>')
    boxes, difficult, invalid=parse_voc(data,'000001')
    assert boxes.tolist()==[[5,4,9,10]] and difficult==[1]
    assert [r['object_index_zero_based'] for r in invalid]==[1,2]
    assert invalid[0]['xyxy']==[5,4,5,10]
    with pytest.raises(ValueError):
        parse_voc(data,'000002')


def test_same_geometry_on_modalities_and_boxes():
    t=np.zeros((1024,1280),np.uint8)
    t[20:40,100:300]=255
    r=np.repeat(t[:,:,None],3,axis=2)
    boxes=[[100,20,300,40],[0,0,1280,1024]]
    a,b,xy=transform(r,t,boxes)
    assert np.array_equal(a[:,:,0],b)
    assert (b[:64]==114).all() and (b[576:]==114).all()
    assert (b[74:84,50:150]==255).all()
    assert xy.tolist()==[[50,74,150,84],[0,64,640,576]]
    assert source_xyxy(xy).tolist()==boxes
    ar,br,fl=transform(r,t,boxes,flip=True)
    assert np.array_equal(ar,a[:,::-1]) and np.array_equal(br,b[:,::-1])
    assert fl.tolist()==[[490,74,590,84],[0,64,640,576]]
    with pytest.raises(ValueError):
        transform(r,t,[[0,0,0,50]])
    with pytest.raises(ValueError):
        transform(r,t[:-1],[])


def test_split_determinism_and_non_development_guard(tmp_path):
    groups={'a':['01'], 'b':['02','03'], 'c':['04','05','06','07','08','09','10']}
    assert group_split(groups)==group_split(dict(reversed(list(groups.items()))))==['b']
    with pytest.raises(ValueError):
        read_development_pair(tmp_path,'000001','test',{})
    with pytest.raises(ValueError):
        read_development_pair(tmp_path,'../bad','fit',{'fit':['../bad']})
    with pytest.raises(ValueError):
        group_split({'one':['id']})


def test_reader_hashes_release_bytes_and_refuses_changes(tmp_path):
    import hashlib
    import cv2

    ident='000001'
    rels=[f'{m}/train/{ident}.jpg' for m in ('visible','infrared')]
    ok,jpeg=cv2.imencode('.jpg',np.zeros((1024,1280,3),np.uint8))
    assert ok
    files=dict.fromkeys(rels,jpeg.tobytes())
    files[f'Annotations/{ident}.xml']=(b'<annotation><filename>000001.jpg</filename>'
        b'<size><width>1280</width><height>1024</height></size></annotation>')
    for rel,data in files.items():
        p=tmp_path/rel
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(data)
    split={'fit':[ident],'validation':[], 'development_sha256':{
        ident:{rel:hashlib.sha256(data).hexdigest() for rel,data in files.items()}}}
    r,t,b,flags=read_development_pair(tmp_path,ident,'fit',split)
    assert r.shape==(1024,1280,3) and t.shape==(1024,1280)
    assert b.shape==(0,4) and flags==[]
    (tmp_path/rels[0]).write_bytes(b'changed')
    with pytest.raises(ValueError,match='Frozen file hash differs'):
        read_development_pair(tmp_path,ident,'fit',split)
