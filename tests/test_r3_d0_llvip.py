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

def test_frozen_release_partitions_and_hashes():
    import hashlib
    import json
    from pathlib import Path

    evidence=Path(__file__).resolve().parents[1]/'reports/evidence'
    folder=evidence/'r3-d0'
    raw=(folder/'split.json').read_bytes()
    split=json.loads(raw)
    audit=json.loads((folder/'audit.json').read_bytes())
    assert hashlib.sha256(raw).hexdigest()==audit['split_sha256']==(
        'bd1090cb2ad4696db3ba2f8e984788204c433a7e0cab656073c5e2b67e53e3d4')
    archive=(evidence/'llvip-precheck/file_manifest.json').read_bytes()
    assert hashlib.sha256(archive).hexdigest()==audit['manifest_sha256']
    rows={r['path']:r for r in json.loads(archive)['files']}
    official={s:{Path(p).stem for p in rows if p.startswith(f'visible/{s}/')}
              for s in ('train','test')}
    fit,val,q=(set(split[k]) for k in ('fit','validation','quarantine'))
    assert (len(fit),len(val),len(q))==(9926,2087,12)
    assert not (fit&val or fit&q or val&q)
    assert fit|val|q==official['train'] and not (fit|val)&official['test']
    seen=set()
    for group, ids in split['proxy_components'].items():
        members=set(ids)
        assert len(members)==len(ids) and not members&seen
        assert members <= (val if group in split['validation_components'] else fit)
        seen.update(members)
    assert seen==fit|val
    suspicious={r['official_train'] for r in audit['cross_split_candidates']
                if r['verdict']=='UNKNOWN'}
    invalid={r['id'] for r in audit['invalid_annotations']}
    assert len(suspicious)==7 and len(invalid)==5 and suspicious|invalid==q
    assert set(split['development_sha256'])==fit|val
    for i in fit|val:
        expected={p:rows[p]['sha256'] for p in
                  (f'visible/train/{i}.jpg', f'infrared/train/{i}.jpg', f'Annotations/{i}.xml')}
        assert split['development_sha256'][i]==expected
    assert audit['official_test_exposure']==dict(xml_read=3463,person_annotations_counted=8302,
        images_pixel_decoded=0,detection_inferences=0,detection_metric_computations=0,
        model_selections=0,historical_thermal_pilot_pairs=64)
    assert len(set(audit['historic_pilot_test_ids']))==64
    assert set(audit['historic_pilot_test_ids'])<=official['test']
