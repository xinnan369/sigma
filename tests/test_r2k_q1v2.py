import numpy as np
import pytest

from experiments.r2k_q1.protocol import ROOT, read, sha
from experiments.r2k_q1v2.features import describe_multiscale


def config():
    return read(ROOT/'configs/R2k_Q1v2_protocol.json')


def test_protocol_has_no_evaluation_exposure():
    c=config()
    assert sha(ROOT/c['parent'])==c['parent_sha256']
    assert c['roles']==['fit','calibration']
    assert c['blur_entirely_held_out'] and 'blur' not in c['families']
    assert c['scales']==[3,9,21]


def test_flat_and_frequency_cases_keep_grid_alignment():
    pytest.importorskip('cv2')
    cfg=config()
    flat=np.full((96,96),100,dtype=np.uint8)
    f,coverage=describe_multiscale(flat,cfg)
    assert f.shape==(12,12,6) and np.isfinite(f).all() and not coverage.any()
    assert np.max(abs(f[1:-1,1:-1,:4]))<1e-6
    y,x=np.indices(flat.shape)
    detail=(((x+y)%2)*255).astype(np.uint8)
    f,_=describe_multiscale(detail,cfg)
    a,_=describe_multiscale(detail,cfg,ablated=True)
    assert float(np.median(f[...,1]))>2
    assert np.array_equal(f[...,2:],a[...,2:])
    scene=np.zeros_like(flat)
    scene[32:64,32:64]=255
    f,_=describe_multiscale(scene,cfg)
    assert f[0,0,-1]==1 and f[5,5,-1]==0
    with pytest.raises(ValueError, match='grayscale'):
        describe_multiscale(scene[...,None],cfg)


def test_inference_keeps_unidentifiable_points():
    pytest.importorskip('resource')
    pytest.importorskip('cv2')
    pytest.importorskip('scipy')
    from experiments.r2k_q1v2.run import predict
    head={'mean':[0]*6,'scale':[1]*6,'weights':[0]*7}
    p,ok=predict(np.zeros((480,640),np.uint8),'thermal',config(),head,[1,0])
    assert p.shape==ok.shape==(60,80)
    assert np.all(p==128) and not ok.any()
