import numpy as np
import pytest

from experiments.r2k_q1.model import operating_threshold
from experiments.r2k_q1.metrics import distribution,metrics
from experiments.r2k_q1.protocol import CONFIG,build,read


def test_frozen_group_partition():
    c=read(CONFIG)
    assert build()==c
    groups=[set(x['groups']) for x in c['partitions'].values()]
    assert sum(map(len,groups))==len(set.union(*groups))
    ids=[i for x in c['partitions'].values() for i in x['images']]
    assert len(ids)==len(set(ids))==627
    assert 'blur' not in c['fit_calibration_families']


def test_perfect_and_tied_threshold_metrics():
    y=np.array([0,0,1,1])
    assert metrics(distribution(y,np.array([0,0,255,255])))['ap']==1
    tied=metrics(distribution(y,np.array([0,0,0,0])))
    assert tied['ap']==tied['auc']==tied['brier']==.5
    assert tied['balanced_brier']==.5
    assert operating_threshold(np.array([0]*19+[7],dtype=np.uint8),.05)==1
    assert operating_threshold(np.array([255],dtype=np.uint8),.05)==256


def test_spatial_inference_api_and_degradation_locality():
    pytest.importorskip('cv2')
    pytest.importorskip('scipy')
    from experiments.r2k_q1.model import degrade,predict,grid
    cfg=read(CONFIG)
    x=np.arange(480*640,dtype=np.uint8).reshape(480,640)
    y,roi,interior,q=degrade(x,'blur',cfg,'00000N',0)
    assert roi.shape==(60,80) and roi.sum()==144 and (~interior | roi).all()
    assert 0<=q<=3 and np.array_equal(x[grid(x)][~roi],y[grid(y)][~roi])
    head={'mean':[0]*6,'scale':[1]*6,'weights':[0]*7}
    assert (predict(y,head,[1,0])==128).all()


def test_independent_ap_and_brier_match_calculation():
    pytest.importorskip('resource')
    from experiments.r2k_q1.audit import ap_and_brier
    h=distribution(np.array([0,1,1,0]),np.array([1,128,128,240]))
    a,b=ap_and_brier(h[None,...])
    assert a[0]==pytest.approx(2/3)
    assert a[0]==pytest.approx(metrics(h)['ap'])
    assert b[0]==pytest.approx(metrics(h)['brier'])
