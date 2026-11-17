import numpy as np
import pytest

pytest.importorskip('resource')
from experiments.r2k_q1d.diagnose import pixel_change, rank_metrics


def test_tied_rank_metrics_and_monotone_mapping():
    target=np.array([1,0,1,0],bool)
    scores=np.array([.9,.6,.6,.1])
    a,b=rank_metrics(target,scores),rank_metrics(target,scores**3)
    assert a['auc']==pytest.approx(.875)
    assert a['ap']==pytest.approx(5/6)
    assert a['auc']==b['auc'] and a['ap']==b['ap']
    assert rank_metrics(target,np.ones(4))['auc']==.5


def test_signed_conversion_prevents_uint_wrap_and_respects_roi():
    raw=np.array([[254,2],[10,20]],np.uint8)
    changed=np.array([[1,3],[9,20]],np.uint8)
    assert pixel_change(raw,changed,np.s_[:,1]).tolist()==[2,1,2,2,2,1]
    assert pixel_change(raw,changed,np.s_[:,:]).tolist()==[4,1,3,3,3,255]
