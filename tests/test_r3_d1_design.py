import numpy as np
import pytest

from experiments.r3_d0.d1_design import budget, mix


def test_fusion_budget_frozen_at_three_feature_levels():
    assert budget()=={'parameters':25152,'mac_equivalents':70425600}


def test_identity_and_no_future_or_cross_image_leakage():
    x=np.array([[1,2],[3,5],[8,13],[21,34]],np.float32)
    identity=[[1,1],[0,0],[0,0]]
    for arm in ('C','D'):
        assert np.array_equal(mix(x,identity,arm),x)
        p=[[0,0],[.5,.5],[1,1]]
        y=mix(x,p,arm)
        assert np.array_equal(mix(x[:2],p,arm),y[:2])
        assert not mix(np.zeros_like(x),p,arm).any()
    with pytest.raises(ValueError):
        mix(x,identity,'A')


def test_impulse_has_local_support_vs_recurrent_tail():
    x=[[1],[0],[0],[0]]
    p=[[0],[.5],[1]]
    assert mix(x,p,'C').ravel().tolist()==[0,.5,1,0]
    a=np.tanh(np.float32(.5))
    np.testing.assert_allclose(mix(x,p,'D').ravel(),[1,a,a*a,a*a*a],rtol=1e-6)
