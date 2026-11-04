import numpy as np
import pytest
from experiments.exp0.metrics import paired_error_stats


def test_new_and_corrected_errors_are_distinct():
    gt=np.array([[1,1,2],[2,1,255]])
    a=np.array([[1,2,2],[2,1,0]])
    b=np.array([[2,1,2],[1,1,1]])
    mask=np.array([[False,False,True],[False,True,False]])
    s=paired_error_stats(a,b,gt,mask)
    assert s['n_valid_outside']==3
    assert s['new_errors_outside']==2
    assert s['corrected_errors_outside']==1
    assert s['new_error_rate_outside']==pytest.approx(2/3)
    assert s['net_error_delta_outside']==pytest.approx(1/3)


def test_ignore_pixels_and_empty_region_are_safe():
    x=np.zeros((2,2),dtype=int)
    s=paired_error_stats(x,x,np.full((2,2),255),np.zeros((2,2),dtype=bool))
    assert s['n_valid_outside']==0 and np.isnan(s['new_error_rate_outside'])


def test_identity_predictions_have_zero_delta():
    gt=np.array([[1,2],[2,1]])
    a=np.array([[1,1],[2,2]])
    m=np.zeros_like(gt,dtype=bool)
    s=paired_error_stats(a,a,gt,m)
    assert s['new_errors_outside']==0 and s['corrected_errors_outside']==0
    assert s['net_error_delta_outside']==0.0


def test_metric_shape_validation():
    with pytest.raises(ValueError):
        paired_error_stats(np.zeros((2,2)),np.zeros((2,2)),np.zeros((2,3)),np.zeros((2,2)))
