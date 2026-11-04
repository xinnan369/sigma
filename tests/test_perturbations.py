import numpy as np
import pytest
from experiments.exp0.perturbations import corrupt_thermal


def test_identity_is_exact_copy_and_zero_mask():
    x = np.arange(36, dtype=np.float32).reshape(6, 6)
    y, m = corrupt_thermal(x, 'identity', (1, 1, 4, 4))
    assert np.array_equal(y, x) and not m.any() and not np.shares_memory(y, x)


def test_occlusion_changes_only_support_and_preserves_rgb():
    rgb = np.full((8, 8, 3), 220, dtype=np.float32)
    rgb_orig = rgb.copy(); t = np.full((8, 8, 3), 100, dtype=np.float32)
    y, m = corrupt_thermal(t, 'occlusion', (2, 2, 6, 5))
    assert m.sum()==12 and (y[m] == 0).all()
    assert np.array_equal(y[~m], t[~m])
    assert np.array_equal(rgb, rgb_orig)


def test_gaussian_is_local_reproducible_and_clipped():
    t = np.full((10, 10), 100, dtype=np.uint8)
    a, ma = corrupt_thermal(t, 'gaussian', (2, 3, 8, 9), seed=3407, sigma=15)
    b, mb = corrupt_thermal(t, 'gaussian', (2, 3, 8, 9), seed=3407, sigma=15)
    assert np.array_equal(a,b) and np.array_equal(ma,mb)
    assert np.array_equal(a[~ma],t[~ma])
    assert a.min() >= 0 and a.max() <= 255 and np.any(a[ma]!=t[ma])


def test_shift_local_does_not_wrap_or_modify_exterior():
    t = np.arange(81,dtype=np.float32).reshape(9,9)
    y,m=corrupt_thermal(t,'shift_local',(2,2,7,7),dx=2,dy=1,fill=7)
    assert np.array_equal(y[~m],t[~m])
    assert y[2,2] == 7
    assert y[4,5] == t[3,3]


def test_no_op_controls_and_input_not_mutated():
    t=np.arange(25,dtype=np.float32).reshape(5,5); keep=t.copy()
    for kind,kw in [('gaussian',{'sigma':0}),('shift_local',{'dx':0,'dy':0})]:
        y,m=corrupt_thermal(t,kind,(1,1,4,4),**kw)
        assert np.array_equal(y,t) and not m.any()
    assert np.array_equal(t,keep)


def test_invalid_roi_or_kind_fails():
    t=np.zeros((4,4))
    with pytest.raises(ValueError): corrupt_thermal(t,'occlusion',(0,0,5,1))
    with pytest.raises(ValueError): corrupt_thermal(t,'invalid',(0,0,1,1))
