import pytest
import torch

from experiments.r1c.state import accumulate_loss, normalize_gradients
from experiments.r1d.state import EpochSampler
from experiments.r2i.augment import train_occlusion
from experiments.r2i_s1.driver import coordinates, verify_history


def test_full_order_and_tail():
    ids = list(range(1254))
    sampler = EpochSampler(len(ids), 3407)
    rows = []
    for epoch in range(1,6):
        seen = []
        for _ in range(157):
            names = sampler.next_window(8)
            seen.extend(names)
            rows.append(dict(epoch=epoch, update=len(rows)+1, cursor=sampler.cursor,
                             sample_ids=names, microbatches=len(names),
                             valid_pixel_counts=[1]*len(names)))
        assert sorted(seen)==ids and len(names)==6
        if epoch<5:
            sampler.advance_epoch()
    assert coordinates(156)==(1,1248,6)
    assert coordinates(157)==(2,0,8)
    assert verify_history(rows,ids,complete=True).cursor==1254
    rows[156]['sample_ids'] = [0]*6
    with pytest.raises(ValueError, match='ordering'):
        verify_history(rows,ids,complete=True)


def test_tail_is_pixel_weighted():
    m = torch.nn.Linear(1,1,bias=False)
    for rate,pixels in ((1,2),(3,6),(7,1),(1,1),(1,1),(1,1)):
        accumulate_loss(m.weight.sum()*rate,pixels)
    normalize_gradients(m,12)
    assert m.weight.grad.item()==pytest.approx(30/12)


def test_epoch_aware_augmentation():
    t = torch.ones((1,3,100,100))
    args = t,(.5,)*3,(.5,)*3,3407,'00057D',1.,64
    a, box1 = train_occlusion(*args,epoch=1)
    b, box2 = train_occlusion(*args,epoch=2)
    again, box3 = train_occlusion(*args,epoch=2)
    assert box1!=box2 and not torch.equal(a,b)
    assert box2==box3 and torch.equal(b,again)
