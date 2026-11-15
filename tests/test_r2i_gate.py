import torch

from experiments.r2i.augment import train_occlusion
from experiments.r2i.gate import Gate


def test_identity_gradients_and_matched_counts():
    torch.manual_seed(3407)
    x, y = (torch.randn(1, 3, 4, 6, requires_grad=True) for _ in range(2))
    for reliable in (True, False):
        gate = Gate(reliable)
        assert sum(p.numel() for p in gate.parameters()) == 4
        a, b = gate(x, y)
        assert torch.equal(a, x) and torch.equal(b, y)
        (a.square().sum() + b.square().sum()).backward()
        assert all(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0
                   for p in gate.parameters())


def test_control_has_no_thermal_scoring_input():
    x, y = torch.rand(2, 4, 4, 8), torch.rand(2, 4, 4, 8)
    for reliability in (True, False):
        gate = Gate(reliability)
        with torch.no_grad():
            gate.score.weight.fill_(.5)
        a, _ = gate(x, y)
        b, _ = gate(x, 2*y)
        assert torch.equal(a, b) is (not reliability)


def test_training_augmentation_is_seeded_and_local():
    x = torch.ones(1, 3, 12, 10)
    args = x, (.5, .25, .125), (.25, .25, .25), 3407, '00057D', 1., 4
    a, box = train_occlusion(*args)
    b, other = train_occlusion(*args)
    assert box == other and torch.equal(a, b) and x.eq(1).all()
    y, z, n = box
    assert a[0, :, y, z].tolist() == [-2., -1., -.5]
    assert a.ne(x).sum().item() == 3*n*n
