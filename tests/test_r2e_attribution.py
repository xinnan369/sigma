import pytest
import torch

from experiments.r2b.state import Scan
from experiments.r2e.paths import factor, geometry, mobius


def scan(x, c):
    u = torch.tensor([[[x]]])
    coeff = torch.tensor([[[[c]]]])
    return Scan(u, u, torch.ones(1, 1), coeff, coeff, torch.tensor([10.]), torch.zeros(1), True, 1)


def test_full_factorial_separates_state_readout_and_skip():
    clean, changed = scan(2., 3.), scan(5., 7.)

    def native(u, delta, a, b, c, d, bias, softplus, nrows):
        return u * (c[:, 0] + d[None, :, None]), None

    result = factor(native, clean, changed, torch.tensor([[[26.]]]), torch.tensor([[[85.]]]))
    points = [result.output(i).item() for i in range(8)]
    assert points == [26, 35, 34, 55, 56, 65, 64, 85]
    assert [mobius(points, i) for i in range(1, 8)] == [9, 8, 12, 30, 0, 0, 0]


def test_scan_geometry_fails_for_wrong_layout():
    layout = geometry('cross.0', 1, (1, 1), scan(2., 3.))
    assert layout['directions'] == [1] and layout['endpoints'][0][0] == ('thermal', 0, 0)
    with pytest.raises(ValueError, match='layout'):
        geometry('encoder', 0, (1, 1), scan(2., 3.))
