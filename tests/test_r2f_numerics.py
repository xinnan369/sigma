import pytest
import torch

from experiments.r2b.state import Scan
from experiments.r2f.numerics import readout, sequential, summarize


def test_independent_grouped_recurrence_and_readout():
    u = torch.tensor([[[1., 2.], [3., 4.]]])
    b = torch.tensor([[[[1., 1.]], [[2., 3.]]]])
    scan = Scan(u, torch.ones_like(u), torch.zeros(2, 1), b, b, torch.ones(2), None, False, 1)
    h = sequential(scan)
    assert h.squeeze(2).tolist() == [[[1., 3.], [6., 18.]]]
    assert torch.equal(sequential(scan, exp2=True, fused=True), h)
    assert readout(scan, h).tolist() == [[[2., 5.], [15., 58.]]]


def test_mixed_tolerance_reports_worst_coordinate_and_exceedances():
    reference = torch.tensor([[[0., 2., 1.]]])
    changed = torch.tensor([[[3e-5, 2.0001, 1.]]])
    stats = summarize(changed, reference)
    assert stats['exceedances'] == 1
    assert stats['worst_mixed']['index'] == (0, 0, 0)
    assert stats['worst_absolute']['index'] == (0, 0, 1)
    assert stats['worst_mixed']['mixed_ratio'] == pytest.approx(1.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA basis probes require GPU')
def test_native_basis_and_hybrid_reconstruction():
    extension = pytest.importorskip('selective_scan_cuda_core')
    pytest.importorskip('triton')
    from experiments.r2f.pilot import synthetic

    with torch.inference_mode():
        cases = synthetic(extension.fwd)
    assert len(cases) == 4 and all(len(x['hybrid']) == 8 for x in cases)
