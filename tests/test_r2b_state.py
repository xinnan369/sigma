import math

import pytest
import torch

from experiments.r2b.controls import Attention, CNN
from experiments.r2b.state import Scan, coordinates, literal, outputs


def test_recurrence_reset_is_post_update_and_changes_only_suffix():
    u = torch.ones(1, 1, 4)
    b = torch.ones(1, 1, 4)
    scan = Scan(u, u, torch.tensor([[math.log(.5)]]), b, b * 3, torch.tensor([2.]), None, False, 1)
    base = literal(scan)
    assert base.flatten().tolist() == [1., 1.5, 1.75, 1.875]
    changed = literal(scan, reset=(1, torch.full((1, 1, 1), 7.)))
    assert changed.flatten().tolist() == [1., 7., 4.5, 3.25]
    assert outputs(scan, changed).flatten().tolist() == [5., 23., 15.5, 11.75]
    assert torch.equal(base, literal(scan, reset=(1, base[..., 1])))


def test_mapping_distinguishes_row_column_reverse_and_modality_boundary():
    assert [coordinates('four', k, 1, 2, 3) for k in range(4)] == [
        ('feature', 0, 1), ('feature', 1, 0), ('feature', 1, 1), ('feature', 0, 2)]
    assert coordinates('concat', 1, 1199, 30, 40) == ('thermal', 0, 0)
    assert coordinates('concat', 1, 1200, 30, 40) == ('rgb', 29, 39)
    assert coordinates('concat', 0, 1200, 30, 40) == ('thermal', 0, 0)
    with pytest.raises(ValueError):
        coordinates('cross', 0, 6, 2, 3)


@pytest.mark.parametrize('block', [CNN, Attention])
@pytest.mark.parametrize('cross', [False, True])
def test_control_interfaces_and_gradients(block, cross):
    torch.manual_seed(3407)
    net = block(16, cross=cross)
    rgb, thermal = (torch.randn(2, 6, 7, 16, requires_grad=True) for _ in range(2))
    y = net(rgb, thermal)
    results = y if isinstance(y, tuple) else (y,)
    assert len(results) == (2 if cross else 1)
    assert all(x.shape == rgb.shape and torch.isfinite(x).all() for x in results)
    sum(x.square().mean() for x in results).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in (rgb, thermal, *net.parameters()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA and Sigma selective scan')
def test_gpu_state_intervention_against_literal_and_cuda():
    extension = pytest.importorskip('selective_scan_cuda_core')
    pytest.importorskip('triton')
    from experiments.r2b.audit import compare, native_call
    from experiments.r2b.instrument import reset_readout
    from experiments.r2b.triton_state import trajectory

    torch.manual_seed(3407)
    u, dt = (torch.randn(2, 8, 35, device='cuda') for _ in range(2))
    b, c = (torch.randn(2, 2, 4, 35, device='cuda') for _ in range(2))
    scan = Scan(u, dt, -torch.ones(8, 4, device='cuda'), b, c, None, None, True, 1)
    native, prefix = native_call(extension, scan)
    h = trajectory(scan)
    compare(scan, h, native, prefix)
    target = h[..., 17].clone()
    target[:, 4:] += .25
    changed, stats = reset_readout(scan, h, native, 1, 17, target, verify=True)
    ref = literal(scan, reset=(17, target))
    torch.testing.assert_close(changed, outputs(scan, ref), atol=3e-5, rtol=2e-4)
    assert torch.equal(changed[..., :17], native[..., :17])
    assert torch.equal(changed[:, :4], native[:, :4])
    assert stats['changed_reset_state'] == 2 * 4 * 4 and stats['changed_future_state_steps'] == 17
