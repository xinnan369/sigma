"""Review regressions: physical Thermal inputs and isolated module interventions."""
import numpy as np
import pytest
import torch

from experiments.exp0.perturbations import corrupt_thermal
from experiments.exp0.metrics import paired_error_stats
from experiments.exp0.hooks import CaptureOutput, ReplaceOutput


def test_gaussian_preserves_replicated_thermal_channels_and_rng():
    x = np.full((5, 6, 3), 100, dtype=np.float32)
    np.random.seed(3407)
    expected = np.random.random(3)
    np.random.seed(3407)
    y, mask = corrupt_thermal(x, 'gaussian', (1, 1, 4, 5), seed=3407)
    assert np.array_equal(y[..., 0], y[..., 1])
    assert np.array_equal(y[..., 1], y[..., 2])
    assert np.array_equal(np.random.random(3), expected)
    assert np.array_equal(y[~mask], x[~mask])
    assert np.all(x == 100)


@pytest.mark.parametrize('kwargs', [{'sigma': float('nan')}, {'sigma': float('inf')}])
def test_nonfinite_noise_rejected(kwargs):
    with pytest.raises(ValueError):
        corrupt_thermal(np.zeros((3, 3)), 'gaussian', (0, 0, 2, 2), **kwargs)


@pytest.mark.parametrize('kind,kwargs', [
    ('occlusion', {'fill': float('nan')}),
    ('occlusion', {'fill': -1}),
    ('shift_local', {'dx': 1.5}),
])
def test_invalid_fill_and_shift_rejected(kind, kwargs):
    with pytest.raises(ValueError):
        corrupt_thermal(np.zeros((3, 3)), kind, (0, 0, 2, 2), **kwargs)


def test_negative_shift_and_oversized_shift_support():
    x = np.arange(25, dtype=np.float32).reshape(5, 5)
    y, mask = corrupt_thermal(x, 'shift_local', (1, 1, 4, 4), dx=-1, dy=-1)
    assert y[1, 1] == x[2, 2] and y[3, 3] == 0
    z, large_mask = corrupt_thermal(x, 'shift_local', (1, 1, 4, 4), dx=20)
    assert np.array_equal(mask, large_mask) and np.all(z[mask] == 0)
    assert np.array_equal(z[~mask], x[~mask])


def test_different_noise_seed_is_negative_control():
    x = np.full((8, 8), 100, dtype=np.float32)
    a, _ = corrupt_thermal(x, 'gaussian', (1, 1, 7, 7), seed=3407)
    b, _ = corrupt_thermal(x, 'gaussian', (1, 1, 7, 7), seed=3408)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize('value', [float('nan'), -1, 256])
def test_invalid_physical_pixel_input_rejected(value):
    x = np.zeros((3, 3)); x[0, 0] = value
    with pytest.raises(ValueError):
        corrupt_thermal(x, 'identity', (0, 0, 2, 2))


def test_invalid_metric_types_rejected():
    labels = np.zeros((2, 2), dtype=int)
    with pytest.raises(TypeError):
        paired_error_stats(labels, labels, labels, np.full((2, 2), np.nan))
    with pytest.raises(TypeError):
        paired_error_stats(labels.astype(float), labels, labels, labels.astype(bool))


def test_wrong_to_wrong_disagreement_and_all_masked_control():
    gt = np.array([[1, 255, 1, 1]])
    a = np.array([[2, 0, 1, 2]])
    b = np.array([[3, 1, 2, 1]])
    s = paired_error_stats(a, b, gt, np.zeros_like(gt, dtype=bool))
    assert s['n_valid_outside'] == 3
    assert s['new_error_rate_outside'] == pytest.approx(1 / 3)
    assert s['corrected_error_rate_outside'] == pytest.approx(1 / 3)
    assert s['net_error_delta_outside'] == 0
    assert s['disagreement_rate_outside'] == 1
    empty = paired_error_stats(a, b, gt, np.ones_like(gt, dtype=bool))
    assert empty['n_valid_outside'] == 0
    assert np.isnan(empty['net_error_delta_outside'])


def test_replacement_snapshot_is_safe_from_inplace_consumers():
    module = torch.nn.Identity()
    clean = torch.ones(1, 2)
    replacement = ReplaceOutput(module, clean)
    clean.add_(8)
    first = module(torch.zeros(1, 2))
    first.add_(20)
    second = module(torch.zeros(1, 2))
    replacement.close()
    assert torch.equal(second, torch.ones(1, 2))


def test_context_cleanup_on_failure_and_nested_capture():
    class Two(torch.nn.Module):
        def forward(self, x):
            return (x, [2 * x])
    module = Two()
    with CaptureOutput(module) as capture:
        source = torch.ones(1, 2, requires_grad=True)
        module(source)
    assert not capture.outputs[0][0].requires_grad
    assert not module._forward_hooks
    with pytest.raises(ValueError):
        with ReplaceOutput(module, (torch.zeros(1, 3), [torch.zeros(1, 2)])):
            module(source)
    assert not module._forward_hooks
