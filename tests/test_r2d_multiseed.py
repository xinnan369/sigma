import numpy as np
import pytest
import torch

from experiments.r2b.controls import WIDTHS
from experiments.r2d.analysis import bootstrap_weights, seed_mean, summarize
from experiments.r2d.models import WideAttention, WideCNN, fusion_parameters
from experiments.r2d.profile import design


@pytest.mark.parametrize('arm,cls', [('cnn', WideCNN), ('attention', WideAttention)])
def test_frozen_search_matches_instantiated_parameter_count(arm, cls):
    config, choices = design()
    widths = choices[arm]['widths']
    count = sum(p.numel() for c, m in zip(WIDTHS, widths, strict=True)
                for cross in (True, False) for p in cls(c, m, cross=cross).parameters())
    assert count == fusion_parameters(arm, widths) == choices[arm]['fusion_parameters']
    assert abs(count - 14_371_200) / config['reference_parameters'] < .01


@pytest.mark.parametrize('cls', [WideCNN, WideAttention])
@pytest.mark.parametrize('cross', [True, False])
def test_wider_block_preserves_interface_and_uses_parameters(cls, cross):
    model = cls(8, 16, cross=cross)
    values = [torch.randn((1, 5, 7, 8)) for _ in range(2)]
    result = model(*values)
    outputs = result if cross else (result,)
    assert len(outputs) == (2 if cross else 1)
    assert all(x.shape == values[0].shape for x in outputs)
    sum(x.square().mean() for x in outputs).backward()
    assert all(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in model.parameters())


def test_crossed_weights_are_shared_and_seed_mean_masks_missing_iou():
    seeds, groups = bootstrap_weights(3, 54, 20, 3407)
    assert np.all(seeds.sum(1) == 3) and np.all(groups.sum(1) == 54)
    repeated_seeds, repeated_groups = bootstrap_weights(3, 54, 20, 3407)
    assert np.array_equal(seeds, repeated_seeds) and np.array_equal(groups, repeated_groups)
    weights = np.array([[2, 0], [0, 2], [1, 1]])
    a = np.array([[1., 2., 3.], [4., np.nan, 7.]])
    b = np.array([[1., 2., 1.], [4., np.nan, 5.]])
    assert seed_mean(a - b, weights) == pytest.approx([0., np.nan, 2.], nan_ok=True)
    result = summarize(np.array([0., 2.]), a - b, weights)
    assert result['point'] == 1 and result['seed_sd'] == pytest.approx(np.sqrt(2))
    assert result['valid_draws'] == 2
