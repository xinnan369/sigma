import numpy as np
import pytest
import torch
from torch import nn

from experiments.r2c.analysis import rate_contrast, risk_values
from experiments.r2c.audit import expected_windows, reconcile_window
from experiments.r2c.protocol import digest_common, load_protocol, optimizer_steps


def test_frozen_schedule_covers_all_records_and_keeps_each_six_record_tail():
    cfg, _, split = load_protocol()
    epochs = list(expected_windows(split['train_ids'], cfg['seed'], cfg['epochs']))
    assert len(epochs) == 5
    assert epochs[0] != epochs[1]
    for epoch in epochs:
        assert [len(window) for window in epoch] == [8] * 156 + [6]
        assert sorted(name for window in epoch for name in window) == sorted(split['train_ids'])


def test_common_state_fingerprint_excludes_only_fusion_lists():
    def make(fusion):
        m = nn.Module()
        m.backbone = nn.Module()
        m.backbone.cross_mamba = fusion
        m.backbone.channel_attn_mamba = nn.Identity()
        m.backbone.register_buffer('running', torch.zeros(2))
        m.decoder = nn.Linear(2, 2)
        return m

    left, right = make(nn.Linear(2, 5)), make(nn.Conv2d(2, 3, 1))
    right.decoder.load_state_dict(left.decoder.state_dict())
    assert digest_common(left) == digest_common(right)
    right.backbone.running[0] = 1
    assert digest_common(left) != digest_common(right)


def test_optimizer_steps_checks_every_tensor_and_the_schedule():
    m = nn.Linear(2, 2)
    cfg, _, _ = load_protocol()
    optimizer = torch.optim.AdamW(m.parameters(), lr=cfg['lr'], betas=tuple(cfg['betas']))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    m(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    optimizer_steps(optimizer, scheduler, 2, 1, cfg)
    optimizer.state[m.bias]['step'].zero_()
    with pytest.raises(ValueError, match='coverage/step'):
        optimizer_steps(optimizer, scheduler, 2, 1, cfg)


def test_independent_audit_rejects_wrong_tail_and_reconciles_pixel_weighted_ce():
    names = list('abcdef')
    event = {'sample_ids': names, 'microbatches': 6, 'valid_pixel_counts': [1, 2, 3, 4, 5, 6],
             'losses': [1., 2., 3., 4., 5., 6.], 'cursor': 1254, 'update': 157, 'epoch_update': 157,
             'epoch': 1, 'gradient_tensor_count': 556, 'gradient_l2': 1., 'lr': 6e-5,
             'pixel_loss_sum': 91., 'mean_pixel_loss': 91. / 21,
             'class_pixels': [21] + [0] * 8, 'augmentation_rng_before_sha256': 'a',
             'augmentation_rng_after_sha256': 'b'}
    opts = dict(epoch=1, index=157, cursor=1254, tensor_count=556, lr=6e-5, previous='a')
    assert reconcile_window(event, names, **opts)[0] == names
    with pytest.raises(ValueError, match='schedule'):
        reconcile_window(event, list('abcdeg'), **opts)
    with pytest.raises(ValueError, match='CE totals'):
        reconcile_window(dict(event, mean_pixel_loss=3.5), names, **opts)


def test_shared_cluster_contrast_and_absolute_risk_use_correct_denominators():
    ssm = np.array([[100, 8, 3, 5, 12], [20, 2, 4, -2, 7]])
    cnn = np.array([[100, 10, 4, 6, 15], [20, 0, 4, -4, 4]])
    weights = np.array([[2, 0], [0, 2], [1, 1]])
    out = rate_contrast(ssm, cnn, weights)
    assert out['new']['point'] == 0
    assert out['new']['ci95'] == pytest.approx(np.percentile([-.02, .1, 0], [2.5, 97.5]))
    assert out['net']['point'] == pytest.approx(1 / 120)
    with pytest.raises(ValueError, match='denominators'):
        rate_contrast(ssm, cnn[::-1], weights)
    risk = risk_values(np.array([100, 8, 3, 5, 12, 13, 18]))
    assert risk['clean_error_rate'] == pytest.approx(.13)
    assert risk['corrupt_error_rate'] == pytest.approx(.18)
    assert risk['relative_net_vs_clean_error'] == pytest.approx(5 / 13)
    assert risk['new_per_clean_correct'] == pytest.approx(8 / 87)
