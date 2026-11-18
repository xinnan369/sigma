import pytest

from experiments.r2k_q2.protocol_checks import (
    frame_input, group_oof, head_budget, matched_budget, selective_risk,
    supervision, support_weights,
)


def test_oof_assigns_whole_groups_once_and_excludes_other_partitions():
    fit = [f'g{i//3}' for i in range(39)]
    folds = group_oof(fit, ['cal'], ['test'])
    assert folds == group_oof(reversed(fit), ['cal'], ['test'])
    assert len(folds) == 5
    for train, held in folds:
        assert held and not train & held and train | held == set(fit)
    assert all(sum(g in held for _, held in folds) == 1 for g in fit)
    with pytest.raises(ValueError):
        group_oof(fit, ['g0'], ['test'])


@pytest.mark.parametrize('arm,d_mask', [('candidate', False), ('o_matched', False),
                                        ('full_support', True)])
def test_o_zero_is_not_a_device_state_and_u_uses_both_states(arm, d_mask):
    assert supervision(0, 0, .5, arm) == (d_mask, 1)
    assert supervision(0, 1, .5, arm) == (d_mask, 0)
    assert supervision(1, 1, .2, arm) == (True, 1)
    assert supervision(0, None, .2, arm) == (False, None)


def test_inference_input_is_independent_of_supervision_and_identity():
    common = {'current_frame': (7, 10)}
    a = dict(common, o=0, device_state=0, eligible=False, reference=(17,), group='a', weight=0)
    b = dict(common, o=1, device_state=1, eligible=True, reference=(23,), group='b', weight=1)
    assert frame_input(a) == frame_input(b) == (7, 10)
    with pytest.raises(KeyError):
        frame_input({'reference': (17,)})


def test_train_eval_weights_match_for_same_support_despite_group_sizes():
    groups = ['a','a','a','a','b']
    scenes = ['u','u','u','v','u']
    mask = [True,False,True,True,True]
    train = support_weights(groups, scenes, mask)
    evaluation = support_weights(tuple(groups), tuple(scenes), tuple(mask))
    assert train == evaluation == [.125, 0., .125, .25, .5]
    assert support_weights(groups,scenes,[False]*5) == [0.]*5


def test_budget_checks_parameters_work_and_measured_latency():
    head = head_budget((12,8,3),4800)
    assert head == {'parameters': 131, 'macs_per_frame': 576000}
    total = dict(head, macs_per_frame=head['macs_per_frame']+400000, median_ms=10.)
    assert matched_budget(total, dict(total, median_ms=11.))
    assert not matched_budget(total, dict(total, median_ms=11.1))
    assert not matched_budget(total, dict(total, parameters=133))
    assert not matched_budget(total, dict(total, macs_per_frame=1030000))


def test_selective_risk_requires_nonzero_accepted_mass_for_each_state():
    y, pred, weights = [0,0,1], [1,0,1], [.25,.25,.5]
    assert selective_risk(y,pred,[False]*3,weights) is None
    assert selective_risk(y,pred,[True,True,False],weights) is None
    assert selective_risk(y,pred,[True,False,True],weights) == .5
    assert selective_risk(y,pred,[True]*3,weights) == .25
