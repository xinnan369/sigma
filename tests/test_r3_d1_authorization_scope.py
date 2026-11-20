import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.r3_d1 import authorization, train
from experiments.r3_d1.checkpoint import hashfile

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / 'reports/evidence/r3-d1b-r2r/authorization-candidate.json'
RUNS = json.loads((CANDIDATE.parent / 'run-signatures.json').read_bytes())['runs']
APPROVAL = ROOT / 'reports/evidence/r3-d1b-r6/human-approval.json'
ARMS = ('A', 'B', 'C', 'D')


def hardened_signature(arm):
    sig = RUNS[f'{arm}/3407']['signature']
    source = dict(sig['source_sha256'])
    path = 'experiments/r3_d1/authorization.py'
    source[path] = hashfile(ROOT / path)
    return dict(sig, source_sha256=source)


def launch_args(tmp_path, **changes):
    args = dict(arm='A', seed=3407, authorization=CANDIDATE, authorize_formal_training=True,
                resume=False, root=tmp_path / 'no-data', split=tmp_path / 'incorrect-split.json',
                upstream=tmp_path / 'no-sigma', output=tmp_path / 'runs')
    return SimpleNamespace(**(args | changes))


def unexpected_access(*args, **kwargs):
    pytest.fail('Rejected request accessed training or checkpoint state')


def simulate_runtime(monkeypatch, signature):
    # Exercise the runner's second admission check without starting CUDA.
    monkeypatch.setattr(train.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(train.torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(train, 'reproducible', lambda *args: None)
    monkeypatch.setattr(train, 'runtime', lambda cuda: {'cublas_workspace': ':4096:8'})
    monkeypatch.setattr(train, 'load_sigma', lambda upstream: (None, None, {}))
    monkeypatch.setattr(train, 'signature', lambda *args: signature)
    monkeypatch.setattr(train, 'run_lock', unexpected_access)


def test_registry_matches_human_scope():
    approval = json.loads(APPROVAL.read_bytes())
    sha = hashfile(CANDIDATE)
    assert set(authorization.TRUSTED_RECEIPTS) == {approval['approved_sha256']} == {sha}
    assert set(authorization.TRUSTED_RECEIPTS[sha]) == set(approval['launch_scope']) == {
        f'{arm}/3407' for arm in ARMS
    }


@pytest.mark.parametrize('arm', ARMS)
def test_approved_scope_and_hardened_signature(arm):
    authorization.require_authorization(CANDIDATE, arm, 3407)
    authorization.require_authorization(CANDIDATE, arm, 3407, hardened_signature(arm))


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('seed', range(3408, 3412))
def test_candidate_signatures_outside_approved_scope_never_start_or_resume(arm, seed, tmp_path, monkeypatch):
    monkeypatch.setattr(train, 'external', unexpected_access)
    monkeypatch.setattr(train, 'Checkpoints', unexpected_access)
    reviewed_signature = RUNS[f'{arm}/{seed}']['signature']
    with pytest.raises(ValueError, match='outside the approved run scope'):
        authorization.require_authorization(CANDIDATE, arm, seed, reviewed_signature)
    for resume in (False, True):
        args = launch_args(tmp_path, arm=arm, seed=seed, resume=resume)
        with pytest.raises(ValueError, match='outside the approved run scope'):
            train.train(args)
    assert not args.output.exists()


@pytest.mark.parametrize(('field', 'value'), [('seed', 3408), ('arm', 'B'), ('split_sha256', '0' * 64)])
def test_approved_run_rejects_wrong_signature(field, value):
    signature = dict(hardened_signature('A'), **{field: value})
    with pytest.raises(ValueError, match='signature mismatch'):
        authorization.require_authorization(CANDIDATE, 'A', 3407, signature)


def test_approved_run_rejects_wrong_split_file(tmp_path, monkeypatch):
    simulate_runtime(monkeypatch, hardened_signature('A'))
    args = launch_args(tmp_path)
    args.split.write_bytes(b'{}')
    with pytest.raises(ValueError, match='Frozen D0 split changed'):
        train.train(args)
    assert not args.output.exists()


def test_resume_checks_runtime_signature_before_checkpoint_access(tmp_path, monkeypatch):
    simulate_runtime(monkeypatch, hardened_signature('B'))
    monkeypatch.setattr(train, 'DevelopmentData', unexpected_access)
    monkeypatch.setattr(train, 'Checkpoints', unexpected_access)
    args = launch_args(tmp_path, resume=True)
    with pytest.raises(ValueError, match='signature mismatch'):
        train.train(args)
    assert not args.output.exists()


def test_modified_receipt_cannot_authorize_even_with_approved_signature(tmp_path):
    path = tmp_path / 'unregistered.json'
    path.write_bytes(CANDIDATE.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='unreviewed authorization'):
        authorization.require_authorization(path, 'A', 3407, hardened_signature('A'))
