import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from experiments.r3_d1 import authorization
from experiments.r3_d1.checkpoint import digest
from experiments.r3_d1.data import SPLIT_SHA256
from experiments.r3_d1.deterministic_scan import fixed_sum, installed
from experiments.r3_d1.model import SEEDS
from experiments.r3_d1.train import train


def test_odd_and_even_fixed_tree():
    x=torch.tensor([[1e20],[1.],[-1e20],[1.],[3.]])
    assert fixed_sum(x).item()==3.
    assert fixed_sum(x[:4]).item()==0.


def test_patch_cleanup_on_failure():
    core=SimpleNamespace(bwd=lambda: None)
    fn=core.bwd
    with pytest.raises(RuntimeError),installed(core):
        assert core.bwd is not fn
        raise RuntimeError('test')
    assert core.bwd is fn


@pytest.mark.parametrize('arm',list('ABCD'))
def test_all_arms_refused_before_any_training_access(arm):
    with pytest.raises(ValueError,match='blocked for all arms'):
        train(SimpleNamespace(authorize_formal_training=True,authorization=None,arm=arm,seed=3407))


@pytest.mark.parametrize('missing',sorted(authorization.REQUIRED))
def test_require_all_gates_even_for_reviewed_hash(monkeypatch,tmp_path,missing):
    sig={'test':'signature'}
    receipt=dict(schema='r3-d1b-r1-authorization-v1',split_sha256=SPLIT_SHA256,
                 gates=dict.fromkeys(authorization.REQUIRED,'PASS'),
                 run_signatures={f'{a}/{s}':digest(sig) for a in 'ABCD' for s in SEEDS})
    receipt['gates'][missing]='FAIL'
    p=tmp_path/'review.json'
    raw=json.dumps(receipt).encode()
    p.write_bytes(raw)
    monkeypatch.setattr(authorization,'TRUSTED_RECEIPTS',frozenset({hashlib.sha256(raw).hexdigest()}))
    for arm in 'ABCD':
        with pytest.raises(ValueError,match='incomplete review gates'):
            authorization.require_authorization(p,arm,3407,sig)


def test_full_review_receipt_binds_signature(monkeypatch,tmp_path):
    sig={'test':'signature'}
    receipt=dict(schema='r3-d1b-r1-authorization-v1',split_sha256=SPLIT_SHA256,
                 gates=dict.fromkeys(authorization.REQUIRED,'PASS'),
                 run_signatures={f'{a}/{s}':digest(sig) for a in 'ABCD' for s in SEEDS})
    p=tmp_path/'review.json'
    raw=json.dumps(receipt).encode()
    p.write_bytes(raw)
    with pytest.raises(ValueError,match='unreviewed'):
        authorization.require_authorization(p,'A',3407,sig)
    monkeypatch.setattr(authorization,'TRUSTED_RECEIPTS',frozenset({hashlib.sha256(raw).hexdigest()}))
    authorization.require_authorization(p,'D',3411,sig)
    with pytest.raises(ValueError,match='signature mismatch'):
        authorization.require_authorization(p,'D',3411,{})
