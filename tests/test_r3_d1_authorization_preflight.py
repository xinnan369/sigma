from copy import deepcopy

import pytest

from experiments.r3_d1 import authorization_preflight as r2
from experiments.r3_d1.train import train


def recovery():
    d=dict(status='PASS_SYNTHETIC_ONLY',backend='deterministic',official_test_pixel_reads=0,
           real_llvip_optimizer_updates=0,formal_training_started=False,
           environment={},upstream={},source_sha256={},arms={})
    for arm in 'ABCD':
        d['arms'][arm]=dict(status='BITWISE_EQUAL',update=2,bn_buffers=True,optimizer=True,
            all_rng=True,event_log_prefix=True,workers=[
                dict(arm=arm,phase=p,updates=n,restored_before_update_exact=p=='resume',
                     native_calls=dict(fwd=9*m if arm=='D' else 0,bwd=36*m if arm=='D' else 0))
                for p,n,m in [('continuous',2,2),('interrupt',1,1),('resume',2,1)]])
    return d


@pytest.mark.parametrize('fault',['backend','test','arm','scan','source'])
def test_candidate_cannot_promote_invalid_recovery(fault):
    doc=recovery()
    sig={k:deepcopy(doc[k]) for k in ('environment','upstream','source_sha256')}
    r2.verify_resume(doc,sig)
    if fault=='backend':
        doc['backend']='native'
    elif fault=='test':
        doc['official_test_pixel_reads']=1
    elif fault=='arm':
        del doc['arms']['B']
    elif fault=='scan':
        doc['arms']['D']['workers'][0]['native_calls']['bwd']=18
    else:
        doc['source_sha256']['other.py']='0'*64
    with pytest.raises(ValueError):
        r2.verify_resume(doc,sig)


def test_candidate_is_not_authority_for_any_arm(tmp_path):
    from types import SimpleNamespace
    from experiments.r3_d1.review_registry import TRUSTED_RECEIPTS
    assert TRUSTED_RECEIPTS==frozenset()
    candidate=tmp_path/'candidate.json'
    r2.write_json(candidate,dict(schema='r3-d1b-r1-authorization-v1',gates=dict.fromkeys(r2.REQUIRED,'PASS')))
    for arm in 'ABCD':
        with pytest.raises(ValueError,match='unreviewed authorization'):
            train(SimpleNamespace(authorize_formal_training=True,authorization=candidate,
                                  arm=arm,seed=3407))
