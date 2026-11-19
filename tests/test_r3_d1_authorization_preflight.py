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


@pytest.mark.parametrize('old,new',[
    ('"status": "ACTIVE"','"status": "HISTORICAL"'),
    ('"superseded_rule_active": false','"superseded_rule_active": true'),
    ('"backend": "sigma-r3-exclusive-block-pairwise-v1"','"backend": "native"'),
    ('"atol": 2e-5','"atol": 1e-4'),
    ('"checkpoint_resume": "BITWISE"','"checkpoint_resume": "CLOSE"'),
    ('旧 D 臂 BLOCKED 条款已由本修订正式取代，不再是现行规则。','旧规则保留。'),
    (r2.ACTIVE_MARKER,'D 臂仍禁止正式运行。\n'+r2.ACTIVE_MARKER),
    (r2.END_MARKER,r2.END_MARKER+'\nD arm BLOCKED'),
])
def test_active_protocol_rejects_conflicts_and_wrong_contract(old,new):
    text=(r2.ROOT/r2.PROTOCOL).read_text(encoding='utf-8')
    assert r2.verify_protocol(text)==r2.ACTIVE_CONTRACT
    assert old in text
    with pytest.raises(ValueError):
        r2.verify_protocol(text.replace(old,new,1))


def test_persisted_signature_and_epoch_order_readback(tmp_path):
    ledger=tmp_path/'runs.json'
    roster=tuple(r2.read(r2.D0)['fit'])
    samples={s:[r2.digest(r2.batches(s,e,roster)) for e in range(10)] for s in r2.SEEDS}
    runs={}
    for arm in 'ABCD':
        for seed in r2.SEEDS:
            s=r2.signature(seed,arm,{}, {})
            runs[f'{arm}/{seed}']=dict(signature=s,sha256=r2.digest(s),fit_orders=samples[seed].copy())
    doc=dict(baseline=r2.BASE,runs=runs)
    r2.write_json(ledger,doc)
    hashes,orders=r2.verify_persisted(ledger,roster,{}, {})
    assert len(set(hashes.values()))==20 and sum(map(len,orders.values()))==50
    bad=deepcopy(doc)
    bad['runs']['A/3407']['signature']['lr']=.5
    bad['runs']['A/3407']['sha256']=r2.digest(bad['runs']['A/3407']['signature'])
    r2.write_json(ledger,bad)
    with pytest.raises(ValueError,match='signature fields'):
        r2.verify_persisted(ledger,roster,{}, {})
    doc['runs']['D/3411']['fit_orders'][9]='0'*64
    r2.write_json(ledger,doc)
    with pytest.raises(ValueError,match='epoch order'):
        r2.verify_persisted(ledger,roster,{}, {})
