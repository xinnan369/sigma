import json

import pytest

from experiments.r3_d1_r3 import independent_audit as r3


def evidence(name):
    return json.loads((r3.ROOT/(r3.OLD+name)).read_bytes())


def test_protocol_marks_native_block_superseded():
    original=(r3.ROOT/r3.PROTOCOL).read_bytes()
    assert r3.protocol(original)['checkpoint_resume']=='BITWISE'
    with pytest.raises(ValueError,match='contract'):
        r3.protocol(original.replace(b'"superseded_rule_active": false',
                                     b'"superseded_rule_active": true'))


def test_raw_gradients_cannot_silently_disappear():
    n=evidence('numerics.json')
    assert r3.numeric(n)['parameter_tensors']==155
    del n['raw'][0]['candidate_vs_native'][0]['du']
    with pytest.raises(ValueError,match='Gradient matrix incomplete'):
        r3.numeric(n)


def test_resume_requires_every_arm_and_actual_native_calls():
    r=evidence('resume.json')
    assert r3.recovery(r)==list('ABCD')
    r['arms']['D']['workers'][0]['native_calls']['bwd']=0
    with pytest.raises(ValueError,match='Native calls D'):
        r3.recovery(r)
