import hashlib
import json

import pytest

from experiments.r3_d1_r7 import run_remaining as r7


def test_new_queue_order_excludes_historical_seed():
    assert r7.RUNS == [(arm, seed) for seed in (3408, 3409, 3410, 3411) for arm in 'ABCD']


def test_statistics_pair_exact_seed_and_distinguish_endpoints():
    rows = []
    for seed in range(3407, 3412):
        for arm in 'ABCD':
            base = (seed - 3400) / 100
            best = {'ap50': base + (.2 if arm == 'D' else 0), 'ap50_95': base * 2}
            end = {'ap50': base, 'ap50_95': base + (.1 if arm == 'C' else 0)}
            rows.append(dict(arm=arm, seed=seed, best_validation=best, completed_epochs=[best,end]))
    result = r7.statistics_for(rows[::-1])
    assert result['best']['paired_summary']['ap50']['mean'] == pytest.approx(.2)
    assert result['best']['paired_summary']['ap50']['sample_std'] == pytest.approx(0)
    assert result['epoch_10']['paired_summary']['ap50_95']['mean'] == pytest.approx(-.1)
    assert result['best']['arms']['C']['ap50']['mean'] == pytest.approx(.09)
    assert result['best']['arms']['C']['ap50']['sample_std'] == pytest.approx(.0158113883)


@pytest.mark.parametrize('damage', ['signature', 'object', 'prefix'])
def test_recovery_requires_signature_object_and_committed_events(tmp_path, damage):
    obj = b'test-only checkpoint placeholder'
    digest = hashlib.sha256(obj).hexdigest()
    logs = b'committed\n'
    (tmp_path/'checkpoint-objects').mkdir()
    (tmp_path/'checkpoint-objects'/f'{digest}.pt').write_bytes(obj)
    (tmp_path/'last.pt').write_bytes(obj)
    (tmp_path/'events.jsonl').write_bytes(logs+b'uncommitted\n')
    r7.write(tmp_path/'runtime.json', {'runtime': 'test'})
    r7.write(tmp_path/'initialization.json', {'signature_sha256': 'approved'})
    r7.write(tmp_path/'checkpoint-ledger.json', dict(
        signature_sha256='approved', last=digest, best=None, previous={'last': None},
        event_log={'offset': len(logs), 'sha256': hashlib.sha256(logs).hexdigest()}))
    r7.checkpoint(tmp_path, 'approved', {'runtime': 'test'})
    signature = 'approved'
    if damage=='signature':
        signature = 'wrong'
    elif damage=='object':
        (tmp_path/'checkpoint-objects'/f'{digest}.pt').write_bytes(b'corrupt')
    else:
        (tmp_path/'events.jsonl').write_bytes(b'wrong')
    with pytest.raises(RuntimeError, match='differs'):
        r7.checkpoint(tmp_path, signature, {'runtime': 'test'})
