import sqlite3

import numpy as np
import pytest

from experiments.r2g.journal import Journal
from experiments.r2e.analysis import contrasts


def test_journal_resume_preserves_results_and_failures(tmp_path):
    path = tmp_path / 'run.sqlite'
    identity = {'ids': ['a', 'b'], 'seed': 3407, 'fingerprint': 'first'}
    with Journal(path, identity) as journal:
        journal.save('a', {'outcome': 42})
        journal.save('b', {'error': 'failed tolerance'}, failure=True)
        with pytest.raises(sqlite3.IntegrityError):
            journal.save('a', {'outcome': -1})
    with Journal(path, identity) as journal:
        assert journal.read()['a']['outcome'] == 42
        assert journal.read(failures=True)['b']['error'] == 'failed tolerance'
    with pytest.raises(ValueError, match='identity differs'):
        Journal(path, dict(identity, fingerprint='second'))


def test_journal_rejects_corrupted_row(tmp_path):
    path = tmp_path / 'run.sqlite'
    identity = {'ids': ['a']}
    with Journal(path, identity) as journal:
        journal.save('a', {'outcome': 42})
    with sqlite3.connect(path) as db:
        db.execute("UPDATE images SET payload = '{}'")
    with Journal(path, identity) as journal:
        with pytest.raises(ValueError, match='row digest differs'):
            journal.read()


def test_factorial_interactions_subtract_lower_order_terms():
    # Main effects 1, 2, 4. Pair effects 8, 16, 32 and triple effect 64.
    effects = [0, 1, 2, 8, 4, 16, 32, 64]
    values = {f'encoder/{m}': {'net': np.array([sum(effects[j] for j in range(8) if j & m == j)])}
              for m in range(1, 8)}
    result = contrasts(values)
    assert result['encoder/h-minus-c']['net'].item() == -1
    assert result['encoder/h-minus-d']['net'].item() == -3
    for mask in (3, 5, 6, 7):
        assert result[f'encoder/interaction-{mask}']['net'].item() == effects[mask]


def test_scope_contrasts_pair_matching_masks():
    pytest.importorskip('triton')
    from experiments.r2g.analysis import scope_contrasts
    from experiments.r2g.replay import scopes

    values = {'natural_corrupt': {'new': np.array([99.])}}
    increments = {'encoder': 3, 'decoder': 2, 'shared': 5, 'fusion': 8}
    for scope in scopes('ssm'):
        for mask in range(1, 8):
            values[f'{scope}/{mask}'] = {'new': np.array([mask * increments.get(scope, 0)])}
    results = scope_contrasts(values, 'ssm')
    assert results['encoder-minus-decoder/7']['new'].item() == 7
    assert results['fusion-minus-shared/7']['new'].item() == 21
    assert 'fusion' not in scopes('cnn') and 'fusion' not in scopes('attention')
    assert set(scopes('ssm')['all']) == set(scopes('ssm')['shared'] + scopes('ssm')['fusion'])

