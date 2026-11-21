"""CPU-only checks for metrics and anti-leakage gates."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.r2d_official_test.metrics import compare_runs, confusion, metrics
from experiments.r2d_official_test.audit import PIXELS_PER_IMAGE, inspect_events


def test_perfect_confusion():
    x = np.array([[0,1,2],[2,1,0]])
    mat = confusion(x,x)
    assert np.trace(mat)==6
    assert metrics(mat)['miou']==1
    assert metrics(mat)['foreground_miou']==1


def test_confusion_records_misclassification():
    gt=np.array([[0,1],[1,2]])
    pred=np.array([[1,1],[0,2]])
    hist=confusion(gt,pred)
    assert hist[0,1]==1 and hist[1,0]==1 and hist.sum()==4
    assert 0<metrics(hist)['miou']<1


def test_ignore_label_255():
    hist=confusion(np.array([[0,255],[1,2]]),np.array([[0,8],[1,2]]))
    assert hist.sum()==3


@pytest.mark.parametrize('gt,pred',[
    (np.array([[9]]),np.array([[0]])),
    (np.array([[-1]]),np.array([[0]])),
    (np.array([[0]]),np.array([[9]])),
    (np.array([[0.5]]),np.array([[0]])),
    (np.array([[0]]),np.array([[0.5]])),
    (np.array([[0]]),np.array([[0,1]])),
])
def test_confusion_rejects_bad_labels(gt,pred):
    with pytest.raises(ValueError): confusion(gt,pred)


def test_all_absent_class_iou_null():
    result=metrics(np.diag([4]+[0]*8))
    assert result['class_present_count']==1
    assert result['foreground_miou'] is None
    assert result['iou_0_to_8'][1] is None


def test_empty_invalid_matrix():
    with pytest.raises(ValueError):metrics(np.zeros((9,9),dtype='int64'))
    with pytest.raises(ValueError):metrics(np.full((9,9),-1,dtype='int64'))


def example_rows():
    return [{'seed':seed,'arm':arm,'metrics':{'miou':0.55+seed/100000+(i/100),
                                             'foreground_miou':0.5+seed/100000+(i/100),
                                             'iou_0_to_8':[0.55+seed/100000+i/100]*9}}
            for seed in range(3407,3412) for i,arm in enumerate(('ssm','cnn','attention'))]


def test_all_15_required_and_paired():
    result=compare_runs(example_rows())
    assert result['status']=='COMPLETE'
    assert len(result['paired_statistics'])==3
    assert abs(result['paired_statistics']['ssm_minus_cnn']['miou']['mean']+0.01)<1e-10


def test_duplicate_or_missing_run_rejected():
    x=example_rows()
    for bad in (x[:-1],x[:-1]+[x[0]]):
        with pytest.raises(ValueError):compare_runs(bad)


def test_nonfinite_scores_rejected():
    x=example_rows()
    x[0]['metrics']['miou']=float('nan')
    with pytest.raises(ValueError):compare_runs(x)


def test_independent_audit_rejects_repeated_id_and_lost_pixels(tmp_path):
    events = tmp_path / 'predictions.jsonl'
    matrix = [[0] * 9 for _ in range(9)]
    matrix[0][0] = PIXELS_PER_IMAGE
    rows = [dict(position=i, id=name, prediction_sha256='a' * 64, confusion_matrix=matrix)
            for i, name in enumerate(('one', 'two'), 1)]

    def write_rows():
        events.write_text('\n'.join(map(json.dumps, rows)) + '\n', encoding='utf-8')

    write_rows()
    assert inspect_events(events, ['one', 'two'])[0][0] == 2 * PIXELS_PER_IMAGE
    rows[1]['id'] = 'one'
    write_rows()
    with pytest.raises(ValueError, match='invalid identity'):
        inspect_events(events, ['one', 'two'])
    rows[1]['id'] = 'two'
    matrix[0][0] -= 1
    write_rows()
    with pytest.raises(ValueError, match='invalid pixel counts'):
        inspect_events(events, ['one', 'two'])
