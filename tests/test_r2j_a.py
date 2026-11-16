import numpy as np
import pytest

from experiments.r2j_a.analyze import FIELDS, overall_miou, stratify, vector


def counts(n, add, fix):
    return dict(zip(FIELDS, (n, add, fix, add-fix, add+fix)))


def test_heterogeneity_and_group_counts_conserve_pixels():
    def row(sample, group, v):
        return dict(id=sample, group=group, outside=v, by_true_class_outside=[v]+[counts(0,0,0)]*8,
                    by_distance={'0-16':v})
    a,b = counts(90,2,5),counts(10,4,1)
    rows = [row('001D','g1',a),row('002N','g2',b)]
    z = stratify(rows,['001D','002N'],['g1','g2'],{'001D':'g1','002N':'g2'},
                 np.array([[1,1],[2,0],[0,2]]))
    assert z['all']['rate_pp']==0
    assert z['time/D']['groups_with_pixels']==1
    assert z['time/D']['valid_draws']==1
    assert z['time/N']['rate_pp']==30
    assert z['class/unlabeled/background']['counts']==z['all']['counts']
    assert z['distance/0-16']['counts']==z['all']['counts']
    assert z['visual_groups']=={'g1':a,'g2':b}


def test_refuse_impossible_pair_counts():
    with pytest.raises(ValueError, match='Contradictory'):
        vector(dict(counts(5,2,1),net=0))

def test_clean_miou_excludes_classes_with_no_union():
    assert overall_miou(np.array([[3, 1, 0], [0, 2, 0], [0, 0, 0]])) == pytest.approx((.75 + 2/3) / 2)
