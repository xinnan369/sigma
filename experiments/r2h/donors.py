"""Generate and independently verify the frozen, cross-group donor assignment."""
import json

from experiments.r1.assets import sha256
from experiments.r1c.data import canonical_sha
from experiments.r1g.partition import EVIDENCE, GRAPH_SHA, ROOT, read, references, require

FILE = ROOT / 'reports/evidence/r2h/donors-v2.json'
VERSION = 'day-night-half-rotation-v2'


def build(split, graph):
    split_hash = canonical_sha({k: v for k, v in split.items() if k != 'split_sha256'})
    require(split_hash == split['split_sha256'], 'Split digest differs')
    require(canonical_sha(graph) == split['graph_sha256'] == GRAPH_SHA, 'Graph digest differs')
    ids = split['validation_ids']
    require(len(ids) == len(set(ids)) == 157, 'Validation roster differs')
    groups = {i: g[0] for g in graph['groups'] for i in g if i in ids}
    require(set(groups) == set(ids) and len(set(groups.values())) == 54, 'Visual groups differ')
    mapping = {}
    for suffix, count, offset in (('D', 90, 45), ('N', 67, 33)):
        strata = sorted(i for i in ids if i.endswith(suffix))
        require(len(strata) == count, 'Day/night roster differs')
        mapping.update(zip(strata, strata[offset:] + strata[:offset], strict=True))
    require(set(mapping) == set(mapping.values()) == set(ids)
            and all(i != d and i[-1] == d[-1] and groups[i] != groups[d]
                    for i, d in mapping.items()), 'Donor assignment is not bijective and cross-group')
    return {'version': VERSION, 'split_sha256': split_hash, 'graph_sha256': GRAPH_SHA,
            'mapping_sha256': canonical_sha(mapping), 'mapping': mapping}


def load(config, split, graph, path=FILE):
    require(config['mismatch'] == VERSION and sha256(path) == config['donor_file_sha256'],
            'Frozen donor file digest differs')
    document = read(path)
    require(document == build(split, graph), 'Frozen donor mapping differs')
    return document


if __name__ == '__main__':
    graph, _ = references()
    print(json.dumps(build(read(EVIDENCE / 'split.json'), graph), indent=2))
