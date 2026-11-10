"""Partition whole visual components while retaining label and day/night support."""
import numpy as np

from experiments.r1c.data import base_id, canonical_sha


def components(ids, edges):
    parent = {name: name for name in ids}

    def root(name):
        while parent[name] != name:
            name = parent[name]
        return name

    for left, right in edges:
        a, b = sorted((root(left), root(right)))
        parent[b] = a
    groups = {}
    for name in sorted(ids):
        groups.setdefault(root(name), []).append(name)
    return list(groups.values())


def support(ids, groups, labels):
    members = set(ids)
    histogram = np.array([labels[i] for i in ids])
    present = histogram > 0
    return {'images': present.sum(0).tolist(), 'pixels': histogram.sum(0).tolist(),
            'groups': np.sum([np.any([np.array(labels[i]) > 0 for i in g], axis=0)
                              for g in groups if set(g) <= members], axis=0).tolist(),
            'day': sum(i.endswith('D') for i in ids), 'night': sum(i.endswith('N') for i in ids)}


def choose(groups, labels):
    from scipy.optimize import Bounds, LinearConstraint, milp

    # Deterministic label-presence balance; no model predictions or validation
    # scores are used. Disjoint group count and image counts are hard minima.
    sizes = np.array([len(g) for g in groups])
    day = np.array([sum(x.endswith('D') for x in g) for g in groups])
    presence = np.array([np.sum([np.array(labels[i]) > 0 for i in g], axis=0) for g in groups]).T
    features = np.vstack((day, presence))
    total = features.sum(1)
    target = np.maximum(np.array([41] + [3] * 9), np.rint(total * 79 / sizes.sum()))
    target[0] = 41
    group_presence = (presence > 0).astype(int)
    hard = np.vstack((sizes, day, presence, group_presence))
    lower = [79, 25] + [3] * 9 + [2] * 9
    upper = [79, 54] + (presence.sum(1) - 3).tolist() + (group_presence.sum(1) - 2).tolist()
    k, n = len(features), len(groups)
    zeros = np.zeros((len(hard), k))
    matrix = np.block([[hard, zeros], [features, -np.eye(k)], [-features, -np.eye(k)]])
    constraint = LinearConstraint(matrix, np.r_[lower, [-np.inf] * (2 * k)], np.r_[upper, target, -target])
    objective = np.r_[np.arange(n) / max(n, 1) * 1e-7, 1 / np.maximum(target, 1)]
    result = milp(objective, integrality=np.r_[[1] * n, [0] * k],
                  bounds=Bounds(0, np.r_[[1] * n, [np.inf] * k]),
                  constraints=constraint, options={'time_limit': 60, 'mip_rel_gap': 1e-5})
    info = {'status': int(result.status), 'message': result.message, 'target_images': target.astype(int).tolist()}
    if result.x is None:
        return None, info
    membership = np.rint(result.x[:n])
    if not np.allclose(result.x[:n], membership) or not np.all((hard @ membership >= lower) & (hard @ membership <= upper)):
        raise ValueError('Optimizer did not yield a feasible integral split')
    info.update(objective=float(result.fun), gap=float(result.mip_gap))
    return sorted(i for use, group in zip(membership, groups) if use for i in group), info


def verify(split, groups, edges, train_list, test_list, labels):
    train = split['train_originals']
    val = split['validation_ids']
    a, b = set(train), set(val)
    known = {base_id(i) for i in train_list}
    if (len(train) != 705 or len(a) != 705 or len(val) != 79 or len(b) != 79
            or a & b or a | b != known or (a | b) & {base_id(i) for i in test_list}
            or split['train_ids'] != sorted(i for i in train_list if base_id(i) in a)
            or len(split['train_ids']) != 1410):
        raise ValueError('Original/flip/test membership violation')
    flat = [i for g in groups for i in g]
    if sorted(flat) != sorted(known) or any(set(g) & a and set(g) & b for g in groups):
        raise ValueError('Component partition invalid or crosses split')
    if any((x in a) != (y in a) for x, y in edges):
        raise ValueError('Visual edge crosses split')
    stats = {'train': support(train, groups, labels), 'validation': support(val, groups, labels)}
    for partition in stats.values():
        if min(partition['images']) < 3 or min(partition['groups']) < 2:
            raise ValueError('Insufficient class support across visual groups')
    if min(stats['validation']['day'], stats['validation']['night']) < 25:
        raise ValueError('Insufficient D/N support')
    if split['split_sha256'] != canonical_sha({k: v for k, v in split.items() if k != 'split_sha256'}):
        raise ValueError('Frozen split digest mismatch')
    return stats
