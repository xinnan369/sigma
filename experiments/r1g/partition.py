"""MILP feasibility and independent certification of the expanded holdout."""
import argparse
import json
from pathlib import Path
import warnings

import numpy as np

from experiments.r1c.data import base_id, canonical_sha, write_json
from experiments.r1f.split import components, support

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'reports/evidence/r1g'
GRAPH_SHA = 'eeef7f60543b90410e81c04cdd4e1caf03cf4594b051e24a4def4f93338a8baf'
R1F_SPLIT_SHA = '648ba9e27e6f020f37052d5f83d517553296351b324aa774572878ff6bf97d59'
ASSETS_SHA = '981e56ff3c783f4659e41ca235bd568ff9919a078661d8d929e3fdff0633591d'
VAL, TRAIN, UPDATES = 157, 1254, 157


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def references():
    root = ROOT / 'reports/evidence/r1f'
    graph, prior = read(root / 'scene-audit.json'), read(root / 'split.json')
    require(canonical_sha(graph) == prior['graph_sha256'] == GRAPH_SHA, 'Frozen R1f graph changed')
    require(prior['split_sha256'] == R1F_SPLIT_SHA
            and canonical_sha({k: v for k, v in prior.items() if k != 'split_sha256'}) == R1F_SPLIT_SHA,
            'Frozen R1f split changed')
    require(graph['r1c_files_sha256'] == prior['r1c_files_sha256'] == ASSETS_SHA, 'MFNet asset digest differs')
    return graph, prior


def selected_files(split):
    manifest = read(ROOT / 'reports/evidence/r1c/manifest.json')
    require(not manifest['errors'] and canonical_sha(manifest['files']) == ASSETS_SHA, 'Frozen asset manifest changed')
    digests = {r['path']: r['sha256'] for r in manifest['files']}
    paths = sorted(f'{folder}/{sample}.png'
                   for sample in split['train_ids'] + split['validation_ids']
                   for folder in ('RGB', 'Modal', 'Label'))
    return [{'path': p, 'sha256': digests[p]} for p in paths]


def constraint_arrays(groups, labels, retained, target=VAL):
    sizes = np.array([len(g) for g in groups])
    day = np.array([sum(i.endswith('D') for i in g) for g in groups])
    night = sizes - day
    images = np.array([np.sum([np.array(labels[i]) > 0 for i in g], axis=0) for g in groups]).T
    scene_counts = (images > 0).astype(int)
    matrix = np.vstack((sizes, day, night, images, scene_counts))
    lower = np.array([target, 50, 50] + [6] * 18)
    upper = np.r_[[target, np.inf, np.inf], images.sum(1) - 6, scene_counts.sum(1) - 6]
    locked = np.array([bool(set(g) & retained) for g in groups], dtype=int)
    require(sum(sizes * locked) == len(retained), 'Retained holdout is not a union of whole visual groups')
    return matrix, lower, upper, locked


def solve(graph, prior):
    import scipy
    from scipy.optimize import Bounds, LinearConstraint, milp

    groups = graph['groups']
    retained = set(prior['validation_ids'])
    matrix, lower, upper, locked = constraint_arrays(groups, graph['label_pixels'], retained)
    options = {'threads': 1, 'random_seed': 3407, 'time_limit': 90, 'presolve': True}
    constraints = LinearConstraint(matrix, lower, upper)
    # Pure feasibility objective. Fixed group order, seed and one HiGHS thread.
    def invoke():
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Unrecognized options detected.*', category=RuntimeWarning)
            return milp(np.zeros(len(groups)), integrality=np.ones(len(groups)), bounds=Bounds(locked, 1),
                        constraints=constraints, options=options)

    result = invoke()
    info = {'status': int(result.status), 'message': result.message, 'scipy': scipy.__version__,
            'solver': 'HiGHS MILP', 'options': options, 'group_variables': len(groups),
            'locked_groups': int(locked.sum()), 'rows': len(matrix), 'objective': 'zero (feasibility only)',
            'graph_sha256': GRAPH_SHA, 'prior_split_sha256': R1F_SPLIT_SHA,
            'asset_manifest_sha256': ASSETS_SHA, 'target_val_originals': VAL,
            'min_images_each_side_each_class': 6, 'min_groups_each_side_each_class': 6,
            'min_day_and_night_in_validation': 50}
    if result.x is None:
        return dict(info, feasible=False)
    x = np.rint(result.x).astype(int)
    require(np.allclose(result.x, x, atol=1e-7, rtol=0) and np.all(x >= locked) and np.all(x <= 1)
            and np.all(matrix @ x >= lower) and np.all(matrix @ x <= upper), 'Invalid MILP witness')
    repeated = invoke()
    require(repeated.x is not None and np.array_equal(x, np.rint(repeated.x)), 'MILP witness not reproducible')
    chosen = [g[0] for use, g in zip(x, groups) if use]
    return dict(info, feasible=True, identical_second_solve=True, chosen_group_ids=chosen,
                constraint_row_totals=(matrix @ x).tolist())


def verify(split, graph, prior, lists):
    groups = graph['groups']
    edges = [(e['a'], e['b']) for e in graph['edges']]
    original = set(graph['label_pixels'])
    train, val = split['train_originals'], split['validation_ids']
    a, b = set(train), set(val)
    expected = set(lists['train.txt'])
    retained = set(prior['validation_ids'])
    require(len(train) == len(a) == 627 and len(val) == len(b) == 157 and not a & b
            and a | b == original and expected == {i + s for i in original for s in ('', '_flip')}
            and retained <= b and len(b - retained) == 78
            and split['retained_ids'] == sorted(retained) and split['added_ids'] == sorted(b - retained),
            'Original membership, retained holdout or added set differs')
    require(len(lists['test.txt']) == 393 and not (a | b) & {base_id(i) for i in lists['test.txt']}
            and split['train_ids'] == sorted(i for i in expected if base_id(i) in a)
            and len(split['train_ids']) == TRAIN, 'Augmentation or official test ID crosses partitions')
    require(components(original, edges) == groups
            and all(not (set(g) & a and set(g) & b) for g in groups)
            and all((x in a) == (y in a) for x, y in edges), 'Frozen visual group or edge crosses partitions')
    require(split['graph_sha256'] == GRAPH_SHA and split['asset_manifest_sha256'] == ASSETS_SHA
            and split['prior_split_sha256'] == R1F_SPLIT_SHA
            and split['selected_assets_sha256'] == canonical_sha(selected_files(split))
            and split['split_sha256'] == canonical_sha({k: v for k, v in split.items() if k != 'split_sha256'}),
            'Frozen split digest or provenance mismatch')
    stats = {}
    for name, subset in [('train', train), ('validation', val), ('retained', sorted(retained)),
                         ('added', sorted(b - retained))]:
        stats[name] = support(subset, groups, graph['label_pixels'])
        stats[name]['visual_group_count'] = sum(set(g) <= set(subset) for g in groups)
    for name in ('train', 'validation'):
        require(min(stats[name]['images']) >= 6 and min(stats[name]['groups']) >= 6,
                'Hard class-support threshold failed')
    require(min(stats['validation']['day'], stats['validation']['night']) >= 50, 'Hard D/N threshold failed')
    return stats


def freeze(graph, prior, witness, lists):
    require(witness['feasible'] and witness['graph_sha256'] == GRAPH_SHA
            and witness['prior_split_sha256'] == R1F_SPLIT_SHA and witness['asset_manifest_sha256'] == ASSETS_SHA,
            'No valid strict MILP feasibility result')
    chosen = set(witness['chosen_group_ids'])
    require(chosen <= {g[0] for g in graph['groups']}, 'Unknown MILP group')
    val = {i for g in graph['groups'] if g[0] in chosen for i in g}
    retained = set(prior['validation_ids'])
    split = {'scope': 'expanded frozen visual proxy holdout', 'graph_sha256': GRAPH_SHA,
             'asset_manifest_sha256': ASSETS_SHA, 'prior_split_sha256': R1F_SPLIT_SHA,
             'train_originals': sorted(set(graph['label_pixels']) - val), 'validation_ids': sorted(val),
             'train_ids': sorted(i for i in lists['train.txt'] if base_id(i) not in val),
             'retained_ids': sorted(retained), 'added_ids': sorted(val - retained)}
    split['selected_assets_sha256'] = canonical_sha(selected_files(split))
    split['split_sha256'] = canonical_sha(split)
    stats = verify(split, graph, prior, lists)
    return split, {'status': 'PASS', 'split_sha256': split['split_sha256'], 'graph_sha256': GRAPH_SHA,
                   'asset_manifest_sha256': ASSETS_SHA, 'edge_crossings': 0,
                   'original_or_flip_crossings': 0, 'test_id_crossings': 0, 'test_pixels_read': 0,
                   'selected_assets_sha256': split['selected_assets_sha256'],
                   'selected_file_count': 3 * (TRAIN + VAL),
                   'class_support': stats}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('feasibility', 'certify'))
    args = parser.parse_args()
    graph, prior = references()
    if args.phase == 'feasibility':
        outcome = solve(graph, prior)
        write_json(EVIDENCE / 'milp-feasibility.json', outcome)
        print(json.dumps({k: v for k, v in outcome.items() if k != 'chosen_group_ids'}, indent=2))
        return 0 if outcome['feasible'] else 2
    stored = read(ROOT / 'reports/evidence/r1c/lists.json')['lists']
    lists = {k: v['ids'] for k, v in stored.items()}
    split, certificate = freeze(graph, prior, read(EVIDENCE / 'milp-feasibility.json'), lists)
    write_json(EVIDENCE / 'split.json', split)
    write_json(EVIDENCE / 'isolation.json', certificate)
    print(json.dumps(certificate, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
