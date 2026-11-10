"""All-pair RGB/T visual proxy plus candidate geometric corroboration."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from experiments.r1c.data import base_id, canonical_sha, read_lists, write_json
from experiments.r1f.split import choose, components, verify

ROOT = Path(__file__).resolve().parents[2]
RULES = {'joint_corr': .90, 'near_id_joint_corr': .80, 'near_id_max_gap': 5,
         'retrieval_top_each': 8, 'sift_ratio': .72, 'minimum_inliers': 18,
         'minimum_inlier_fraction': .55, 'minimum_hull_fraction': .06,
         'minimum_warp_overlap': .60, 'minimum_other_modality_corr': .72}


def norm(image):
    x = np.asarray(image, dtype=np.float32).ravel().copy()
    x -= x.mean()
    size = np.linalg.norm(x)
    return x / size if size else x


def load_images(root, ids, digest):
    labels, images, hashes = {}, [], []
    metadata, exact_flips, mismatched = Counter(), Counter(), []
    for name in ids:
        pair, sig = [], [[], []]
        for folder in ('RGB', 'Modal', 'Label'):
            values = []
            for suffix in ('', '_flip'):
                rel = f'{folder}/{name}{suffix}.png'
                raw = (root / rel).read_bytes()
                if hashlib.sha256(raw).hexdigest() != digest[rel]:
                    raise ValueError(f'Pinned source bytes changed: {rel}')
                with Image.open(io.BytesIO(raw)) as im:
                    metadata.update(im.info.keys())
                    values.append(np.asarray(im))
            original, flipped = values
            if np.array_equal(original[:, ::-1], flipped):
                exact_flips[folder] += 1
            else:
                mismatched.append(f'{folder}/{name}_flip.png')
            if folder == 'Label':
                if original.ndim != 2 or not set(np.unique(original)) <= set(range(9)):
                    raise ValueError(f'Invalid original mask: {name}')
                labels[name] = np.bincount(original.ravel(), minlength=9).tolist()
            else:
                for orient in (0, 1):
                    x = original if orient == 0 else original[:, ::-1]
                    sig[orient].append(hashlib.sha256(x.tobytes()).hexdigest())
                gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY) if folder == 'RGB' else original
                pair.append(cv2.resize(gray, (320, 240), interpolation=cv2.INTER_AREA))
        images.append(pair)
        hashes.append(sig)
    return images, labels, hashes, dict(metadata), dict(exact_flips), mismatched


def global_scores(images):
    vectors = np.stack([[[norm(cv2.resize(x if o == 0 else x[:, ::-1], (64, 48), interpolation=cv2.INTER_AREA))
                          for x in pair] for o in (0, 1)] for pair in images])
    return [np.stack([np.clip(vectors[:, 0, m] @ vectors[:, o, m].T, -1, 1)
                      for m in (0, 1)]) for o in (0, 1)]


def near_id(a, b):
    return a[-1] == b[-1] and abs(int(a[:-1]) - int(b[:-1])) <= RULES['near_id_max_gap']


def geometric(left, right, features_l, features_r):
    for channel in (0, 1):
        pa, da = features_l[channel]
        pb, db = features_r[channel]
        if da is None or db is None or min(len(da), len(db)) < RULES['minimum_inliers']:
            continue
        matches = cv2.BFMatcher().knnMatch(da, db, k=2)
        selected = [a for a, b in matches if a.distance < RULES['sift_ratio'] * b.distance]
        if len(selected) < RULES['minimum_inliers']:
            continue
        x = np.float32([pa[m.queryIdx].pt for m in selected])
        y = np.float32([pb[m.trainIdx].pt for m in selected])
        transform, inliers = cv2.findHomography(x, y, cv2.RANSAC, 2.5, maxIters=750)
        if transform is None or not np.isfinite(transform).all():
            continue
        good = inliers.ravel().astype(bool)
        count = int(good.sum())
        if count < RULES['minimum_inliers'] or good.mean() < RULES['minimum_inlier_fraction']:
            continue
        hull = min(cv2.contourArea(cv2.convexHull(z[good])) for z in (x, y)) / (320 * 240)
        if hull < RULES['minimum_hull_fraction']:
            continue
        corners = np.float32([[[0, 0], [319, 0], [319, 239], [0, 239]]])
        projected = cv2.perspectiveTransform(corners, transform)[0]
        area = cv2.contourArea(projected, oriented=True) / (320 * 240)
        if not cv2.isContourConvex(projected) or not .55 <= area <= 1.8:
            continue
        mask = cv2.warpPerspective(np.ones((240, 320), dtype=np.uint8), transform, (320, 240)).astype(bool)
        if mask.mean() < RULES['minimum_warp_overlap']:
            continue
        other = 1 - channel
        warped = cv2.warpPerspective(left[other], transform, (320, 240))
        corr = float(norm(warped[mask]) @ norm(right[other][mask]))
        if corr >= RULES['minimum_other_modality_corr']:
            return {'feature_modality': ('RGB', 'Thermal')[channel], 'inliers': count,
                    'inlier_fraction': float(good.mean()), 'minimum_hull_fraction': hull,
                    'warp_overlap': float(mask.mean()), 'other_modality_corr': corr}
    return None


def group_images(ids, images, hashes):
    global_views = global_scores(images)
    joint = [np.min(s, axis=0) for s in global_views]
    candidates = set()
    for scores in [np.maximum(*joint)] + [np.maximum(global_views[0][m], global_views[1][m]) for m in (0, 1)]:
        np.fill_diagonal(scores, -np.inf)
        for i, order in enumerate(np.argsort(-scores, axis=1, kind='stable')[:, :RULES['retrieval_top_each']]):
            candidates.update(tuple(sorted((i, int(j)))) for j in order)
    edges, exact_pairs, near_pairs = [], 0, 0
    accepted = set()
    for i, a in enumerate(ids):
        for j in range(i + 1, len(ids)):
            b = ids[j]
            adjacent = near_id(a, b)
            near_pairs += adjacent
            if adjacent:
                candidates.add((i, j))
            orient = int(joint[1][i, j] > joint[0][i, j])
            score = float(joint[orient][i, j])
            exact = [o for o in (0, 1) if hashes[i][0] == hashes[j][o]]
            exact_pairs += bool(exact)
            reason = ('decoded_exact_pair' if exact else 'dual_global' if score >= RULES['joint_corr']
                      else 'number_neighbor_with_dual_visual_match' if adjacent and score >= RULES['near_id_joint_corr']
                      else None)
            if reason:
                edges.append({'a': a, 'b': b, 'reason': reason, 'flip_b': bool(exact[0] if exact else orient),
                              'joint_corr': score})
                accepted.add((i, j))

    sift = cv2.SIFT_create(nfeatures=240, contrastThreshold=.015)
    image_views = [(pair, [x[:, ::-1].copy() for x in pair]) for pair in images]
    features = [[tuple(sift.detectAndCompute(x, None) for x in view) for view in pair] for pair in image_views]
    checked = 0
    for i, j in sorted(candidates - accepted):
        checked += 1
        # Try both orientations, prioritizing the stronger joint retrieval score.
        first = int(joint[1][i, j] > joint[0][i, j])
        for o in (first, 1 - first):
            match = geometric(image_views[i][0], image_views[j][o], features[i][0], features[j][o])
            if match:
                edges.append({'a': ids[i], 'b': ids[j], 'reason': 'dual_geometric', 'flip_b': bool(o),
                              'joint_corr': float(joint[o][i, j]), **match})
                break
    return edges, {'all_pairs': len(ids) * (len(ids) - 1) // 2, 'exact_pairs': exact_pairs,
                   'numeric_neighbor_pairs': near_pairs, 'geometric_candidates': len(candidates),
                   'geometric_checked_after_global': checked}, joint


def audit(root, output):
    cv2.setNumThreads(2)
    cv2.setRNGSeed(3407)
    lists = read_lists(root)
    ids = sorted({base_id(x) for x in lists['train.txt']})
    if len(ids) != 784 or set(lists['train.txt']) != {i + s for i in ids for s in ('', '_flip')}:
        raise ValueError('Unexpected train.txt original/flip inventory')
    evidence = ROOT / 'reports/evidence/r1c'
    manifest = json.loads((evidence / 'manifest.json').read_text())
    if manifest['errors'] or canonical_sha(manifest['files']) != manifest['files_sha256']:
        raise ValueError('R1c manifest altered')
    files = {f['path']: f['sha256'] for f in manifest['files']}
    images, labels, hashes, metadata, flips, mismatches = load_images(root, ids, files)
    print(f'Loaded {len(ids)} verified original/flip triplets; mismatches={len(mismatches)}', flush=True)
    edges, counts, joint = group_images(ids, images, hashes)
    pairs = [(e['a'], e['b']) for e in edges]
    groups = components(ids, pairs)
    previous = set(json.loads((evidence / 'split.json').read_text())['validation_ids'])
    audit_result = {'scope': 'visual scene grouping proxy; no capture-sequence or temporal ground truth',
                    'test_pixels_read': 0, 'r1c_files_sha256': manifest['files_sha256'], 'originals': len(ids),
                    'rules': RULES, **counts, 'png_metadata_keys': metadata, 'exact_flips': flips,
                    'flip_mismatches': mismatches, 'groups': groups, 'edges': edges, 'label_pixels': labels,
                    'r1e_edges_crossing': sum((a in previous) != (b in previous) for a, b in pairs),
                    'reasons': dict(Counter(e['reason'] for e in edges))}
    print(f'Visual groups={len(groups)}, edges={len(edges)}, largest={max(map(len, groups))}', flush=True)
    write_json(output / 'scene-audit.json', audit_result)
    if mismatches:
        print('BLOCKED: author augmentation mismatch requires provenance review')
        return 2
    val, solver = choose(groups, labels)
    audit_result['solver'] = solver
    if val is None:
        write_json(output / 'scene-audit.json', audit_result)
        print(f'BLOCKED: no feasible supported partition: {solver}')
        return 2
    val_set = set(val)
    train = sorted(set(ids) - val_set)
    indices = {x: i for i, x in enumerate(ids)}
    cross = np.maximum(*joint)[np.ix_([indices[i] for i in val], [indices[i] for i in train])]
    i, j = np.unravel_index(np.argmax(cross), cross.shape)
    audit_result['r1f_max_cross_split_joint'] = {'a': val[i], 'b': train[j], 'score': float(cross[i, j])}
    audit_result['r1f_cross_split_numeric_neighbors'] = sum(near_id(a, b) for a in val for b in train)
    audit_result['r1e_cross_split_numeric_neighbors'] = sum(near_id(a, b) for a in previous for b in set(ids) - previous)
    write_json(output / 'scene-audit.json', audit_result)
    split = {'scope': 'visual scene grouping proxy', 'train_originals': train, 'validation_ids': val,
             'train_ids': sorted(i for i in lists['train.txt'] if base_id(i) not in val_set),
             'graph_sha256': canonical_sha(audit_result), 'r1c_files_sha256': manifest['files_sha256']}
    split['split_sha256'] = canonical_sha(split)
    stats = verify(split, groups, pairs, lists['train.txt'], lists['test.txt'], labels)
    write_json(output / 'split.json', split)
    write_json(output / 'isolation.json', {'status': 'PASS', 'split_sha256': split['split_sha256'],
                                          'edge_overlap': 0, 'identity_overlap': 0, 'test_pixels_read': 0,
                                          'class_support': stats})
    print(json.dumps({'status': 'PASS', 'split_sha256': split['split_sha256'], 'support': stats, **counts,
                      'maximum_cross_split_correlation': audit_result['r1f_max_cross_split_joint']}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    return audit(args.root, args.output)


if __name__ == '__main__':
    raise SystemExit(main())
