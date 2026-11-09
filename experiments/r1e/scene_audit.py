"""Screen fixed training/holdout images for proximity, without accessing test pixels."""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from experiments.r1c.data import base_id, internal_split, read_lists, write_json


def load_vectors(root, ids, folder, metadata):
    vectors = []
    for name in ids:
        path = root / folder / (name + '.png')
        with Image.open(path) as image:
            metadata.update(image.info.keys())
            array = np.asarray(image.convert('L'))
        small = cv2.resize(array, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
        small -= small.mean()
        norm = np.linalg.norm(small)
        vectors.append(small / norm if norm else small)
    return np.stack(vectors)


def audit(root):
    split = internal_split(read_lists(root), 3407)
    train = sorted(set(map(base_id, split['train_ids'])))
    val = split['validation_ids']
    metadata = Counter()
    rgb, thermal = [load_vectors(root, train, folder, metadata) for folder in ('RGB', 'Modal')]
    vrgb, vthermal = [load_vectors(root, val, folder, metadata) for folder in ('RGB', 'Modal')]
    # Both orientations are present in the fixed training set. Use the better
    # joint score, keeping RGB and thermal in the same orientation.
    def correlations(query, vectors):
        normal = query.reshape(len(val), -1) @ vectors.reshape(len(train), -1).T
        flip = query.reshape(len(val), -1) @ vectors[:, :, ::-1].reshape(len(train), -1).T
        return np.clip(normal, -1, 1), np.clip(flip, -1, 1)

    r0, r1 = correlations(vrgb, rgb)
    t0, t1 = correlations(vthermal, thermal)
    score0, score1 = np.minimum(r0, t0), np.minimum(r1, t1)
    score = np.maximum(score0, score1)
    results = []
    for i, name in enumerate(val):
        nearby = min((x for x in train if x[-1] == name[-1]),
                     key=lambda x: (abs(int(x[:-1]) - int(name[:-1])), x))
        j = int(np.argmax(score[i]))
        flip = bool(score1[i, j] > score0[i, j])
        results.append({'validation_id': name, 'closest_same_suffix_id': nearby,
                        'numeric_distance': abs(int(nearby[:-1]) - int(name[:-1])),
                        'most_similar_training_id': train[j], 'flipped_orientation': flip,
                        'rgb_corr': float((r1 if flip else r0)[i, j]),
                        'thermal_corr': float((t1 if flip else t0)[i, j]),
                        'joint_score': float(score[i, j])})
    return {'split_sha256': split['split_sha256'], 'train_originals': len(train), 'validation_originals': len(val),
            'test_pixels_read': 0, 'identity_overlap': sorted(set(train) & set(val)),
            'png_metadata_keys': dict(metadata),
            'numeric_distance_threshold_counts': {str(k): sum(r['numeric_distance'] <= k for r in results)
                                                  for k in (1, 2, 5, 10)},
            'joint_correlation_threshold_counts': {str(k): sum(r['joint_score'] >= k for r in results)
                                                   for k in (.90, .95, .98)},
            'methods': 'Nearest numeric ID within D/N suffix. Max across training orientations of min(RGB, thermal) centered grayscale 64x48 Pearson correlations.',
            'limits': 'Numeric adjacency is not a timestamp; correlation is a scene-similarity screen, not capture-session ground truth.',
            'per_validation_image': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    write_json(args.output, result)
    print({k: v for k, v in result.items() if k != 'per_validation_image'})


if __name__ == '__main__':
    main()
