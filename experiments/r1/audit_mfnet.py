"""Audit every fixed-test triplet and publish hashes/format metadata, not pixels."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from experiments.r1.assets import sha256
from experiments.r1.fetch_mfnet import SPLIT_SHA


def audit(root: Path):
    import cv2
    import numpy as np
    from PIL import Image
    split = root / 'test.txt'
    if sha256(split) != SPLIT_SHA:
        raise ValueError('Fixed test.txt hash mismatch')
    names = split.read_text(encoding='utf-8').splitlines()
    if len(names) != 393 or len(set(names)) != 393:
        raise ValueError('Test split does not have 393 unique samples')
    records, errors = [], []
    histogram = np.zeros(256, dtype=np.int64)
    modes = Counter()
    for sample in names:
        for folder in ('RGB', 'Modal', 'Label'):
            relative = f'{folder}/{sample}.png'
            path = root / relative
            if not path.is_file():
                errors.append({'path': relative, 'error': 'missing'})
                continue
            try:
                with Image.open(path) as image:
                    image.load()
                    mode, size, fmt = image.mode, image.size, image.format
                    label_raw = np.asarray(image) if folder == 'Label' else None
                raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if fmt != 'PNG' or size != (640, 480) or raw is None or raw.dtype != np.uint8:
                    raise ValueError('Expected 640x480 uint8 PNG')
                if folder == 'RGB' and raw.shape != (480, 640, 3):
                    raise ValueError('RGB must contain three source channels')
                entry = {'path': relative, 'sha256': sha256(path), 'bytes': path.stat().st_size,
                         'pil_mode': mode, 'raw_shape': list(raw.shape), 'dtype': str(raw.dtype)}
                if folder == 'Label':
                    label = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    values = np.unique(label)
                    if not set(values.tolist()) <= set(range(9)) | {255}:
                        raise ValueError('Label contains values outside 0..8 or ignore=255')
                    if label_raw.shape != label.shape or not np.array_equal(label_raw, label):
                        raise ValueError('PIL labels disagree with official cv2 grayscale loader; no remapping applied')
                    histogram += np.bincount(label.ravel(), minlength=256)
                    entry['label_values'] = values.tolist()
                elif folder == 'Modal':
                    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if gray.shape != (480, 640) or gray.dtype != np.uint8:
                        raise ValueError('Thermal does not match the official grayscale input contract')
                    entry['official_loader_shape_after_replication'] = [480, 640, 3]
                modes[(folder, mode)] += 1
                records.append(entry)
            except Exception as exc:
                errors.append({'path': relative, 'error': str(exc)})
    canonical = json.dumps(records, sort_keys=True, separators=(',', ':')).encode()
    return {'scope': 'Real author-provided MFNet test assets; no preprocessing or label remapping applied',
            'test_txt_sha256': SPLIT_SHA, 'sample_count': len(names), 'expected_files': 1179,
            'valid_file_records': len(records), 'errors': errors, 'cv2_version': cv2.__version__,
            'mode_counts': {f'{folder}/{mode}': count for (folder, mode), count in sorted(modes.items())},
            'class_pixel_counts': {str(i): int(histogram[i]) for i in range(9)},
            'ignore_255_pixel_count': int(histogram[255]),
            'manifest_digest_basis': 'SHA256 of UTF-8 json.dumps(file_records, sort_keys=True, separators=(comma,colon))',
            'file_records_sha256': hashlib.sha256(canonical).hexdigest(), 'files': records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2), encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))
    return int(bool(result['errors']) or result['valid_file_records'] != 1179)


if __name__ == '__main__':
    raise SystemExit(main())
