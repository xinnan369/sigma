"""Pin, retrieve and audit both author training lists without using test pixels."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import urllib.request

from experiments.r1.assets import sha256
from experiments.r1.fetch_mfnet import FOLDERS, SPLIT_SHA, public_index, validate_png

SPLITS = {'test.txt': (SPLIT_SHA, 393),
          'train.txt': ('e14ef52501a5d0000361e79acb42734f391427ac5a4164ad7864b0d2718269f7', 1568),
          'train2.txt': ('503cdbc99aaab424db5b1f33c5be1e163e46ec9df90256919ad05f1d3f6530c1', 1176)}
SOURCE = 'https://drive.google.com/drive/folders/1bD5b4xXkDBasOLhupUBDdw2rq9oogbu_'
_sessions = threading.local()


def fetch_training_file(root, folder, name, file_id):
    """Reuse connections per worker; only publish fully decoded PNG payloads."""
    import requests
    path = root / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        validate_png(path.read_bytes())
        return
    if not hasattr(_sessions, 'session'):
        _sessions.session = requests.Session()
        _sessions.session.proxies.update(urllib.request.getproxies())
    url = f'https://drive.usercontent.google.com/download?id={file_id}&export=download'
    for attempt in range(3):
        try:
            with _sessions.session.get(url, timeout=(15, 45), stream=True) as response:
                response.raise_for_status()
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > 5 * 1024 * 1024:
                        raise ValueError('Unexpected oversized PNG')
                    chunks.append(chunk)
            payload = b''.join(chunks)
            validate_png(payload)
            temporary = path.with_suffix('.png.part')
            temporary.write_bytes(payload)
            temporary.replace(path)
            return
        except (requests.RequestException, ValueError, OSError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def base_id(name):
    return name.removesuffix('_flip')


def read_lists(root):
    lists = {}
    for name, (digest, count) in SPLITS.items():
        if sha256(root / name) != digest:
            raise ValueError(f'Pinned author list changed: {name}')
        ids = (root / name).read_text().splitlines()
        if len(ids) != count or len(set(ids)) != count or any(not re.fullmatch(r'[0-9]+[DN](?:_flip)?', x) for x in ids):
            raise ValueError(f'Invalid list: {name}')
        lists[name] = ids
    test = {base_id(x) for x in lists['test.txt']}
    for name in ('train.txt', 'train2.txt'):
        if test & {base_id(x) for x in lists[name]}:
            raise ValueError('Training/test source identity overlap')
    return lists


def internal_split(lists, seed=3407):
    # Ten percent per day/night stratum, rounded up. Original and flip stay together.
    originals = sorted({base_id(x) for x in lists['train.txt']})
    validation = []
    for condition in ('D', 'N'):
        group = [x for x in originals if x.endswith(condition)]
        ranked = sorted(group, key=lambda x: hashlib.sha256(f'{seed}:{x}'.encode()).hexdigest())
        validation.extend(ranked[:(len(group) + 9) // 10])
    validation = sorted(validation)
    train = sorted(x for x in lists['train.txt'] if base_id(x) not in validation)
    if ({base_id(x) for x in train} & set(validation) or
            ({base_id(x) for x in train} | set(validation)) & set(lists['test.txt'])):
        raise ValueError('Split leakage')
    result = {'seed': seed, 'regime': 'independent train.txt; hold out ceil(10%) by day/night source identity',
              'ranking': 'ascending SHA256 UTF-8 seed:base_id within D/N; no label/statistic selection',
              'train_ids': train, 'validation_ids': validation,
              'official_list_sha256': {k: v[0] for k, v in SPLITS.items()},
              'test_pixels_used': False}
    result['split_sha256'] = canonical_sha(result)
    return result


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def audit_files(root, names, sources):
    import cv2
    import numpy as np
    from PIL import Image
    records, errors = [], []
    counts = np.zeros(256, dtype=np.int64)
    for sample in names:
        for folder in FOLDERS:
            relative = f'{folder}/{sample}.png'
            path = root / relative
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im.load()
                    mode, fmt = im.mode, im.format
                    arr = np.asarray(im)
                expected = (480, 640, 3) if folder == 'RGB' else (480, 640)
                if fmt != 'PNG' or arr.shape != expected or arr.dtype != np.uint8:
                    raise ValueError('Expected original uint8 640x480 PNG with fixed channels')
                record = {'path': relative, 'sha256': sha256(path), 'bytes': path.stat().st_size,
                          'shape': list(arr.shape), 'mode': mode, 'source_file_id': sources.get(relative)}
                if folder == 'Label':
                    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if not np.array_equal(arr, gray) or not set(np.unique(arr)) <= set(range(9)) | {255}:
                        raise ValueError('Label mapping must remain raw 0..8/255')
                    record['label_values'] = np.unique(arr).tolist()
                    counts += np.bincount(arr.ravel(), minlength=256)
                records.append(record)
            except Exception as exc:
                errors.append({'path': relative, 'error': str(exc)})
    return {'files': records, 'files_sha256': canonical_sha(records), 'errors': errors,
            'sample_count': len(names), 'file_count': len(records), 'class_pixels': counts[:9].tolist(),
            'ignore_pixels': int(counts[255]), 'cv2_version': cv2.__version__}


def pair_hash_overlap(left_records, left_ids, right_records, right_ids):
    def pairs(records, ids):
        hashes = {r['path']: r['sha256'] for r in records}
        result = {}
        for sample in ids:
            pair = tuple(hashes[f'{folder}/{sample}.png'] for folder in ('RGB', 'Modal'))
            result.setdefault(pair, []).append(sample)
        return result
    left, right = pairs(left_records, left_ids), pairs(right_records, right_ids)
    return [{'left_ids': left[k], 'right_ids': right[k]} for k in sorted(left.keys() & right.keys())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--download-only', action='store_true', help='Decode/hash audit can run separately in WSL')
    parser.add_argument('--reuse-index', action='store_true', help='Reuse this run\'s saved public file IDs')
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    root = args.root.resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[2]) or not 1 <= args.workers <= 16:
        parser.error('External data root and 1..16 workers required')
    lists = read_lists(root)
    split = internal_split(lists)
    write_json(args.output / 'split.json', split)
    names = sorted(set(lists['train.txt']) | set(lists['train2.txt']))
    metadata = {'source': SOURCE, 'folders': FOLDERS, 'lists': {
        k: {'sha256': SPLITS[k][0], 'count': len(v), 'base_count': len({base_id(x) for x in v}), 'ids': v}
        for k, v in lists.items()}, 'train_union_count': len(names), 'canonical_test_overlap': 0,
        'train_train2_base_intersection': len({base_id(x) for x in lists['train.txt']} & set(lists['train2.txt'])),
        'internal_train_count': len(split['train_ids']), 'internal_val_count': len(split['validation_ids'])}
    write_json(args.output / 'lists.json', metadata)
    sources = {}
    if args.download:
        jobs = []
        if args.reuse_index:
            sources = json.loads((args.output / 'source-index.json').read_text())
        for folder, fid in FOLDERS.items():
            index = ({k.split('/')[1]: v for k, v in sources.items() if k.startswith(folder + '/')}
                     if args.reuse_index else public_index(fid))
            print(f'{folder}: {len(index)} public entries', flush=True)
            for sample in names:
                name = sample + '.png'
                if name not in index:
                    raise FileNotFoundError(f'Official folder lacks {folder}/{name}')
                sources[f'{folder}/{name}'] = index[name]
                jobs.append((folder, name, index[name]))
        write_json(args.output / 'source-index.json', sources)
        failures = []
        start = time.perf_counter()
        with ThreadPoolExecutor(args.workers) as pool:
            futures = {pool.submit(fetch_training_file, root, *job): job for job in jobs}
            for count, future in enumerate(as_completed(futures), 1):
                try:
                    future.result()
                except Exception as exc:
                    failures.append({'file': '/'.join(futures[future][:2]), 'error': str(exc)})
                if count % 100 == 0 or count == len(jobs):
                    print(f'{count}/{len(jobs)} files; failures={len(failures)}; seconds={time.perf_counter()-start:.1f}', flush=True)
                    write_json(args.output / 'download-progress.json', {'checked': count, 'total': len(jobs),
                               'failure_count': len(failures), 'seconds': time.perf_counter()-start})
        write_json(args.output / 'download.json', {'failures': failures, 'seconds': time.perf_counter()-start, 'jobs': len(jobs)})
        if failures:
            return 1
        if args.download_only:
            return 0
    elif (args.output / 'source-index.json').exists():
        sources = json.loads((args.output / 'source-index.json').read_text())
    result = audit_files(root, names, sources)
    if not result['errors']:
        # Compare prior audited test hashes, without reopening any final-test pixels.
        test_manifest_path = Path(__file__).resolve().parents[2] / 'reports/evidence/r1b/data-manifest.json'
        test_manifest = json.loads(test_manifest_path.read_text())
        if test_manifest['test_txt_sha256'] != SPLIT_SHA or canonical_sha(test_manifest['files']) != test_manifest['file_records_sha256']:
            raise ValueError('Prior final-test hash manifest changed')
        overlap = {'training_test_exact_rgb_thermal_pair_duplicates': pair_hash_overlap(
            result['files'], names, test_manifest['files'], lists['test.txt']),
            'internal_train_val_exact_rgb_thermal_pair_duplicates': pair_hash_overlap(
                result['files'], split['train_ids'], result['files'], split['validation_ids']),
            'scope': 'Exact file SHA256 pairs only; not a guarantee of scene/temporal independence',
            'test_pixels_reopened': False}
        write_json(args.output / 'leakage-audit.json', overlap)
        if overlap['training_test_exact_rgb_thermal_pair_duplicates'] or overlap['internal_train_val_exact_rgb_thermal_pair_duplicates']:
            result['errors'].append({'error': 'Cross-split exact paired-image duplicates'})
    write_json(args.output / 'manifest.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2), flush=True)
    return int(bool(result['errors']) or result['file_count'] != 3 * len(names))


if __name__ == '__main__':
    raise SystemExit(main())
