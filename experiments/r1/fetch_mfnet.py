"""Download only the unchanged author test split from public author folders.

Payloads must be outside the bridge repository. Public HTML directory indexes
are used without authentication, API keys or permission bypasses.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import html
import io
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

FOLDERS = {'RGB': '11UaWNMGksgnLzfJZAJ8gnq-bFY54HPdG',
           'Modal': '1k2ZXJFhPeIBAfHa_EsOVaraq5L9mJW0W',
           'Label': '1HCyScTV71IQ2KQODypKrKpJKIujOkref'}
SPLIT_SHA = 'a236aa6ac7e55ec6b5e419b55c6934439611453939eee2c0bf7b0f0c34917c9d'


def validate_png(data):
    from PIL import Image
    if not data.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('Not a PNG payload')
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        image.load()


def public_index(folder_id):
    url = 'https://drive.google.com/embeddedfolderview?id=' + folder_id
    with urllib.request.urlopen(url, timeout=45) as response:
        text = response.read().decode('utf-8')
    pairs = re.findall(r'id="entry-([A-Za-z0-9_-]+)".*?class="flip-entry-title">([^<]+)</div>', text, re.S)
    index = {}
    for file_id, name in pairs:
        name = html.unescape(name)
        if name in index:
            raise ValueError('Ambiguous duplicate filename in public source folder')
        index[name] = file_id
    if not index:
        raise ValueError('Public directory listing unavailable or format changed')
    return index


def fetch_one(root, folder, name, file_id):
    path = root / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://drive.usercontent.google.com/download?id={file_id}&export=download'
    if path.exists():
        validate_png(path.read_bytes())
        return {'path': f'{folder}/{name}', 'source_file_id': file_id, 'status': 'already_present'}
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read(5 * 1024 * 1024)
            if not data.startswith(b'\x89PNG\r\n\x1a\n'):
                raise ValueError('Download is not PNG (possibly login/quota HTML); no file written')
            validate_png(data)
            path.write_bytes(data)
            return {'path': f'{folder}/{name}', 'source_file_id': file_id, 'status': 'downloaded'}
        except urllib.error.HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--index-output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    bridge = Path(__file__).resolve().parents[2]
    if root.is_relative_to(bridge) or not 1 <= args.workers <= 16:
        parser.error('Use an external asset root and 1..16 workers')
    raw = (root / 'test.txt').read_bytes()
    if hashlib.sha256(raw).hexdigest() != SPLIT_SHA:
        raise ValueError('Author test.txt bytes differ from the fixed reference')
    names = raw.decode('utf-8').splitlines()
    if len(names) != 393 or len(set(names)) != 393:
        raise ValueError('Expected the original 393 unique test IDs')
    jobs = []
    for folder, fid in FOLDERS.items():
        index = public_index(fid)
        print(f'{folder}: {len(index)} source entries; matching fixed test IDs', flush=True)
        for sample in names:
            filename = sample + '.png'
            if filename not in index:
                raise FileNotFoundError(folder + '/' + filename + ' absent in author listing')
            jobs.append((folder, filename, index[filename]))
    records, failures = [], []
    start = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as executor:
        futures = {executor.submit(fetch_one, root, *job): job for job in jobs}
        for count, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                failures.append({'path': f'{job[0]}/{job[1]}', 'error_type': type(exc).__name__,
                                 'http_status': getattr(exc, 'code', None)})
            if count % 100 == 0 or count == len(jobs):
                print(f'{count}/{len(jobs)} checked; failures={len(failures)}', flush=True)
    args.index_output.parent.mkdir(parents=True, exist_ok=True)
    args.index_output.write_text(json.dumps({'test_txt_sha256': SPLIT_SHA, 'folders': FOLDERS,
        'records': sorted(records, key=lambda r: r['path']), 'failures': failures,
        'elapsed_seconds': time.perf_counter() - start}, indent=2), encoding='utf-8', newline='\n')
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
