"""Hash actual local assets without unpickling weights or altering data."""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_record(path: Path, source_url: str) -> dict:
    if not source_url:
        raise ValueError('Checkpoint provenance URL is required')
    with path.open('rb') as stream:
        prefix = stream.read(128)
    if prefix.startswith(b'version https://git-lfs.github.com/spec/v1'):
        raise ValueError('An LFS pointer is not a checkpoint')
    if path.stat().st_size == 0:
        raise ValueError('Empty checkpoint')
    return {'name': path.name, 'source_url': source_url,
            'bytes': path.stat().st_size, 'sha256': sha256(path),
            'state_dict_validation': 'NOT RUN; hashing does not validate checkpoint compatibility'}


def dataset_record(root: Path, source_url: str, expected_count: int = 393) -> dict:
    if not source_url:
        raise ValueError('Dataset provenance URL is required')
    split = root / 'test.txt'
    names = split.read_text(encoding='utf-8').splitlines()
    if (len(names) != expected_count or len(set(names)) != len(names)
            or any(not name or name.strip() != name or '/' in name or '\\' in name
                   or name in ('.', '..') for name in names)):
        raise ValueError('Official test split must have unique, nonempty, safe sample identifiers and the expected count')
    records = []
    for name in names:
        for folder in ('RGB', 'Modal', 'Label'):
            path = root / folder / (name + '.png')
            records.append({'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size,
                            'sha256': sha256(path)})
    return {'source_url': source_url, 'test_txt_sha256': sha256(split),
            'sample_count': len(names), 'files': records,
            'split_provenance': 'Caller must compare test.txt hash to the documented official source',
            'pixel_validation': 'NOT RUN; hashes alone do not verify image channels or label mapping'}
