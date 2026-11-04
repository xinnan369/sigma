"""Fail closed on missing reference assets; never fabricate their hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from experiments.r1.assets import checkpoint_record, dataset_record

UPSTREAM_SHA = '98ba3c6938d29169bbd27f91519084bb462561df'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--dataset-source')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-source')
    args = parser.parse_args()
    result = {'evaluation': 'NOT RUN', 'blockers': []}
    try:
        head = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
        if head != UPSTREAM_SHA:
            raise ValueError('Upstream HEAD differs from pinned SHA')
        subprocess.run(['git', '-C', str(args.upstream), 'diff', '--quiet', 'HEAD', '--'], check=True)
        result['upstream_head'] = head
        for key, name in [('official_eval_sha256', 'eval.py'),
                          ('mfnet_config_sha256', 'configs/config_MFNet.py')]:
            blob = subprocess.check_output(['git', '-C', str(args.upstream), 'show', 'HEAD:' + name])
            result[key] = hashlib.sha256(blob).hexdigest()
        result['source_hash_basis'] = 'Git blobs (independent of checkout CRLF conversion)'
    except Exception as exc:
        result['blockers'].append('Upstream: ' + type(exc).__name__)
    for label, path, source, record_fn in [
        ('dataset', args.dataset, args.dataset_source, dataset_record),
        ('checkpoint', args.checkpoint, args.checkpoint_source, checkpoint_record),
    ]:
        if path is None:
            result[label] = {'status': 'NOT PROVIDED', 'sha256': None}
            result['blockers'].append(label + ' missing')
            continue
        try:
            result[label] = record_fn(path, source)
        except Exception as exc:
            result[label] = {'status': 'INVALID', 'error_type': type(exc).__name__}
            result['blockers'].append(label + ' invalid')
    result['scope'] = 'Provenance/hash preflight only; CUDA, state_dict, pixels and split provenance still require validation'
    print(json.dumps(result, indent=2))
    return 2 if result['blockers'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
