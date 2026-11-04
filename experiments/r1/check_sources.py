"""Check public reference pages without downloading datasets or checkpoints."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

SOURCES = {
    'sigma_tiny_mfnet_view': 'https://drive.google.com/file/d/1N9UU9G5K8qxKsZOuEzSLiCGXC5XCaMaU/view',
    'sigma_tiny_mfnet_download': 'https://drive.google.com/uc?export=download&id=1N9UU9G5K8qxKsZOuEzSLiCGXC5XCaMaU',
    'sigma_processed_datasets': 'https://drive.google.com/drive/folders/1GD4LYF208h9-mHJ_lxW11UM0TPlRmv0z',
    'mfnet_official_project': 'https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/',
}


def main():
    records = []
    for name, url in SOURCES.items():
        record = {'name': name, 'url': url, 'asset_sha256': None,
                  'scope': 'HTTP page availability only; not downloaded asset verification'}
        try:
            with urllib.request.urlopen(url, timeout=25) as response:
                record['http_status'] = response.status
                record['content_type'] = response.headers.get('Content-Type')
                response.read(1024)  # Do not persist payloads or equate a folder page with a dataset.
        except urllib.error.HTTPError as exc:
            record['http_status'] = exc.code
        except Exception as exc:
            record['error_type'] = type(exc).__name__
        records.append(record)
    print(json.dumps({'sources': records, 'local_assets': 'NOT INSPECTED by URL-only checker'}, indent=2))
    return 2 if any(r.get('http_status') != 200 for r in records) else 0


if __name__ == '__main__':
    raise SystemExit(main())
