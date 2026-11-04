"""Fingerprint installed scan artifacts without publishing binaries or local paths."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

from experiments.r1.assets import sha256


def main():
    import torch  # Load libtorch dependencies before importing the compiled extension.
    result = {'scope': 'Installed extension/interface provenance, not model checkpoint hashes'}
    result['torch_version'] = torch.__version__
    for name in ('selective_scan_cuda_core', 'selective_scan'):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError('Scan artifact did not load from the new R1 overlay environment')
        result[name] = {'filename': path.name, 'sha256': sha256(path),
                        'loaded_from_new_r1_overlay': True}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
