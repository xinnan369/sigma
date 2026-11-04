"""Import dependencies and actually execute a deterministic CUDA operation."""
from __future__ import annotations

import importlib
import json
import platform
import sys
import time


def main():
    result = {'python': platform.python_version(), 'platform': platform.system(),
              'seed': 3407, 'scope': 'CUDA/import smoke only, not model evaluation'}
    failures = []
    modules = {}
    for name in ('torch', 'torchvision', 'numpy', 'timm', 'einops', 'cv2',
                 'selective_scan_cuda_core', 'selective_scan'):
        try:
            modules[name] = importlib.import_module(name)
            result[name] = getattr(modules[name], '__version__', 'import OK')
        except Exception as exc:
            result[name] = type(exc).__name__ + ': ' + str(exc)
            failures.append(name)
    torch = modules.get('torch')
    if torch is not None:
        result['torch_cuda_build'] = torch.version.cuda
        result['cuda_available'] = torch.cuda.is_available()
        if not torch.cuda.is_available():
            failures.append('CUDA unavailable')
        else:
            torch.manual_seed(3407)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            x = torch.arange(256, device='cuda', dtype=torch.float32).reshape(16, 16)
            y = x @ x.T
            torch.cuda.synchronize()
            torch.testing.assert_close(y.cpu(), x.cpu() @ x.cpu().T, rtol=0, atol=0)
            result.update(device=torch.cuda.get_device_name(0),
                          compute_capability=torch.cuda.get_device_capability(0),
                          total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                          smoke_elapsed_seconds=time.perf_counter() - start,
                          smoke_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          cuda_matmul='PASS')
    result['failures'] = failures
    print(json.dumps(result, indent=2))
    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())
