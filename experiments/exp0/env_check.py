"""R0 environment inventory; safe without CUDA or external data."""
from __future__ import annotations
import platform
import sys
import importlib.util
import importlib.metadata
import subprocess
import shutil


def main():
    print('python:', sys.version.split()[0]); print('system:', platform.platform())
    for package in ['numpy','pytest','torch','cv2','timm','einops', 'selective_scan_cuda_core']:
        print(f'{package}:', 'installed' if importlib.util.find_spec(package) else 'missing')
    for distribution in ['numpy', 'pytest', 'torch', 'opencv-python', 'timm', 'einops']:
        try:
            print(f'{distribution}_distribution_version:', importlib.metadata.version(distribution))
        except importlib.metadata.PackageNotFoundError:
            print(f'{distribution}_distribution_version:', 'missing')
    torch_ok = True
    try:
        import torch
        print('torch_version:', torch.__version__)
        print('torch_cuda_build:', torch.version.cuda)
        print('cudnn_version:', torch.backends.cudnn.version())
        print('cuda_available:', torch.cuda.is_available())
        if torch.cuda.is_available():
            print('cuda_device:', torch.cuda.get_device_name(0))
            print('cuda_total_memory_bytes:', torch.cuda.get_device_properties(0).total_memory)
    except Exception as exc:
        torch_ok = False
        print('torch_load_error:', type(exc).__name__, str(exc)[:160])
    try:
        r = subprocess.run(['git','--version'], text=True, capture_output=True, timeout=10)
        print('git:', r.stdout.strip() or r.stderr.strip(), 'exit:', r.returncode)
    except Exception as exc:
        print('git_check_error:', type(exc).__name__, str(exc)[:160])
    for command in [
        ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
        ['nvcc', '--version'],
    ]:
        if not shutil.which(command[0]):
            print(command[0] + ': missing')
            continue
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=10)
            print(command[0] + ':', (result.stdout or result.stderr).strip(), 'exit:', result.returncode)
        except Exception as exc:
            print(command[0] + '_check_error:', type(exc).__name__)
    print('scope: inventory only; does not validate Sigma/CUDA kernels or MFNet readiness')
    return 0 if torch_ok else 1

if __name__=='__main__':
    raise SystemExit(main())
