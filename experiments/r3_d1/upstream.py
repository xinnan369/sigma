"""External, pinned Sigma fusion; no local recurrence or fallback."""
import hashlib
from importlib import import_module, metadata, util
from pathlib import Path
import subprocess

PIN = '98ba3c6938d29169bbd27f91519084bb462561df'
SOURCE = 'models/encoders/vmamba.py'
SOURCES = (SOURCE, 'models/encoders/dual_vmamba.py',
           'models/encoders/selective_scan/selective_scan/selective_scan_interface.py')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_sigma(root):
    root = Path(root).resolve()
    def git(*args):
        return subprocess.check_output(['git','-C',str(root),*args])
    if git('rev-parse','HEAD').decode().strip()!=PIN:
        raise RuntimeError('Sigma commit mismatch')
    if git('status','--porcelain','--untracked-files=no','--','models/encoders'):
        raise RuntimeError('Sigma encoder tree has local changes')
    hashes = {}
    for rel in SOURCES:
        data=(root/rel).read_bytes()
        if data.replace(b'\r\n',b'\n')!=git('show',f'{PIN}:{rel}'):
            raise RuntimeError(f'Sigma source bytes mismatch: {rel}')
        hashes[rel]=sha(root/rel)
    core=import_module('selective_scan_cuda_core')
    api=import_module('selective_scan.selective_scan_interface')
    if (api.selective_scan_cuda is not core or
            Path(api.__file__).read_bytes().replace(b'\r\n',b'\n')!=git('show',f'{PIN}:{SOURCES[2]}')):
        raise RuntimeError('CrossMamba selective_scan wrapper differs from pinned Sigma')
    spec=util.spec_from_file_location('_r3d1_pinned_sigma_vmamba',root/SOURCE)
    vm=util.module_from_spec(spec)
    spec.loader.exec_module(vm)
    if vm.selective_scan_cuda is not core or vm.selective_scan_fn_v1 is not api.selective_scan_fn:
        raise RuntimeError('Unexpected upstream scan implementation')
    return vm, core, dict(commit=PIN, source_sha256=hashes,
                          source_file=str(root/SOURCE), extension_file=core.__file__,
                          extension_sha256=sha(core.__file__), wrapper_file=api.__file__,
                          wrapper_sha256=sha(api.__file__),
                          extension_version=metadata.version('selective_scan'))
