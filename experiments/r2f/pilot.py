"""Validate the readout repair, then replay the original R2e pilot only."""
import argparse
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.r1.assets import sha256
from experiments.r1c.data import write_json
from experiments.r1g.partition import ROOT, read, require
from experiments.r2b.state import Scan
from experiments.r2e import run as r2e
from experiments.r2f.replay import validated_factor, verify_validated


def synthetic(native):
    gen = torch.Generator(device='cuda').manual_seed(3411)

    def rand(*shape):
        return torch.randn(shape, device='cuda', generator=gen)

    outputs = []
    for length, channels, groups, n, rows, scale in (
        (33, 16, 1, 4, 1, .01), (300, 32, 2, 4, 2, 1.),
        (1200, 32, 4, 16, 4, .5), (2067, 32, 4, 16, 2, 2.)):
        a, d, bias = -rand(channels, n).exp(), rand(channels), rand(channels) - 2.
        scans = [Scan(rand(1, channels, length) * scale, rand(1, channels, length), a,
                      rand(1, groups, n, length), rand(1, groups, n, length), d, bias, True, rows)
                 for _ in range(2)]
        native_y = [native(s.u, s.delta, s.a, s.b, s.c, s.d, s.bias, s.softplus, s.nrows)[0] for s in scans]
        outputs.append({'length': length, 'channels': channels, 'groups': groups, 'state': n, 'nrows': rows,
                        'scale': scale, **validated_factor(native, *scans, *native_y).audit})
    return outputs


def main():
    parser = argparse.ArgumentParser()
    for name in ('upstream', 'init', 'root', 'evidence', 'checkpoint-dir'):
        parser.add_argument('--' + name, required=True, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    import selective_scan_cuda_core as extension

    fixed = read(args.evidence / 'fixed-call.json')
    require(fixed['triton_vs_cuda_readout']['max_mixed_ratio'] > 1
            and fixed['cuda_states_fused_readout_vs_cuda']['max_mixed_ratio'] <= 1,
            'Fixed failing call and candidate state repair are required')
    with torch.inference_mode():
        write_json(args.evidence / 'synthetic-replay.json', synthetic(extension.fwd))
    args.phase = 'pilot'
    with patch.object(r2e, 'factor', validated_factor), patch.object(r2e, 'verify_reference', verify_validated):
        r2e.run(args)
    # Deliberately use a distinct certificate phase so R2e cannot authorize
    # its original full runner with this implementation's pilot evidence.
    certificate = args.evidence / 'pilot/certificate.json'
    result = read(certificate)
    result['phase'] = 'r2f-pilot'
    result['fingerprints']['r2f_implementation_sha256'] = {
        name: sha256(ROOT / 'experiments/r2f' / name) for name in ('numerics.py', 'replay.py', 'pilot.py')}
    write_json(certificate, result)


if __name__ == '__main__':
    main()
