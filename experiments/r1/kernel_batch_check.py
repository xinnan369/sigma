"""Supplement upstream tests: compare entire B=2 tensors, not tuple-unpacked batch 0.

Uses the fixed upstream reference implementation in place; copies no kernel code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import subprocess


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', type=Path, required=True)
    args = parser.parse_args()
    head = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if head != '98ba3c6938d29169bbd27f91519084bb462561df':
        raise ValueError('Pinned upstream mismatch')
    subprocess.run(['git', '-C', str(args.upstream), 'diff', '--quiet', 'HEAD', '--',
                    'models/encoders/selective_scan/test_selective_scan.py'], check=True)
    functions = runpy.run_path(str(args.upstream / 'models/encoders/selective_scan/test_selective_scan.py'))
    results = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for nrows in (1, 2, 3, 4):
            torch.manual_seed(0)
            batch, dim, state, length = 2, 24, 8, 257
            shape = (batch, dim, length)
            tensors = [torch.randn(shape, device='cuda', dtype=dtype),
                       0.5 * torch.rand(shape, device='cuda', dtype=dtype),
                       -0.5 * torch.rand((dim, state), device='cuda'),
                       torch.randn((batch, 2, state, length), device='cuda', dtype=dtype),
                       torch.randn((batch, 2, state, length), device='cuda', dtype=dtype),
                       torch.randn(dim, device='cuda'),
                       0.5 * torch.rand(dim, device='cuda')]
            actual_inputs = [x.detach().clone().requires_grad_() for x in tensors]
            reference_inputs = [x.detach().clone().requires_grad_() for x in tensors]
            # Do not use `out, *rest = ...`: these functions return a BxDxL Tensor.
            out = functions['selective_scan_fn'](*actual_inputs, delta_softplus=True, nrows=nrows)
            ref = functions['selective_scan_ref'](*reference_inputs, delta_softplus=True)
            assert out.shape == (batch, dim, length)
            rtol, atol = ((6e-4, 2e-3) if dtype == torch.float32 else
                          ((3e-2, 5e-2) if dtype == torch.bfloat16 else (3e-3, 5e-3)))
            torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
            gradient = torch.randn_like(out)
            out.backward(gradient)
            ref.backward(gradient)
            for index, (actual, expected) in enumerate(zip(actual_inputs, reference_inputs)):
                grtol, gatol = ((1e-3, 1e-3) if index in (2, 5, 6) else
                               ((rtol * 5, atol * 10) if index == 1 else
                                ((rtol * 2, atol * 2) if index == 0 else (rtol, atol))))
                torch.testing.assert_close(actual.grad, expected.grad, rtol=grtol, atol=gatol)
            assert reference_inputs[0].grad[1].abs().sum().item() > 0
            results.append({'dtype': str(dtype), 'nrows': nrows, 'batch': batch, 'length': length,
                            'forward_max_abs_diff': (out - ref).abs().max().item(),
                            'all_batch_forward_and_all_input_gradients': 'PASS'})
    torch.cuda.synchronize()
    print(json.dumps({'upstream_head': head, 'seed': 0, 'cases': results,
                      'scope': 'Synthetic full-batch kernel compatibility only; not Sigma/MFNet evaluation'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
