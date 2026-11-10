"""Compare explicit h[t] with the native CUDA output and chunk-end states."""
import json

import torch

from experiments.r2b.state import Scan, check_close, coordinates, literal, outputs
from experiments.r2b.triton_state import trajectory


def compare(scan, h, native_output, prefix):
    length = scan.u.shape[-1]
    ends = [min(i + 2048, length) - 1 for i in range(0, length, 2048)]
    return {
        'output': check_close(outputs(scan, h), native_output, 'CUDA output'),
        'chunk_end_h': check_close(h[..., ends].transpose(-1, -2), prefix[..., 1::2], 'CUDA chunk-end h'),
        'chunk_end_indices': ends,
    }


def native_call(extension, scan):
    return extension.fwd(scan.u, scan.delta, scan.a, scan.b, scan.c, scan.d,
                         scan.bias, scan.softplus, scan.nrows)


def four_orders(height, width):
    return [[row * width + col for _, row, col in
             (coordinates('four', k, t, height, width) for t in range(height * width))] for k in range(4)]


@torch.inference_mode()
def synthetic(extension, *, device='cuda'):
    gen = torch.Generator(device=device).manual_seed(3407)

    def normal(*shape):
        return torch.randn(shape, generator=gen, device=device)

    rows = []
    for height, width, batch, nrows in [(1, 1, 1, 1), (3, 11, 2, 1), (15, 20, 3, 2),
                                         (7, 147, 2, 4), (13, 159, 2, 4), (30, 40, 1, 1)]:
        length = height * width
        order = torch.tensor(four_orders(height, width), device=device)

        def reorder(x):
            return torch.stack([x[:, k, :, order[k]] for k in range(4)], dim=1).contiguous()

        u = reorder(normal(batch, 4, 8, length)).reshape(batch, 32, length)
        delta = reorder(normal(batch, 4, 8, length)).reshape_as(u)
        b, c = (reorder(normal(batch, 4, 4, length)) for _ in range(2))
        scan = Scan(u, delta, -normal(32, 4).exp(), b, c, normal(32), normal(32) - 3, True, nrows)
        y, prefix = native_call(extension, scan)
        h = trajectory(scan)
        results = compare(scan, h, y, prefix)
        if length <= 33:
            results['sequential_all_steps'] = check_close(h, literal(scan), 'literal h')
        per_direction = [check_close(x, z, f'direction {k}') for k, (x, z) in
                         enumerate(zip(outputs(scan, h).split(8, dim=1), y.split(8, dim=1), strict=True))]
        rows.append({'grid': [height, width], 'batch': batch, 'length': length, 'nrows': nrows,
                     'per_direction_output': per_direction, **results})
    return rows


def main():
    import selective_scan_cuda_core as extension
    import triton

    torch.backends.cuda.matmul.allow_tf32 = False
    print(json.dumps({'status': 'PASS', 'torch': torch.__version__, 'triton': triton.__version__,
                      'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
                      'cases': synthetic(extension)}, indent=2))


if __name__ == '__main__':
    main()
