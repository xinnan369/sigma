"""CPU reference for the preregistered local and recurrent fusion mixers."""
import numpy as np

LEVELS = (80, 40, 20)
WIDTH = 64


def budget(width=WIDTH, levels=LEVELS):
    # Per level: shared-shape bias-free 2W->W projection and three W-vectors.
    # Padded taps count, and the standalone state injection multiply is charged
    # as one MAC equivalent in both arms.
    per_channel = 2*width + 3
    return {'parameters': len(levels)*width*per_channel,
            'mac_equivalents': sum(s*s for s in levels)*width*per_channel}


def mix(x, coefficients, arm):
    """Apply one mixer to row-major tokens [N, C]; reset per image and level."""
    x = np.asarray(x, dtype=np.float32)
    p = np.asarray(coefficients, dtype=np.float32)
    if x.ndim != 2 or p.shape != (3, x.shape[1]) or not np.isfinite(x).all() or not np.isfinite(p).all():
        raise ValueError('Invalid tokens or coefficients')
    y = np.zeros_like(x)
    if arm == 'C':
        y[:] = p[0]*x
        y[1:] += p[1]*x[:-1]
        y[2:] += p[2]*x[:-2]
    elif arm == 'D':
        a = np.tanh(p[1])
        h = np.zeros(x.shape[1], dtype=np.float32)
        for t, token in enumerate(x):
            h = a*h + p[2]*token
            y[t] = h + p[0]*token
    else:
        raise ValueError('Unknown fusion arm')
    return y
