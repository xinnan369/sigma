import hashlib

import numpy as np
import torch


def rng(seed, epoch, identity):
    raw = hashlib.sha256(f'{seed}/{epoch}/{identity}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(raw[:16], 'big'))


def position(gen, height, width, size):
    return int(gen.integers(height - size + 1)), int(gen.integers(width - size + 1))


def train_occlusion(thermal, mean, std, seed, name, probability, size):
    gen = rng(seed, 1, name)
    if gen.random() >= probability:
        return thermal, None
    y, x = position(gen, *thermal.shape[-2:], size)
    output = thermal.clone()
    fill = torch.as_tensor(-np.asarray(mean) / np.asarray(std), device=thermal.device, dtype=thermal.dtype)
    output[..., y:y+size, x:x+size] = fill[:, None, None]
    return output, [y, x, size]
