"""Persist local Epoch 5 predictions and public summary records."""
import numpy as np

from experiments.r1g.train import write_rows
from experiments.r2i.evaluate import evaluate


def evaluate_cell(model, root, config, cfg, split, graph, r2a, folder):
    rows, predictions = evaluate(model, root, config, cfg, split, graph, r2a, None)
    write_rows(folder / 'validation.jsonl', rows)
    np.savez_compressed(folder / 'predictions.npz', **{f'{i}/{c}': p for (i, c), p in predictions.items()})
