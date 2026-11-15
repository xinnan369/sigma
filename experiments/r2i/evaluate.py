"""Compare predictions with the same-training-track natural baseline."""
import cv2
import numpy as np
import torch

from experiments.r1c.pilot import read_sample
from experiments.r1g.partition import require
from experiments.r2a.metrics import confusion, digest, distance_bands, pair, scores
from experiments.r2a.run import normalized_thermal, perturbed
from experiments.r2i.augment import position, rng


def evaluate(model, root, config, protocol, split, graph, r2a, baseline):
    assignment = {i: g[0] for g in graph['groups'] for i in g}
    results, predictions = [], {}
    model.eval()
    with torch.inference_mode():
        for sample in split['validation_ids']:
            rgb, clean, label = read_sample(root, sample, config, False)
            raw = cv2.imread(str(root / 'Modal' / (sample + '.png')), cv2.IMREAD_GRAYSCALE)
            require(torch.equal(clean, normalized_thermal(raw, config)), 'Thermal preprocessor differs')
            y, x = position(rng(protocol['random_occlusion_seed'], 0, sample), *raw.shape, 64)
            cases = {'primary': r2a['primary'],
                     'random_occlusion': {'kind': 'occlusion', 'position': 'random', 'fill': 0},
                     'gaussian': {'kind': 'gaussian', 'position': 'center', 'sigma': 20, 'seed': 3407}}
            roi_cfg = {**r2a, 'positions_y0_x0_y1_x1': {**r2a['positions_y0_x0_y1_x1'],
                         'random': [y, x, y+64, x+64]}}
            gt = label[0].cpu().numpy()
            for case in protocol['validation']:
                roi = np.zeros(raw.shape, dtype=bool)
                thermal = clean
                if case != 'clean':
                    changed, roi, _ = perturbed(raw, cases[case], sample, roi_cfg)
                    thermal = normalized_thermal(changed, config)
                logits = model(rgb, thermal)
                require(torch.isfinite(logits).all().item(), 'Nonfinite evaluation')
                pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
                key = sample, case
                matrix = confusion(gt, pred).tolist()
                row = {'id': sample, 'group': assignment[sample], 'case': case, 'matrix': matrix,
                       'prediction_sha256': digest(pred), 'roi_sha256': digest(roi)}
                if baseline is None:
                    predictions[key] = pred
                else:
                    ref = baseline[key]
                    bins = distance_bands(roi, r2a['distance_upper_bounds_px'])
                    row.update(pair(ref, pred, gt, roi, bins, scores(confusion(gt, ref))))
                results.append(row)
    return results, predictions
