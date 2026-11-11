"""Prediction-level R2a perturbations on all three audited Epoch 5 checkpoints."""
import argparse
import contextlib
from pathlib import Path
import time

import numpy as np
import torch

from experiments.r1.assets import sha256
from experiments.r1c.data import write_json
from experiments.r1c.pilot import construct_model, read_sample
from experiments.r1g.partition import ROOT, read, require
from experiments.r2a.metrics import confusion, digest, distance_bands, mask_record, pair, scores
from experiments.r2a.run import append, cases, masks, model_digest, normalized_thermal, perturbed, predict, records
from experiments.r2b.controls import fusion, set_seed
from experiments.r2c.protocol import ARMS, gpu_inputs


def evaluate_arm(args, family, cfg, r2a, split, assignment, signature, audited, all_masks, bands, specs):
    import cv2

    training = args.evidence / 'training' / family
    last = args.checkpoint_dir / family / 'last.pt'
    ledger = read(training / 'checkpoint-ledger.json')
    final = ledger[-1]
    require(len(ledger) == final['epoch'] == cfg['primary_epoch'] and final['updates'] == 785
            and sha256(last) == final['last_sha256'] == audited['last_sha256'], 'Epoch 5 checkpoint differs')
    state = torch.load(last, map_location='cpu', weights_only=False)
    require(state['format'] == cfg['version'] and state['family'] == family and state['signature'] == signature
            and state['initialization'] == read(training / 'initialization.json')
            and state['updates'] == 785 and state['sampler']['epoch'] == 5 and state['sampler']['cursor'] == 1254
            and state['history'] == read(training / 'epochs.json'), 'Checkpoint metadata differs')
    archive = records(training / 'images-epoch5.jsonl')
    require([r['id'] for r in archive] == split['validation_ids'], 'Training clean archive order changed')
    output = args.evidence / 'perturbation' / family
    output.mkdir(parents=True, exist_ok=False)
    set_seed(cfg['seed'])
    start = time.perf_counter()
    with construct_model(args.upstream, args.init) as (model, upstream_cfg, _), fusion(model, family):
        keys = model.load_state_dict(state['model'], strict=True)
        del state
        model = model.cuda().eval()
        require(model_digest(model) == final['model_sha256'], 'Loaded epoch weights differ')
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), contextlib.ExitStack() as stack:
            clean_log = stack.enter_context((output / 'clean.jsonl').open('x', encoding='utf-8', newline='\n'))
            pairs = {s['id']: stack.enter_context((output / f'pairs-{s["id"]}.jsonl')
                                                 .open('x', encoding='utf-8', newline='\n')) for s in specs}
            for i, ref in enumerate(archive, 1):
                sample = ref['id']
                require(ref['group'] == assignment[sample], 'Validation group mapping differs')
                rgb, thermal, label = read_sample(args.root, sample, upstream_cfg, False)
                rgb_before, label_before = rgb.clone(), label.clone()
                gt = label[0].cpu().numpy()
                raw = cv2.imread(str(args.root / 'Modal' / f'{sample}.png'), cv2.IMREAD_GRAYSCALE)
                require(raw is not None and raw.shape == tuple(r2a['image_shape']) and raw.dtype == np.uint8
                        and torch.equal(thermal, normalized_thermal(raw, upstream_cfg)), 'Clean thermal protocol differs')
                clean = predict(model, rgb, thermal)
                matrix = confusion(gt, clean)
                require(matrix.tolist() == ref['confusion_matrix'], 'Epoch 5 clean per-image confusion differs')
                clean_digest = digest(clean)
                clean_scores = scores(matrix)
                append(clean_log, {'id': sample, 'group': ref['group'], 'prediction_sha256': clean_digest,
                                   'confusion_matrix': matrix.tolist(), **clean_scores})
                for spec in specs:
                    raw_out, support, seed = perturbed(raw, spec, sample, r2a)
                    key = 'identity' if spec['kind'] == 'identity' else spec['position']
                    require(np.array_equal(support, all_masks[key]), 'Boolean support differs from R2a')
                    changed = normalized_thermal(raw_out, upstream_cfg)
                    outside = torch.from_numpy(~support).cuda()
                    require(torch.equal(changed[:, :, outside], thermal[:, :, outside]), 'Changed thermal outside ROI')
                    pred = predict(model, rgb, changed)
                    if key == 'identity':
                        require(np.array_equal(clean, pred), 'Identity changed prediction')
                    append(pairs[spec['id']], {
                        'id': sample, 'group': ref['group'], 'case': spec['id'], 'support': key,
                        'effective_seed': seed, 'raw_perturbed_sha256': digest(raw_out),
                        'changed_input_pixels': int(np.count_nonzero(raw_out != raw)),
                        'prediction_sha256': digest(pred), 'clean_prediction_sha256': clean_digest,
                        **pair(clean, pred, gt, support, bands[key], clean_scores)})
                require(torch.equal(rgb, rgb_before) and torch.equal(label, label_before), 'RGB or GT mutated')
                if i % 25 == 0 or i == len(archive):
                    print(f'{family}: evaluated {i}/{len(archive)} images', flush=True)
        require(model_digest(model) == final['model_sha256'] and sha256(last) == final['last_sha256']
                and not model.training, 'Checkpoint or inference model changed')
        cert = {'status': 'PASS', 'family': family, 'signature': signature, 'checkpoint_sha256': final['last_sha256'],
                'model_sha256': final['model_sha256'], 'strict_model_keys': str(keys), 'images': len(archive),
                'identity_exact': len(archive), 'clean_per_image_confusion_exact': True,
                'pair_count': len(archive) * len(specs), 'test_pixels_read': 0, 'seconds': time.perf_counter() - start,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'evidence_sha256': {p.name: sha256(p) for p in sorted(output.glob('*.jsonl'))}}
        write_json(output / 'certificate.json', cert)


def run(args):
    cfg, r2a, split, graph, signature = gpu_inputs(args, validation_only=True)
    audit = read(args.evidence / 'audit-epoch5.json')
    require(audit['status'] == 'PASS' and audit['through_epoch'] == 5
            and audit['signature'] == signature and tuple(audit['arms']) == ARMS,
            'All three matched Epoch 5 runs must pass audit before evaluation')
    all_masks = masks(r2a)
    bands = {k: distance_bands(m, r2a['distance_upper_bounds_px']) for k, m in all_masks.items()}
    specs = list(cases(r2a))
    out = args.evidence / 'perturbation'
    out.mkdir(exist_ok=False)
    write_json(out / 'cases.json', specs)
    write_json(out / 'support.json', {k: mask_record(m) for k, m in all_masks.items()})
    assignment = {i: g[0] for g in graph['groups'] for i in g}
    for arm in ARMS:
        evaluate_arm(args, arm, cfg, r2a, split, assignment, signature, audit['arms'][arm], all_masks, bands, specs)
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    for name in ('upstream', 'init', 'root', 'evidence', 'checkpoint-dir'):
        parser.add_argument('--' + name, required=True, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    require(not args.checkpoint_dir.is_relative_to(ROOT), 'Checkpoint files must stay outside the repository')
    run(args)


if __name__ == '__main__':
    main()
