"""Run frozen path interventions with a 12-image, three-architecture pilot."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from experiments.r1.assets import sha256
from experiments.r1c.data import canonical_sha, write_json
from experiments.r1c.pilot import construct_model, read_sample
from experiments.r1g.partition import ROOT, read, require
from experiments.r2a.metrics import confusion, digest, distance_bands, pair, scores
from experiments.r2a.run import append, masks, model_digest, normalized_thermal, perturbed, records
from experiments.r2b.run import verify_mapping
from experiments.r2b.state import Scan
from experiments.r2c.protocol import ARMS, gpu_inputs
from experiments.r2d.models import fusion
from experiments.r2e.paths import factor, geometry, intercept, verify_reference

R2D = ROOT / 'reports/evidence/r2d'


def assets(checkpoints, seeds):
    result = {}
    for seed in seeds:
        base = R2D / f'seed{seed}'
        audit = read(base / 'audit-epoch5.json')
        require(audit['status'] == 'PASS' and audit['signature']['seed'] == seed, 'R2d seed audit differs')
        for arm in ARMS:
            ledger = read(base / 'training' / arm / 'checkpoint-ledger.json')[-1]
            path = checkpoints / f'seed{seed}' / arm / 'last.pt'
            require(ledger['epoch'] == 5 and ledger['updates'] == 785 and path.is_file()
                    and sha256(path) == ledger['last_sha256'] == audit['arms'][arm]['last_sha256'],
                    f'Epoch 5 checkpoint unavailable or mismatched: seed {seed}, {arm}')
            result[seed, arm] = (path, ledger, audit['signature'])
    return result


def predict(logits):
    require(logits.shape == (1, 9, 480, 640) and torch.isfinite(logits).all().item(), 'Invalid replay logits')
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def differences(current, baseline):
    tensors = current if isinstance(current, tuple) else (current,)
    refs = baseline if isinstance(baseline, tuple) else (baseline,)
    return [float((a - b).square().mean().sqrt().item()) for a, b in zip(tensors, refs, strict=True)]


def baseline_trace(model, vmamba, arm, rgb, clean, corrupt, refs):
    layouts = {}

    def observe(key, scope, grid, native, args):
        entry = {'call': list(key), 'scope': scope, **geometry(scope, key[2], grid, Scan(*args))}
        require(key not in layouts or entry == layouts[key], 'Paired layout differs')
        layouts[key] = entry
        return native(*args)

    with intercept(model, vmamba, arm, observe):
        a = model(rgb, clean)
    with intercept(model, vmamba, arm, observe):
        b = model(rgb, corrupt)
    require(torch.equal(a, model(rgb, clean)) and torch.equal(b, model(rgb, corrupt)), 'Trace changed native outputs')
    hashes = [digest(predict(x)) for x in (a, b)]
    require(hashes == [r['prediction_sha256'] for r in refs], 'R2d baseline hashes differ')
    return list(layouts.values()), hashes


def one_image(model, vmamba, arm, rgb, clean, corrupt, label, refs, roi, bands, *, pilot):
    captured, replacements, layouts, audits, local = {}, {}, {}, {}, {}
    module_clean, module_deltas = {}, {}

    def capture(key, scope, grid, native, args):
        scan = Scan(*args)
        index = key[2]
        layouts[key] = (scope, geometry(scope, index, grid, scan))
        out, prefix = native(*args)
        captured[key] = scan, out, prefix if pilot else None
        return out, prefix

    def compare(key, scope, grid, native, args):
        scan = Scan(*args)
        require(layouts[key] == (scope, geometry(scope, key[2], grid, scan)), 'Paired layout differs')
        out, prefix = native(*args)
        prior, base, prior_prefix = captured.pop(key)
        values = factor(native, prior, scan, base, out)
        replacements[key] = values
        if pilot:
            try:
                audits[key] = verify_reference(prior, scan, base, out, prior_prefix, prefix, values)
            except ValueError as exc:
                raise ValueError(f'{key} {scope} {layouts[key][1]}: {exc}') from exc
            local[key] = {'u_delta_rms': differences(scan.u, prior.u)[0],
                          'delta_delta_rms': differences(scan.step_sizes(), prior.step_sizes())[0],
                          'b_delta_rms': differences(scan.b, prior.b)[0], 'c_delta_rms': differences(scan.c, prior.c)[0],
                          'a_delta_rms': 0., 'scan_delta_rms_by_mask': [differences(values.output(m), base)[0]
                                                                         for m in range(8)]}
        return out, prefix

    with intercept(model, vmamba, arm, capture, module_clean.__setitem__ if pilot else None):
        clean_logits = model(rgb, clean)

    def summarize_module(key, output):
        module_deltas[key] = differences(output, module_clean.pop(key))

    with intercept(model, vmamba, arm, compare, summarize_module if pilot else None):
        corrupt_logits = model(rgb, corrupt)
    require(not captured and not module_clean and set(layouts) == set(replacements), 'Incomplete paired scans')
    a, b = predict(clean_logits), predict(corrupt_logits)
    require(digest(a) == refs[0]['prediction_sha256'] and digest(b) == refs[1]['prediction_sha256'],
            'Clean/corrupt prediction hash differs from R2d')
    gt = label[0].cpu().numpy()
    baseline = scores(confusion(gt, a))
    masks_by_scope = sorted({s for s, _ in layouts.values()})
    masks_by_scope.append('all')
    outcomes = {'natural_corrupt': pair(a, b, gt, roi, bands, baseline)}
    for scope in masks_by_scope:
        for mask in range(8):
            if mask == 0 and not pilot:
                continue
            seen = set()

            def replay(key, name, grid, native, args):
                require(key not in seen and layouts[key][0] == name, 'Replay call identity differs')
                seen.add(key)
                if name == scope or scope == 'all':
                    out = replacements[key].output(mask)
                    require(out.shape == args[0].shape and out.dtype == args[0].dtype, 'Replay output shape differs')
                    # Prefix is unused by the inference-only upstream autograd wrapper.
                    return out, None
                return native(*args)

            with intercept(model, vmamba, arm, replay):
                output = model(rgb, clean)
            require(seen == set(layouts), 'Replay omitted a native scan')
            if mask == 0:
                require(torch.equal(output, clean_logits), 'Identity replay differs from native clean logits')
            else:
                outcomes[f'{scope}/{mask}'] = pair(a, predict(output), gt, roi, bands, baseline)
    if pilot:
        require(torch.equal(clean_logits, model(rgb, clean)) and torch.equal(corrupt_logits, model(rgb, corrupt)),
                'Hooks or scan patch leaked after intervention')
    describe = [{'call': list(k), 'scope': s, **g} for k, (s, g) in layouts.items()]
    detail = [{'call': list(k), 'reference': a, **local[k]} for k, a in audits.items()]
    return outcomes, describe, detail, [{'module': list(k), 'delta_rms': v} for k, v in module_deltas.items()]


def run(args):
    cfg = read(ROOT / 'configs/R2e_protocol.json')
    _, r2a, split, _, _ = gpu_inputs(args, validation_only=True)
    frozen = assets(args.checkpoint_dir, cfg['seeds'])
    widths = read(ROOT / 'configs/R2d_protocol.json')['widths']
    fingerprints = {'config_sha256': canonical_sha(cfg),
                    'sources': {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))
                                if p.name != 'analysis.py'},
                    'checkpoints': {f'{s}/{a}': meta['last_sha256'] for (s, a), (_, meta, _) in frozen.items()}}
    pilot = args.phase != 'full'
    if args.phase == 'full':
        gate = read(args.evidence / 'pilot/certificate.json')
        require(gate['status'] == 'PASS' and gate['fingerprints'] == fingerprints
                and gate['ids'] == r2a['pilot_ids'], 'All-arm pilot gate failed')
    ids = r2a['pilot_ids'] if pilot else split['validation_ids']
    roi = masks(r2a)['center']
    bands = distance_bands(roi, r2a['distance_upper_bounds_px'])
    folder = args.evidence / args.phase
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / 'assets.json', fingerprints)
    summary = []
    for seed in ([cfg['pilot_seed']] if pilot else cfg['seeds']):
        for arm in ARMS:
            import cv2

            start = time.perf_counter()
            path, meta, signature = frozen[seed, arm]
            state = torch.load(path, map_location='cpu', weights_only=False)
            require(state['family'] == arm and state['signature'] == signature and state['updates'] == 785
                    and state['sampler']['epoch'] == 5 and state['sampler']['cursor'] == 1254, 'Checkpoint metadata differs')
            prior = R2D / f'seed{seed}' / 'perturbation' / arm
            references = [{r['id']: r for r in records(prior / file)}
                          for file in ('clean.jsonl', 'pairs-primary.jsonl')]
            outfile = folder / f'{seed}-{arm}.jsonl'
            with (construct_model(args.upstream, args.init) as (model, upstream_cfg, _),
                  fusion(model, arm, widths), torch.inference_mode(),
                  outfile.open('x', encoding='utf-8', newline='\n') as stream):
                from models.encoders import vmamba

                model.load_state_dict(state['model'], strict=True)
                del state
                model = model.cuda().eval()
                require(model_digest(model) == meta['model_sha256'], 'Loaded weights differ')
                for h, w in ((120, 160), (60, 80), (30, 40), (15, 20)):
                    verify_mapping(vmamba, h, w)
                torch.cuda.reset_peak_memory_stats()
                for i, sample in enumerate(ids, 1):
                    rgb, clean, label = read_sample(args.root, sample, upstream_cfg, False)
                    raw = cv2.imread(str(args.root / 'Modal' / f'{sample}.png'), cv2.IMREAD_GRAYSCALE)
                    changed, actual_roi, _ = perturbed(raw, {'id': 'primary', **r2a['primary']}, sample, r2a)
                    require(np.array_equal(actual_roi, roi) and torch.equal(clean, normalized_thermal(raw, upstream_cfg))
                            and digest(changed) == references[1][sample]['raw_perturbed_sha256'], 'Frozen input differs')
                    refs = [r[sample] for r in references]
                    if args.phase == 'baseline':
                        layout, hashes = baseline_trace(model, vmamba, arm, rgb, clean,
                                                        normalized_thermal(changed, upstream_cfg), refs)
                        append(stream, {'id': sample, 'group': refs[0]['group'], 'prediction_sha256': hashes})
                        if i == 1:
                            write_json(folder / f'{seed}-{arm}-layout.json', layout)
                        continue
                    outcomes, layout, numeric, module = one_image(
                        model, vmamba, arm, rgb, clean, normalized_thermal(changed, upstream_cfg), label,
                        refs, roi, bands, pilot=pilot)
                    require(all(outcomes['natural_corrupt'][k] == refs[1][k] for k in outcomes['natural_corrupt']),
                            'Natural perturbation metrics differ from R2d')
                    append(stream, {'id': sample, 'group': refs[0]['group'], 'clean_matrix': refs[0]['confusion_matrix'],
                                    'outcomes': outcomes, **({'numeric': numeric, 'module': module} if pilot else {})})
                    stream.flush()
                    if i == 1:
                        write_json(folder / f'{seed}-{arm}-layout.json', layout)
                    if pilot or i % 10 == 0 or i == len(ids):
                        print(f'{args.phase} {seed}/{arm} {i}/{len(ids)}', flush=True)
                require(model_digest(model) == meta['model_sha256'] and sha256(path) == meta['last_sha256'],
                        'Model or checkpoint changed')
            summary.append({'seed': seed, 'arm': arm, 'count': len(ids), 'seconds': time.perf_counter() - start,
                            'peak_allocated': torch.cuda.max_memory_allocated(), 'records_sha256': sha256(outfile)})
            del model
            torch.cuda.empty_cache()
    write_json(folder / 'certificate.json', {'status': 'PASS', 'phase': args.phase, 'ids': ids, 'summary': summary,
                                            'fingerprints': fingerprints, 'test_pixels_read': 0})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('baseline', 'pilot', 'full'))
    for name in ('upstream', 'init', 'root', 'evidence', 'checkpoint-dir'):
        parser.add_argument('--' + name, required=True, type=lambda p: Path(p).resolve())
    run(parser.parse_args())


if __name__ == '__main__':
    main()
