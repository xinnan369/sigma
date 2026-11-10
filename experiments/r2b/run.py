"""Run frozen model pilot and gated full-cohort state resets on MFNet."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from experiments.r1.assets import sha256
from experiments.r1c.data import canonical_sha, read_lists, write_json
from experiments.r1c.pilot import construct_model, read_sample
from experiments.r1e.train import signatures
from experiments.r1g.partition import ROOT, read, references, require, selected_files, verify
from experiments.r2a.hooks import modules
from experiments.r2a.metrics import digest, distance_bands, pair
from experiments.r2a.run import append, masks, model_digest, normalized_thermal, perturbed, records
from experiments.r2b.audit import compare, four_orders, synthetic
from experiments.r2b.instrument import intercept, reset_readout, summaries, unpack
from experiments.r2b.state import coordinates
from experiments.r2b.triton_state import trajectory

R1G = ROOT / 'reports/evidence/r1g'
R2A = ROOT / 'reports/evidence/r2a'


def verify_mapping(vmamba, height, width):
    count = height * width
    rgb = torch.arange(count, dtype=torch.float32).reshape(1, 1, height, width)
    thermal = rgb + count
    observed = vmamba.CrossScan_multimodal.apply(rgb, thermal)[0, :, 0].tolist()
    expected = [[row * width + col + (count if modality == 'thermal' else 0)
                 for modality, row, col in (coordinates('concat', d, t, height, width) for t in range(2 * count))]
                for d in range(2)]
    require(observed == expected and vmamba.CrossScan.apply(rgb)[0, :, 0].tolist() == four_orders(height, width),
            'Time-to-grid mapping does not match upstream scan')


def one_image(model, vmamba, cfg, r2a, upstream_cfg, root, sample, ref, primary_ref, *, pilot):
    from experiments.r2b.state import check_close

    primary = cfg['primary']
    target = primary['module']
    rgb, thermal, label = read_sample(root, sample, upstream_cfg, False)
    raw = cv2.imread(str(root / 'Modal' / (sample + '.png')), cv2.IMREAD_GRAYSCALE)
    spec = {'id': 'primary', **r2a['primary']}
    raw_corrupt, roi, _ = perturbed(raw, spec, sample, r2a)
    require(np.array_equal(roi, masks(r2a)['center'])
            and torch.equal(normalized_thermal(raw, upstream_cfg), thermal), 'Frozen R2a raw input changed')
    corrupted = normalized_thermal(raw_corrupt, upstream_cfg)
    inverse = torch.from_numpy(~roi).cuda()
    require(torch.equal(corrupted[:, :, inverse], thermal[:, :, inverse]), 'Thermal changed outside ROI')
    saved, trace, audited = {}, [], []

    def forward(x, callback):
        output = []
        hook = modules(model)[target].register_forward_hook(lambda _, args, y: output.append(y.detach()))
        try:
            with intercept(model, vmamba.selective_scan_cuda, callback):
                logits = model(rgb, x)
        finally:
            hook.remove()
        require(len(output) == 1 and logits.shape == (1, 9, 480, 640)
                and torch.isfinite(logits).all().item(), 'Invalid fusion output or logits')
        return logits, output[0]

    def capture(name, grid, index, args, native, prefix, mode):
        scan = unpack(name, grid, index, args)
        h = trajectory(scan)
        audited.append({'mode': mode, 'module': name, 'call': index, 'grid': list(grid),
                        'u_shape': list(scan.u.shape), 'nrows': scan.nrows, 'state_width': scan.a.shape[1],
                        **compare(scan, h, native, prefix)})
        key = name, index
        if mode == 'clean':
            saved[key] = h, scan.c
        else:
            clean, clean_c = saved[key]
            trace.append({'module': name, 'call': index, 'grid': list(grid),
                          'states': summaries(name.split('.')[0], grid, index, scan, h, clean, clean_c,
                                              cfg['state_summary_block'])})
        return native

    clean_logits, clean_out = forward(thermal, lambda *args: capture(*args, 'clean'))
    corrupt_logits, corrupt_out = forward(corrupted, lambda *args: capture(*args, 'corrupt'))

    def pred(logits):
        return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)

    clean_pred, corrupt_pred = pred(clean_logits), pred(corrupt_logits)
    require(digest(clean_pred) == ref['prediction_sha256'] and digest(corrupt_pred) == primary_ref['prediction_sha256'],
            'Instrumented clean or corrupt predictions differ from accepted R2a')
    clean_score = {k: ref[k] for k in ('miou', 'foreground_miou', 'iou_0_to_8')}
    bands = distance_bands(roi, r2a['distance_upper_bounds_px'])
    gt = label[0].cpu().numpy()
    stats = {}
    for condition, x, t, baseline_logits, baseline_out in [
        ('primary_reset', corrupted, primary['time'], corrupt_logits, corrupt_out),
        ('wrong_time', corrupted, cfg['wrong_time'], corrupt_logits, corrupt_out),
        *([('identity', thermal, primary['time'], clean_logits, clean_out),
           ('corrupt_self', corrupted, primary['time'], corrupt_logits, corrupt_out)] if pilot else []),
    ]:
        meta = []

        def intervention(name, grid, index, args, native, prefix):
            if name != target:
                return native
            scan = unpack(name, grid, index, args)
            direction = primary['direction']
            if condition == 'primary_reset':
                require(t == grid[0] * grid[1] - 1 and direction == 1, 'Reset must use reverse modality seam')
            coord = coordinates('concat', direction, t, *grid)
            h = trajectory(scan)
            ref_state = h[..., t] if condition == 'corrupt_self' else saved[name, index][0][..., t]
            output, values = reset_readout(scan, h, native, direction, t, ref_state, verify=pilot)
            if pilot:
                values['native_audit'] = compare(scan, h, native, prefix)
            values['coordinate'] = coord
            values['next_coordinate'] = coordinates('concat', direction, t + 1, *grid)
            meta.append(values)
            return output

        logits, out = forward(x, intervention)
        require(len(meta) == 1, 'State replacement applied to the wrong CUDA call')
        changed = pred(logits)
        if condition in ('identity', 'corrupt_self'):
            require(torch.equal(logits, baseline_logits) and torch.equal(out, baseline_out)
                    and meta[0]['changed_reset_state'] == meta[0]['changed_future_state_steps']
                    == meta[0]['changed_readout'] == 0, 'Identity reset changed values')
        stats[condition] = {**meta[0], 'module_output_max_delta': float((out-baseline_out).abs().amax().item()),
                            'logit_max_delta': float((logits-baseline_logits).abs().amax().item()),
                            'prediction_flips_vs_input_baseline': int(np.count_nonzero(changed != pred(baseline_logits))),
                            'prediction_sha256': digest(changed),
                            **pair(clean_pred, changed, gt, roi, bands, clean_score)}
    if pilot:
        require(torch.equal(model(rgb, thermal), clean_logits) and torch.equal(model(rgb, corrupted), corrupt_logits),
                'Instrumentation persisted after hook cleanup')
    return {'id': sample, 'group': ref['group'], 'clean_prediction_sha256': digest(clean_pred),
            'corrupt_prediction_sha256': digest(corrupt_pred), 'audits': audited, 'trajectory_summary': trace,
            'base_corrupt': pair(clean_pred, corrupt_pred, gt, roi, bands, clean_score), 'conditions': stats}


def run(args):
    cfg = read(ROOT / 'configs/R2b_protocol.json')
    r2a = read(ROOT / 'configs/R2a_exp0.json')
    graph, prior = references()
    split = read(R1G / 'split.json')
    require(cfg['split_sha256'] == r2a['split_sha256'] == split['split_sha256']
            and verify(split, graph, prior, read_lists(args.root)) == read(R1G / 'isolation.json')['class_support'],
            'R1g split isolation changed')
    cfg_runtime, _ = signatures(read(ROOT / 'configs/R1e_convergence.json'))
    require(cfg_runtime['config']['upstream_commit'] == cfg['upstream_commit']
            and sha256(args.init) == cfg_runtime['config']['init_sha256']
            and sha256(args.checkpoint) == cfg['checkpoint_sha256'] == r2a['checkpoint_sha256'],
            'Checkpoint, source or initializer differs')
    git = ['git', '-C', str(args.upstream)]
    require(subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip() == cfg['upstream_commit'],
            'Sigma upstream SHA differs')
    subprocess.run(git + ['diff', '--quiet', 'HEAD', '--'], check=True)
    sources = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py')) if p.name != 'analysis.py'
               and p.name != 'controls.py'}
    signature = {'config_sha256': canonical_sha(cfg), 'r2a_sha256': canonical_sha(r2a),
                 'sources': sources, 'runtime': cfg_runtime, 'split_sha256': cfg['split_sha256'],
                 'checkpoint_sha256': cfg['checkpoint_sha256']}
    pilot = args.phase == 'pilot'
    ids = r2a['pilot_ids'] if pilot else split['validation_ids']
    if not pilot:
        cert = read(args.evidence / 'pilot/certificate.json')
        require(cert['signature'] == signature and cert['status'] == 'PASS' and cert['ids'] == r2a['pilot_ids']
                and cert['causal_chain_with_prediction_flip'] > 0
                and cert['records_sha256'] == sha256(args.evidence / 'pilot/images.jsonl'), 'Pilot gate failed')
    for asset in selected_files(split):
        if Path(asset['path']).stem in ids:
            require(sha256(args.root / asset['path']) == asset['sha256'], 'Validation asset hash differs')
    refs = {row['id']: row for row in records(R2A / 'clean/clean.jsonl')}
    primary = {row['id']: row for row in records(R2A / 'full/pairs-primary.jsonl')}
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    require(state['format'] == 'r1g-expanded-scene-v1' and state['sampler']['epoch'] == 5
            and state['sampler']['cursor'] == 1254 and state['history'] == read(R1G / 'gpu/epochs.json')
            and state['signature']['runtime'] == cfg_runtime, 'Checkpoint metadata differs')
    require(torch.cuda.device_count() == 1, 'Expose one CUDA GPU')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = args.evidence / args.phase
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    path = output / 'images.jsonl'
    chains = 0
    with construct_model(args.upstream, args.init) as (model, upstream_cfg, _), torch.inference_mode():
        from models.encoders import vmamba

        model.load_state_dict(state['model'], strict=True)
        del state
        model = model.cuda().eval()
        original = model_digest(model)
        require(original == read(R2A / 'full/certificate.json')['model_state_sha256'], 'R2a model fingerprint differs')
        for h, w in ((120, 160), (60, 80), (30, 40), (15, 20)):
            verify_mapping(vmamba, h, w)
        if pilot:
            write_json(output / 'synthetic.json', synthetic(vmamba.selective_scan_cuda))
        torch.cuda.reset_peak_memory_stats()
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            for i, sample in enumerate(ids, 1):
                row = one_image(model, vmamba, cfg, r2a, upstream_cfg, args.root, sample, refs[sample],
                                primary[sample], pilot=pilot)
                chain = row['conditions']['primary_reset']
                fields = ('changed_reset_state', 'changed_future_state_steps', 'changed_readout',
                          'module_output_max_delta', 'logit_max_delta', 'prediction_flips_vs_input_baseline')
                chains += int(all(chain[field] > 0 for field in fields))
                append(stream, row)
                stream.flush()
                if pilot or i % 10 == 0 or i == len(ids):
                    print(f'{args.phase}: {i}/{len(ids)}, causal chains with argmax flips: {chains}', flush=True)
        require(model_digest(model) == original and not model.training
                and sha256(args.checkpoint) == cfg['checkpoint_sha256'], 'Model or checkpoint changed')
    cert = {'status': 'PASS' if chains else 'BLOCKED_NO_PREDICTION_CHANGE', 'phase': args.phase,
            'signature': signature, 'ids': ids, 'records_sha256': sha256(path),
            'causal_chain_with_prediction_flip': chains, 'model_state_sha256': original,
            'test_pixels_read': 0, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'seconds': time.perf_counter() - start}
    write_json(output / 'certificate.json', cert)
    print(json.dumps({k: v for k, v in cert.items() if k not in ('ids', 'signature')}))
    require(cert['status'] == 'PASS', 'No primary reset completed the causal chain to an argmax flip')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('pilot', 'full'))
    for name in ('upstream', 'init', 'root', 'checkpoint', 'evidence'):
        parser.add_argument('--' + name, required=True, type=lambda p: Path(p).resolve())
    run(parser.parse_args())


if __name__ == '__main__':
    main()
