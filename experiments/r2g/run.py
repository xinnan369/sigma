"""Gated full-scale paired scan replay on the frozen R2d cohort."""
import argparse
import gzip
import os
from pathlib import Path
import platform
import subprocess
import time

import cv2
import numpy as np
import torch

from experiments.r1.assets import sha256
from experiments.r1c.pilot import construct_model, read_sample
from experiments.r1g.partition import ROOT, read, require
from experiments.r2a.metrics import digest, distance_bands
from experiments.r2a.run import masks, model_digest, normalized_thermal, perturbed, records
from experiments.r2b.run import verify_mapping
from experiments.r2c.protocol import ARMS, gpu_inputs
from experiments.r2d.models import fusion
from experiments.r2e.run import R2D, assets
from experiments.r2g.journal import Journal, checksum, encode
from experiments.r2g.replay import image


CONFIG = ROOT / 'configs/R2g_protocol.json'


def publish(path, payload):
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def freeze(args):
    import selective_scan_cuda_core
    import triton

    cfg = read(CONFIG)
    require(cfg['version'] == 'r2g-native-fma-v1' and cfg['seeds'] == list(range(3407, 3412))
            and tuple(cfg['arms']) == ARMS and cfg['pilot_seed'] == 3407
            and cfg['masks'] == list(range(8)) and cfg['images'] == 157 and cfg['groups'] == 54,
            'Frozen R2g protocol differs')
    base = ['git', '-C', str(ROOT)]
    tracked_paths = ['experiments', 'configs', 'reports/evidence/r1g', 'reports/evidence/r2d',
                     'reports/evidence/r2f', 'reports/evidence/r2a']
    subprocess.run(base + ['diff', '--quiet', cfg['r2f_merge'], '--', *tracked_paths,
                           ':(exclude)experiments/r2g', ':(exclude)configs/R2g_protocol.json'], check=True)
    tracked = subprocess.check_output(base + ['ls-tree', '-r', '--name-only', cfg['r2f_merge'],
                                              '--', 'experiments', 'configs'], text=True).splitlines()
    sources = {name: sha256(ROOT / name) for name in tracked if name.endswith(('.py', '.json'))}
    sources.update({str(p.relative_to(ROOT).as_posix()): sha256(p) for p in Path(__file__).parent.glob('*.py')})
    pilot_file = ROOT / 'reports/evidence/r2f/pilot/certificate.json'
    pilot = read(pilot_file)
    require(pilot['phase'] == 'r2f-pilot' and pilot['status'] == 'PASS'
            and all(sources[f'experiments/r2f/{name}'] == value for name, value in
                    pilot['fingerprints']['r2f_implementation_sha256'].items()), 'Accepted R2f code differs')
    _, r2a, split, _, runtime = gpu_inputs(args, validation_only=True)
    require(len(set(split['validation_ids'])) == 157 and len(set(r2a['pilot_ids'])) == 12
            and set(r2a['pilot_ids']) <= set(split['validation_ids']), 'Pilot and validation cohort differ')
    frozen = assets(args.checkpoint_dir, cfg['seeds'])
    checkpoints = {f'{s}/{a}': meta['last_sha256'] for (s, a), (_, meta, _) in frozen.items()}
    require(checkpoints == pilot['fingerprints']['checkpoints'], 'Pilot checkpoint ledger differs')
    refs = {f'{s}/{a}/{f}': sha256(R2D / f'seed{s}' / 'perturbation' / a / f)
            for s, a in frozen for f in ('clean.jsonl', 'pairs-primary.jsonl')}
    device = torch.cuda.get_device_properties(0)
    fingerprint = {'protocol_sha256': sha256(CONFIG), 'source_sha256': checksum(encode(sources)),
                   'sources': sources, 'upstream': runtime['upstream_commit'],
                   'split_sha256': split['split_sha256'], 'graph_sha256': split['graph_sha256'],
                   'assets_sha256': split['selected_assets_sha256'], 'init_sha256': runtime['init_sha256'],
                   'r2f_certificate_sha256': sha256(pilot_file), 'checkpoints': checkpoints, 'r2d_refs': refs,
                   'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                                   'cuda': torch.version.cuda, 'triton': triton.__version__,
                                   'cv2': cv2.__version__, 'numpy': np.__version__,
                                   'extension_sha256': sha256(Path(selective_scan_cuda_core.__file__)),
                                   'gpu': device.name, 'gpu_memory_bytes': device.total_memory}}
    return cfg, r2a, split, frozen, fingerprint


def inputs(args, sample, config, r2a, roi, ref):
    rgb, clean, label = read_sample(args.root, sample, config, False)
    raw = cv2.imread(str(args.root / 'Modal' / f'{sample}.png'), cv2.IMREAD_GRAYSCALE)
    changed, actual, _ = perturbed(raw, {'id': 'primary', **r2a['primary']}, sample, r2a)
    require(np.array_equal(actual, roi) and torch.equal(clean, normalized_thermal(raw, config))
            and digest(changed) == ref['raw_perturbed_sha256'], 'Frozen input differs')
    corrupt = normalized_thermal(changed, config)
    tensors = rgb, clean, corrupt, label
    return tensors, {name: digest(t.cpu().numpy()) for name, t in zip(
        ('rgb', 'thermal', 'thermal_corrupt', 'label'), tensors, strict=True)}


def cohort(args, folder, seed, arm, ids, frozen, fingerprint, r2a, roi, bands):
    start = time.perf_counter()
    key = f'{seed}-{arm}'
    path, meta, signature = frozen[seed, arm]
    prior = R2D / f'seed{seed}' / 'perturbation' / arm
    references = [{r['id']: r for r in records(prior / name)} for name in ('clean.jsonl', 'pairs-primary.jsonl')]
    require(all(set(ids) <= set(ref) and len(ref) == 157 for ref in references)
            and len({r['group'] for r in references[0].values()}) == 54, 'R2d reference cohort differs')
    identity = {'phase': args.phase, 'seed': seed, 'arm': arm, 'ids': ids, 'checkpoint_sha256': meta['last_sha256'],
                'fingerprint_sha256': checksum(encode(fingerprint))}
    with Journal(folder / f'{key}.sqlite', identity) as journal:
        completed = journal.read()
        require(not journal.read(failures=True), f'{key} has preserved failures; cohort blocked')
        require(all(r['group'] == references[0][i]['group'] and r['cleanup'] and
                    r['prediction_sha256']['clean'] == references[0][i]['prediction_sha256'] and
                    r['prediction_sha256']['natural_corrupt'] == references[1][i]['prediction_sha256']
                    for i, r in completed.items()), 'Resumed result differs from native reference')
        if completed:
            layout_digest = sha256(folder / f'{key}-layout.json')
            require(all(r['layout_sha256'] == layout_digest and r['seed'] == seed and r['arm'] == arm
                        and r['checkpoint_sha256'] == meta['last_sha256']
                        and r['source_sha256'] == fingerprint['source_sha256']
                        and r['protocol_sha256'] == fingerprint['protocol_sha256']
                        for r in completed.values()), 'Resumed record or layout fingerprint differs')
        if len(completed) != len(ids):
            state = torch.load(path, map_location='cpu', weights_only=False)
            require(state['family'] == arm and state['signature'] == signature and state['updates'] == 785
                    and state['sampler']['epoch'] == 5 and state['sampler']['cursor'] == 1254,
                    'Checkpoint metadata differs')
            widths = read(ROOT / 'configs/R2d_protocol.json')['widths']
            with (construct_model(args.upstream, args.init) as (model, config, _),
                  fusion(model, arm, widths), torch.inference_mode()):
                from models.encoders import vmamba

                model.load_state_dict(state['model'], strict=True)
                del state
                model = model.cuda().eval()
                require(model_digest(model) == meta['model_sha256'], 'Loaded weights differ')
                for h, w in ((120, 160), (60, 80), (30, 40), (15, 20)):
                    verify_mapping(vmamba, h, w)
                for sample in ids:
                    if sample in completed:
                        continue
                    refs = [r[sample] for r in references]
                    begin = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats()
                    try:
                        tensors, hashes = inputs(args, sample, config, r2a, roi, refs[1])
                        row, layout = image(model, vmamba, arm, *tensors, refs, roi, bands)
                        torch.cuda.synchronize()
                        if completed:
                            require(checksum(encode(layout)) == next(iter(completed.values()))['layout_sha256'],
                                    'Scan layout varies across images')
                        else:
                            publish(folder / f'{key}-layout.json', encode(layout).encode())
                    except Exception as exc:
                        journal.save(sample, {'error_type': type(exc).__name__, 'error': str(exc)}, failure=True)
                        print(f'FAIL {seed}/{arm}/{sample}: {exc}', flush=True)
                        raise
                    row.update({'group': refs[0]['group'], 'input_sha256': hashes,
                                'seed': seed, 'arm': arm, 'checkpoint_sha256': meta['last_sha256'],
                                'source_sha256': fingerprint['source_sha256'],
                                'protocol_sha256': fingerprint['protocol_sha256'],
                                'layout_sha256': checksum(encode(layout)),
                                'seconds': time.perf_counter() - begin,
                                'peak_allocated': torch.cuda.max_memory_allocated(),
                                'peak_reserved': torch.cuda.max_memory_reserved()})
                    journal.save(sample, row)
                    completed[sample] = row
                    print(f'{args.phase} {seed}/{arm} {len(completed)}/{len(ids)} {sample} '
                          f'{row["seconds"]:.2f}s peak={row["peak_allocated"]}', flush=True)
                require(model_digest(model) == meta['model_sha256'] and sha256(path) == meta['last_sha256'],
                        'Model or checkpoint changed')
            del model
            torch.cuda.empty_cache()
        completed = journal.read()
    rows = [completed[i] for i in ids]
    layout_path = folder / f'{key}-layout.json'
    require(all(sha256(layout_path) == r['layout_sha256'] for r in rows), 'Saved layout hash differs')
    name = f'{key}.jsonl.gz'
    payload = '\n'.join(encode({'sha256': checksum(encode(r)), 'record': r}) for r in rows) + '\n'
    publish(folder / name, gzip.compress(payload.encode(), mtime=0))
    return {'seed': seed, 'arm': arm, 'count': len(rows), 'records': name, 'sha256': sha256(folder / name),
            'identity': identity, 'wall_seconds_this_launch': time.perf_counter() - start,
            'image_seconds': sum(r['seconds'] for r in rows), 'peak_allocated': max(r['peak_allocated'] for r in rows),
            'peak_reserved': max(r['peak_reserved'] for r in rows)}


def validate_gate(folder, fingerprint, ids, seed):
    gate = read(folder / 'certificate.json')
    require(gate['phase'] == 'r2g-pilot' and gate['status'] == 'PASS' and gate['fingerprints'] == fingerprint
            and gate['ids'] == ids and {(r['seed'], r['arm']) for r in gate['summary']} == {(seed, a) for a in ARMS}
            and len(gate['summary']) == len(ARMS), 'Independent R2g pilot gate failed')
    for item in gate['summary']:
        require(item['count'] == len(ids) and sha256(folder / item['records']) == item['sha256'],
                'Independent R2g pilot evidence differs')
    return sha256(folder / 'certificate.json')


def run(args):
    cfg, r2a, split, frozen, fingerprint = freeze(args)
    print(encode({'phase': args.phase, 'source_sha256': fingerprint['source_sha256'],
                  'protocol_sha256': fingerprint['protocol_sha256'], 'environment': fingerprint['environment']}),
          flush=True)
    if args.phase == 'preflight':
        return
    pilot = args.phase == 'pilot'
    gate = None if pilot else validate_gate(args.evidence / 'pilot', fingerprint, r2a['pilot_ids'], cfg['pilot_seed'])
    ids = r2a['pilot_ids'] if pilot else split['validation_ids']
    roi = masks(r2a)['center']
    require(int(roi.sum()) == 64 * 64 and r2a['primary']['fill'] == 0, 'Primary ROI differs')
    bands = distance_bands(roi, r2a['distance_upper_bounds_px'])
    folder = args.evidence / args.phase
    folder.mkdir(parents=True, exist_ok=True)
    # Keep a single writer across cohorts, including pre/post-flight exports.
    import fcntl

    with (folder / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        summary = [cohort(args, folder, s, a, ids, frozen, fingerprint, r2a, roi, bands)
                   for s in ([cfg['pilot_seed']] if pilot else cfg['seeds']) for a in ARMS]
        cert = {'phase': f'r2g-{args.phase}', 'status': 'PASS', 'ids': ids, 'summary': summary,
                'fingerprints': fingerprint, 'pilot_gate_sha256': gate, 'official_test_pixels_read': 0}
        publish(folder / 'certificate.json', (encode(cert) + '\n').encode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('preflight', 'pilot', 'full'))
    for name in ('upstream', 'init', 'root', 'evidence', 'checkpoint-dir'):
        parser.add_argument('--' + name, required=True, type=lambda p: Path(p).resolve())
    run(parser.parse_args())


if __name__ == '__main__':
    main()

