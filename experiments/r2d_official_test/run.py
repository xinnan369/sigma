"""One-shot, zero-finetuning R2d evaluation on the MFNet 393-image official test.

Checkpoint hashes and 1179 official test-file SHA256 hashes are validated before
running the first model forward.  Never retrain, pick an epoch, or tune on test.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from experiments.r2d_official_test.metrics import compare_runs, confusion, metrics

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (3407, 3408, 3409, 3410, 3411)
ARMS = ('ssm', 'cnn', 'attention')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, obj):
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    with pending.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    pending.replace(path)


def test_assets(root):
    """Use author-provided official ID and manifest files; never rediscover/sort a subset."""
    ids = read(ROOT / 'reports/evidence/r1c/lists.json')['lists']['test.txt']
    manifest = read(ROOT / 'reports/evidence/r1b/data-manifest.json')
    train_split = read(ROOT / 'reports/evidence/r1g/split.json')
    names = ids['ids']
    require(len(names) == len(set(names)) == 393 and
            not (set(names) & (set(train_split['train_originals']) | set(train_split['validation_ids']))),
            'Official test ID list overlaps frozen R2d train/validation')
    require(sha(root / 'test.txt') == ids['sha256'] == manifest['test_txt_sha256'],
            'Official test.txt SHA256 differs')
    listed = (root / 'test.txt').read_text(encoding='utf-8').splitlines()
    require(listed == names, 'Official test.txt order differs')
    require(manifest['errors'] == [] and manifest['sample_count'] == 393 and
            manifest['valid_file_records'] == manifest['expected_files'] == 1179,
            'Original MFNet test archive is incomplete')
    records = manifest['files']
    wanted = {f'{kind}/{name}.png' for name in names for kind in ('RGB', 'Modal', 'Label')}
    provided = [entry['path'] for entry in records]
    require(len(provided) == len(set(provided)) == len(wanted) and set(provided) == wanted,
            'Original MFNet test-file inventory differs')
    for entry in records:
        relative = entry['path']
        path = root / relative
        require(path.is_file() and path.stat().st_size == entry['bytes']
                and sha(path) == entry['sha256'], f'Official test asset SHA256 differs: {relative}')
    return {
        'sample_count': len(names), 'file_count': len(records),
        'test_txt_sha256': ids['sha256'],
        'test_records_sha256': manifest['file_records_sha256'],
        'training_validation_overlap_count': 0,
        'test_pixel_decodes_in_preflight': 0,
        'class_pixel_counts': [manifest['class_pixel_counts'][str(i)] for i in range(9)],
        'ignore_255_pixel_count': manifest['ignore_255_pixel_count'],
    }, names


def checkpoint_inventory(folder, evidence=ROOT / 'reports/evidence/r2d'):
    """Check every epoch-5 checkpoint before opening any pickle."""
    require(folder.name == 'sigma-r2d-runs' and folder.is_dir(),
            'Use original external sigma-r2d-runs directory')
    result = {}
    for seed in SEEDS:
        audit = read(evidence / f'seed{seed}/audit-epoch5.json')
        require(audit['status'] == 'PASS' and audit['through_epoch'] == 5
                and audit['test_pixels_read'] == 0 and audit['signature']['seed'] == seed,
                f'Untrusted R2d seed audit: {seed}')
        for arm in ARMS:
            key = f'{arm}/{seed}'
            original = evidence / f'seed{seed}/training/{arm}'
            ledger = read(original / 'checkpoint-ledger.json')
            init = read(original / 'initialization.json')
            last = folder / f'seed{seed}' / arm / 'last.pt'
            require(len(ledger) == 5 and ledger[-1]['epoch'] == 5
                    and ledger[-1]['updates'] == 785
                    and audit['arms'][arm]['last_sha256'] == ledger[-1]['last_sha256']
                    and init['signature'] == audit['signature'] and init['family'] == arm,
                    f'Archive ledger or initialization mismatch: {key}')
            require(last.is_file(), f'Checkpoint missing: {key}')
            actual_sha256 = sha(last)
            require(actual_sha256 == ledger[-1]['last_sha256'], f'Checkpoint modified: {key}')
            result[key] = {'checkpoint': str(last), 'sha256': actual_sha256,
                           'model_sha256': ledger[-1]['model_sha256'],
                           'signature': audit['signature'], 'family': arm, 'seed': seed,
                           'initialization': init,
                           'training_history': read(original / 'epochs.json')}
    return result


def load_frozen(model, state, expected, version):
    require(state.get('format') == version and state.get('family') == expected['family']
            and state.get('signature') == expected['signature']
            and state.get('initialization') == expected['initialization']
            and state.get('updates') == 785
            and state.get('history') == expected['training_history']
            and state.get('sampler', {}).get('epoch') == 5
            and state.get('sampler', {}).get('cursor') == 1254,
            'Checkpoint metadata is not the approved fifth-epoch state')
    model.load_state_dict(state['model'], strict=True)


def run(args):
    import torch
    from experiments.r1.assets import sha256
    from experiments.r1c.pilot import construct_model, read_sample
    from experiments.r2a.run import model_digest
    from experiments.r2c.protocol import gpu_inputs
    from experiments.r2d.models import fusion
    from experiments.r2d.profile import design

    require(args.root.is_dir() and args.upstream.is_dir() and args.init.is_file(),
            'Missing external dataset, pinned Sigma checkout, or initialization')
    require(not args.output.is_relative_to(ROOT)
            and not args.checkpoint_dir.is_relative_to(ROOT)
            and not args.output.is_relative_to(args.root)
            and not args.root.is_relative_to(args.output),
            'Output must be outside the repository and dataset')

    # Independent, immutable inputs before any official-test inference.
    frozen, names = test_assets(args.root)
    checkpoints = checkpoint_inventory(args.checkpoint_dir)
    config, _ = design()
    gpu_cfg, _, _, _, runtime = gpu_inputs(args, validation_only=True)
    require(sha256(args.init) == config.get('init_sha256', gpu_cfg['init_sha256']),
            'Pinned initializer differs')
    profile = read(ROOT / 'reports/evidence/r2d/resource-profile.json')
    require(profile['status'] == 'PASS' and profile['runtime'] == runtime,
            'R2d runtime profile differs')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Official test requires exactly one CUDA GPU')
    # Reconstructed checkpoint run signatures will be verified against the audited
    # history inside the loaded state; existing validated model files remain frozen.
    output = args.output
    output.mkdir(parents=True, exist_ok=args.resume)
    with (output / '.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = {'status': 'INPUTS_VERIFIED', 'dataset': 'MFNet official test',
                'frozen_393': frozen, 'checkpoint_sha256':
                    {k: v['sha256'] for k,v in sorted(checkpoints.items())},
                'seeds': list(SEEDS), 'arms': list(ARMS),
                'selection_rule': 'Fixed Epoch 5 from R2d; no best-epoch selection',
                'test_time_adaptation': False, 'test_time_training': False,
                'notes': 'Official MFNet holdout; NOT cross-dataset testing'}
        plan['runtime'] = runtime
        plan['runner_sha256'] = sha(Path(__file__).resolve())
        plan['metrics_sha256'] = sha(Path(__file__).with_name('metrics.py').resolve())
        if args.resume:
            require(read(output / 'input-contract.json') == plan,
                    'Cannot resume with changed input contract')
        else:
            save(output / 'input-contract.json', plan)
        completed = []
        for seed in SEEDS:
            for arm in ARMS:
                key = f'{arm}/{seed}'
                folder = output / f'seed{seed}'
                folder.mkdir(exist_ok=True)
                final = folder / f'{arm}.json'
                events = folder / f'{arm}.jsonl'
                expected = checkpoints[key]
                if args.resume and final.exists():
                    row = read(final)
                    require(row['seed'] == seed and row['arm'] == arm
                            and row['checkpoint_sha256'] == expected['sha256']
                            and row['per_image_sha256'] == sha(events)
                            and row['sample_count'] == len(names),
                            f'Existing completed result changed: {key}')
                    completed.append(row)
                    continue
                # The previous interrupted attempt is preserved; it cannot be
                # silently replaced or appended to a completed dataset result.
                require(not final.exists() and not events.exists(),
                        f'Incomplete attempt exists for {key}; archive before retry')
                print(f'official test: loading {key}', flush=True)
                started = time.monotonic()
                from experiments.r2b.controls import set_seed
                set_seed(seed)
                with construct_model(args.upstream, args.init) as (model, cfg, _):
                    with fusion(model, arm, config['widths']):
                        # Check SHA before deserializing locally trusted checkpoint.
                        require(sha(expected['checkpoint']) == expected['sha256'],
                                f'Checkpoint changed during run: {key}')
                        state = torch.load(expected['checkpoint'], map_location='cpu', weights_only=False)
                        load_frozen(model, state, expected, gpu_cfg['version'])
                        del state
                        model = model.cuda().eval()
                        before = model_digest(model)
                        require(before == expected['model_sha256'],
                                f'Loaded final tensor hash differs: {key}')
                        hist = np.zeros((9,9), dtype=np.int64)
                        with events.open('x', encoding='utf-8', newline='\n') as stream:
                            with torch.inference_mode():
                                for pos, sample in enumerate(names, 1):
                                    rgb, thermal, label = read_sample(args.root, sample, cfg, False)
                                    logits = model(rgb, thermal)
                                    require(logits.shape == (1,9,480,640)
                                            and torch.isfinite(logits).all().item(),
                                            f'Invalid model output: {key} / {sample}')
                                    pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
                                    gt = label[0].cpu().numpy()
                                    matrix = confusion(gt, pred)
                                    hist += matrix
                                    record = {'id': sample, 'position': pos,
                                              'prediction_sha256': hashlib.sha256(pred.tobytes()).hexdigest(),
                                              'confusion_matrix': matrix.tolist()}
                                    stream.write(json.dumps(record, separators=(',', ':'),allow_nan=False)+'\n')
                                    if pos % 50 == 0 or pos == len(names):
                                        stream.flush()
                                        os.fsync(stream.fileno())
                                        print(f'official test {key}: {pos}/{len(names)}', flush=True)
                        require(hist.sum(axis=1).tolist() == frozen['class_pixel_counts'],
                                f'Official test class pixel counts changed: {key}')
                        require(sha(expected['checkpoint']) == expected['sha256']
                                and model_digest(model) == before and not model.training,
                                f'Model changed during official test: {key}')
                row = {'status': 'COMPLETE', 'seed': seed, 'arm': arm,
                       'checkpoint_sha256': expected['sha256'],
                       'model_sha256': before, 'sample_count': len(names),
                       'per_image_sha256': sha(events), 'metrics': metrics(hist),
                       'wall_seconds': time.monotonic()-started,
                       'test_time_training': False, 'test_time_adaptation': False}
                save(final,row)
                completed.append(row)
                torch.cuda.empty_cache()
        summary = compare_runs(completed)
        summary['data_contract_sha256'] = sha(output/'input-contract.json')
        summary['run_file_sha256'] = {
            f'seed{seed}/{arm}.json': sha(output/f'seed{seed}/{arm}.json')
            for seed in SEEDS for arm in ARMS}
        summary['per_image_file_sha256'] = {
            f'seed{seed}/{arm}.jsonl': sha(output/f'seed{seed}/{arm}.jsonl')
            for seed in SEEDS for arm in ARMS}
        summary['official_test_pixel_reads_per_run'] = len(names)
        summary['cross_dataset_validation'] = False
        save(output/'summary.json',summary)
        print(json.dumps({'status':'COMPLETE','summary':str(output/'summary.json'),
                          'arms':len(completed), 'samples_each':len(names)}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','upstream','init','checkpoint-dir','output'):
        parser.add_argument('--'+name, type=lambda p: Path(p).expanduser().resolve(), required=True)
    parser.add_argument('--resume',action='store_true',help='Skip only fully audited completed arms')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
