"""R5 reviews merged evidence and runs only non-training launch and model probes."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

from . import independent_audit as prior

MERGE = '70456bf60630cdeb593115316c9997021bd067dc'
R4 = 'f800f26ca15b4cafe9b4f38e5cdc031a93f0e5e1'
TREE = '1c352ab618d5249f3eeded6d001260df76d69666'
R3 = 'b532d580a5dc431bc4a4048833d94db433dddd4f'
INTEGRATION = 'aa6f081126fc0da517b5e4a7dc111c5cdbe0a62c'
EVIDENCE = 'reports/evidence/'
R3_DIR = EVIDENCE + 'r3-d1b-r3/'
R4_DIR = EVIDENCE + 'r3-d1b-r4/'
D1_DIR = EVIDENCE + 'r3-d1/review/'
LEDGER = D1_DIR + 'gt-per-image.jsonl'
READ_ONLY_GUARDS = ('external', 'runtime', 'load_sigma', 'DevelopmentData',
                    'Checkpoints', 'run_lock', 'update')


def checked_blob(path):
    raw = prior.git('show', f'{MERGE}:{path}')
    prior.must(raw == (prior.ROOT / path).read_bytes(), f'Checkout differs: {path}')
    return raw


def manifest(directory, filename='verification.json'):
    manifest_path = directory + filename
    raw = checked_blob(manifest_path)
    obj = json.loads(raw)
    entries = obj.get('files_sha256')
    if entries is None:
        entries = {name: record['sha256'] for name, record in obj.items()}
    paths = set(prior.git('ls-tree', '-r', '--name-only', MERGE, directory).decode().splitlines())
    prior.must(paths - {manifest_path, directory + '.gitattributes'} ==
               {directory + name for name in entries}, f'Incomplete manifest: {directory}')
    hashes = {directory + name: prior.sha(checked_blob(directory + name)) for name in entries}
    prior.must(hashes == {directory + name: h for name, h in entries.items()},
               f'Archive digest mismatch: {directory}')
    return obj, {**hashes, manifest_path: prior.sha(raw)}


def cpu_result(log):
    raw = log.read_text(encoding='utf-8')
    summary = re.search(r'^(\d+) passed, (\d+) skipped(?:, \d+ warnings?)? in ', raw, re.M)
    prior.must(summary and prior.exit_code(log) == 0 and ' failed' not in summary[0],
               f'Incomplete CPU evidence: {log}')
    return {'passed': int(summary[1]), 'skipped': int(summary[2]), 'exit_code': 0}


def archived_review(out):
    prior.must(prior.git('show', '-s', '--format=%T %P', MERGE).decode().split() ==
               [TREE, INTEGRATION, R4] and
               prior.git('show', '-s', '--format=%T', R4).decode().strip() == TREE and
               prior.git('merge-base', MERGE, 'HEAD').decode().strip() == MERGE,
               'Wrong merged R4 tree or ancestry')
    hashes = {}
    for directory, file in ((EVIDENCE + 'r3-d0/', 'final-verification.json'),
                            (D1_DIR, 'verification.json'), (prior.OLD, 'verification.json'),
                            (R3_DIR, 'verification.json'), (R4_DIR, 'verification.json')):
        _, found = manifest(directory, file)
        hashes.update(found)

    r4 = json.loads(checked_blob(R4_DIR + 'verification.json'))
    old = json.loads(checked_blob(R3_DIR + 'independent.json'))
    prior.must(prior.git('rev-parse', f'{R3}:{R3_DIR}independent.json') ==
               prior.git('rev-parse', f'{MERGE}:{R3_DIR}independent.json'),
               'Original R3 audit changed after archival')
    prior.must(r4['tested_head'] == INTEGRATION and
               json.loads(checked_blob(R4_DIR + 'tested-head.json')) == {'head': INTEGRATION},
               'R4 CPU tested the wrong tree')
    prior.must(not prior.git('diff', '--name-only', INTEGRATION, MERGE, '--',
                             'tests', 'experiments/r3_d1', 'experiments/r2k_q1v2_llvip',
                             prior.PROTOCOL, prior.SPLIT), 'Tested production tree differs')
    r4_cpu = cpu_result(prior.ROOT / R4_DIR / 'cpu.log')
    prior.must(r4_cpu == {'passed': 207, 'skipped': 3, 'exit_code': 0} and
               prior.exit_code(prior.ROOT / R4_DIR / 'compileall.log') == 0,
               'R4 CPU evidence differs')

    recheck = out / 'r4-inputs'
    recheck.mkdir()
    for name in ('numerics.json', 'numerics.log', 'resume.json', 'resume.log',
                 'data-integrity.json', 'data.log', 'gt-coverage.json', 'gt-per-image.jsonl'):
        path = LEDGER if name == 'gt-per-image.jsonl' else R3_DIR + name
        raw = checked_blob(path)
        prior.must(prior.sha(raw) == r4['retained_artifacts_sha256'][name] ==
                   old['fresh_file_sha256'][name], f'R3/R4 retention mismatch: {name}')
        (recheck / name).write_bytes(raw)
    for name in ('cpu.log', 'compileall.log'):
        (recheck / name).write_bytes(checked_blob(R4_DIR + name))

    independent = prior.audit(recheck)
    stored = json.loads(checked_blob(R4_DIR + 'closure.json'))
    prior.must(all(v == stored[k] for k, v in independent.items()) and
               independent['status'] == 'PASS' and not independent['authorized'],
               'Independent evaluation differs from R4 archive')
    return independent, hashes, r4_cpu


def split_and_fresh_data(out, root):
    split = json.loads(checked_blob(prior.SPLIT))
    d0 = json.loads(checked_blob(EVIDENCE + 'r3-d0/audit.json'))
    roles = {r: set(split[r]) for r in ('fit', 'validation', 'quarantine')}
    fit, validation, quarantine = roles.values()
    development = fit | validation
    union = development | quarantine
    components = {k: set(v) for k, v in split['proxy_components'].items()}
    invalid = {row['id'] for row in d0['invalid_annotations']}
    candidates = d0['cross_split_candidates']
    uncertain = {row['official_train'] for row in candidates if row['verdict'] == 'UNKNOWN'}
    prior.must([len(v) for v in roles.values()] == [9926, 2087, 12] and len(union) == 12025 and
               len(candidates) == len(uncertain) == 7 and len(invalid) == 5 and
               invalid | uncertain == quarantine and
               set().union(*components.values()) == development and
               sum(map(len, components.values())) == len(development) and
               all(v and (v <= fit or v <= validation) for v in components.values()) and
               set(split['validation_components']) == {k for k, v in components.items() if v <= validation} and
               set(split['development_sha256']) == development and
               not split['official_test_used_for_selection'] and
               len(d0['historic_pilot_test_ids']) == 64 and not union.intersection(d0['historic_pilot_test_ids'])
               and d0['official_test_exposure']['historical_thermal_pilot_pairs'] == 64,
               'Frozen development roster differs')
    prior.must(all(len(m) == 3 and set(m) == {f'visible/train/{i}.jpg',
               f'infrared/train/{i}.jpg', f'Annotations/{i}.xml'}
               for i, m in split['development_sha256'].items()), 'Nondevelopment paths in split')
    prior.must(all((root / p).is_dir() for p in ('visible/train', 'infrared/train', 'Annotations')),
               'Missing local development directories')

    current = out / 'development'
    raw = (current / 'gt-per-image.jsonl').read_bytes()
    fresh_raw = (current / 'data-integrity.json').read_bytes()
    coverage_raw = (current / 'gt-coverage.json').read_bytes()
    data = json.loads(fresh_raw)
    prior.must(prior.exit_code(out / 'data.log') == 0 and raw == checked_blob(LEDGER) and
               data['coverage_sha256'] == prior.sha(coverage_raw),
               'Fresh development audit differs from frozen ledger')
    totals = prior.data(data, json.loads(coverage_raw), raw, split)
    fresh_hashes = {
        'data-integrity.json': prior.sha(fresh_raw), 'gt-coverage.json': prior.sha(coverage_raw),
        'gt-per-image.jsonl': prior.sha(raw),
    }
    return split, totals, fresh_hashes


def guard_probes(candidate_file):
    from experiments.r3_d1 import authorization, review_registry, train

    prior.must(authorization.TRUSTED_RECEIPTS == review_registry.TRUSTED_RECEIPTS == frozenset(),
               'Production trust registry is not empty')
    scenarios = ((False, candidate_file, 'explicit authorization flag'),
                 (True, None, 'no reviewed authorization'),
                 (True, candidate_file, 'unreviewed authorization'))
    rejections = []
    patches = {name: lambda *args, **kwargs: prior.must(False, 'Passed authorization boundary')
               for name in READ_ONLY_GUARDS}
    with patch.multiple(train, **patches):
        for arm in 'ABCD':
            for seed in prior.SEEDS:
                for resume in (False, True):
                    for flag, receipt, reason in scenarios:
                        args = SimpleNamespace(arm=arm, seed=seed, resume=resume,
                                               authorization=receipt, authorize_formal_training=flag)
                        try:
                            train.train(args)
                        except ValueError as exc:
                            prior.must(reason in str(exc), f'Unexpected rejection: {exc}')
                            rejections.append([arm, seed, resume, reason])
                        else:
                            raise ValueError(f'Unauthorized train accepted for {arm}/{seed}')
    return rejections


def rejected_cli(candidate_file, out):
    # Nonexistent inputs catch accidental traversal even if the caller omitted mock sentinels.
    argv = [sys.executable, '-m', 'experiments.r3_d1.train',
            '--root', str(out / 'forbidden-data'), '--split', str(out / 'forbidden-split'),
            '--upstream', str(out / 'forbidden-upstream'), '--output', str(out / 'forbidden-output'),
            '--seed', '3407', '--arm', 'D', '--authorize-formal-training',
            '--authorization', str(candidate_file), '--resume']
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    prior.must(result.returncode == 1 and
               'Formal training blocked for all arms: unreviewed authorization' in result.stderr
               and not (out / 'forbidden-output').exists(), 'CLI did not reject unreviewed resume')
    return {'argv': argv[1:], 'exit_code': result.returncode,
            'error': result.stderr.splitlines()[-1]}


def launch_plan(split, upstream, checkpoint_root, archived):
    import torch
    from experiments.r3_d1 import budget, plan, train
    from experiments.r3_d1.model import Detector
    from experiments.r3_d1.preflight import fingerprint
    from experiments.r3_d1.upstream import load_sigma

    prior.must(torch.cuda.is_available() and torch.cuda.device_count() == 1,
               'Exactly one visible CUDA GPU is required')
    train.reproducible(3407, True)
    env = train.runtime(True)
    vm, core, source = load_sigma(upstream)
    numerics = json.loads(checked_blob(R3_DIR + 'numerics.json'))
    prior.must(env == numerics['environment'] and source == numerics['upstream'],
               'Live GPU, environment or pinned import fingerprint differs')
    candidate = json.loads(checked_blob(prior.OLD + 'authorization-candidate.json'))
    ledger = json.loads(checked_blob(prior.OLD + 'run-signatures.json'))
    signed = numerics['source_sha256']
    for path, h in signed.items():
        runtime_bytes = (prior.ROOT / path).read_bytes()
        merged_bytes = prior.git('show', f'{MERGE}:{path}')
        prior.must(prior.sha(runtime_bytes) == h and
                   runtime_bytes.replace(b'\r\n', b'\n') == merged_bytes == prior.blob(path),
                   f'Merged source differs: {path}')
    epoch_hashes = prior.signatures(candidate, ledger, split, env, source, signed)
    run_rows = {}
    all_paths = set()
    for arm in 'ABCD':
        for seed in prior.SEEDS:
            key = f'{arm}/{seed}'
            output = train.external(checkpoint_root) / arm / str(seed)
            prior.must(output not in all_paths and not output.exists(), f'Existing output: {output}')
            all_paths.add(output)
            sig = train.signature(seed, arm, source, env)
            prior.must(sig == ledger['runs'][key]['signature'] and
                       prior.digest(sig) == candidate['run_signatures'][key],
                       f'Actual runner would sign a different run: {key}')
            run_rows[key] = {'signature_sha256': prior.digest(sig), 'output': str(output),
                             'epochs': plan.EPOCHS, 'batch': plan.BATCH,
                             'updates_per_epoch': train.UPDATES}

    # A fresh synthetic forward traces the actual pinned kernel; no gradients or optimizer.
    initial = json.loads(checked_blob(D1_DIR + 'gpu.json'))
    zero = torch.zeros((1, 3, 640, 640), device='cuda')
    counts = {}
    timings = {}
    for arm in 'ABCD':
        model = Detector(arm, seed=3407, widths=plan.WIDTHS, upstream=vm).cuda().eval()
        inputs = {'rgb': zero if arm != 'B' else None, 'thermal': zero if arm != 'A' else None}
        counts[arm] = budget.complete_count(model, inputs, core if arm == 'D' else None)
        counts[arm]['initial_state_sha256'] = fingerprint(model.state_dict())
        prior.must(counts[arm] == initial['counts'][arm], f'Model/operator counts drift: {arm}')
        del model
        backend = 'deterministic' if arm == 'D' else 'native'
        batch4, batch2 = (numerics['detector'][f'{arm}/{size}'][backend] for size in (4, 2))
        hours = 50 * (2481 * batch4['update_ms'] + batch2['update_ms']) / 3_600_000
        prior.must(abs(hours - numerics['five_seed_compute_hours'][arm]) < 1e-9,
                   f'Resource arithmetic drift: {arm}')
        timings[arm] = {'compute_hours': hours, 'batch4': batch4, 'batch2': batch2}

    cnn, mamba = counts['C'], counts['D']
    delta = {k: abs(cnn[k] / mamba[k] - 1) for k in ('parameters', 'mac')}
    prior.must(delta['parameters'] <= .01 and delta['mac'] <= .05, 'Fairness budget exceeded')
    free_gpu, total_gpu = torch.cuda.mem_get_info()
    disk = shutil.disk_usage(checkpoint_root.parent)
    peak_reserved = max(row['batch4']['peak_reserved_bytes'] for row in timings.values())
    prior.must(free_gpu > peak_reserved and disk.free > 3 * 1024**3,
               'Current GPU or checkpoint disk headroom is insufficient')
    prior.must(archived['candidate_sha256'] == prior.CANDIDATE, 'Candidate drift')
    return dict(runtime=env, upstream=source, run_signatures=run_rows, epoch_orders=epoch_hashes,
                merged_source_sha256=signed, model_counts=counts, fair_budget_relative=delta,
                estimated_compute_hours=sum(row['compute_hours'] for row in timings.values()),
                timing_by_arm=timings, gpu_free_bytes=free_gpu, gpu_total_bytes=total_gpu,
                output_disk_free_bytes=disk.free, output_root=str(checkpoint_root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'data', 'output', 'checkpoint-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    out, root = args.output.resolve(), args.checkpoint_root.resolve()
    prior.must(not out.is_relative_to(prior.ROOT) and not root.is_relative_to(prior.ROOT),
               'Audit and planned training directories must be external')
    archived, hashes, r4_cpu = archived_review(out)
    split, totals, fresh_hashes = split_and_fresh_data(out, args.data)
    new_cpu = cpu_result(out / 'cpu.log')
    prior.must(new_cpu == r4_cpu and prior.exit_code(out / 'compileall.log') == 0,
               'CPU or compileall regression in R5')
    candidate_file = prior.ROOT / prior.OLD / 'authorization-candidate.json'
    rejected = guard_probes(candidate_file)
    cli = rejected_cli(candidate_file, out)
    plan = launch_plan(split, args.upstream, root, archived)
    for path in (prior.PROTOCOL, prior.SPLIT):
        hashes[path] = prior.sha(checked_blob(path))
    registry_blob = prior.git('show', f'{MERGE}:{prior.REGISTRY}')
    registry_bytes = (prior.ROOT / prior.REGISTRY).read_bytes().replace(b'\r\n', b'\n')
    prior.must(registry_blob == registry_bytes == prior.blob(prior.REGISTRY), 'Registry drift')
    hashes[prior.REGISTRY] = prior.sha(registry_blob)
    gates = dict.fromkeys(('A', 'B', 'C', 'D', 'data_integrity', 'protocol_signature'), 'PASS')
    result = dict(status='PASS', authorized=False, gates=gates, merged_commit=MERGE, merged_tree=TREE,
                  reran=['cpu', 'compileall', 'development_data_and_fcos', 'synthetic_model_forwards',
                         'live_runtime_fingerprints', 'signatures_and_orders', 'authorization_rejections'],
                  evidence_rechecked_only=['cuda_backward_numerics', 'four_arm_bitwise_resume'],
                  r4_head=R4, candidate_sha256=prior.CANDIDATE, historical_gates=archived,
                  archived_cpu=r4_cpu, fresh_cpu=new_cpu, fresh_development_totals=totals,
                  fresh_data_sha256=fresh_hashes, checked_archive_sha256=hashes,
                  launch_plan=plan, negative_authorization_probes=rejected, cli_rejection=cli,
                  official_test_pixel_reads=0, real_llvip_optimizer_updates=0,
                  formal_training_started=False)
    with (out / 'authorization-review.json').open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    print('PASS: independent technical review; no training authorization issued')


if __name__ == '__main__':
    main()
