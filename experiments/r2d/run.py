"""Use the accepted R2c driver with frozen widths and per-seed namespaces."""
import argparse
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.r1.assets import sha256
from experiments.r1c.data import canonical_sha
from experiments.r1g.partition import ROOT, read, require
from experiments.r2a.run import append
from experiments.r2c import analysis, audit, evaluate, train
from experiments.r2c.protocol import gpu_inputs
from experiments.r2d.models import fusion
from experiments.r2d.profile import design


def write_window(path, row):
    with path.open('a', encoding='utf-8', newline='\n') as stream:
        append(stream, row)


def seed_inputs(inputs, seed, design_config, profile_path):
    base, perturbations, split, graph, prior_signature = inputs
    cfg = {**base, 'seed': seed}
    sources = {name: sha256(ROOT / name) for name in (
        'experiments/r2d/run.py', 'experiments/r2d/models.py',
        'experiments/r2c/audit.py', 'experiments/r2c/evaluate.py')}
    signature = {**prior_signature, 'config_sha256': canonical_sha(cfg),
                 'sources': {**prior_signature['sources'], **sources},
                 'r2d_protocol_sha256': canonical_sha(design_config),
                 'resource_profile_sha256': sha256(profile_path), 'seed': seed}
    return cfg, perturbations, split, graph, signature


def run(args):
    import fcntl

    config, _ = design()
    path = args.evidence / 'resource-profile.json'
    measured = read(path)
    require(measured['status'] == 'PASS' and measured['config_sha256'] == canonical_sha(config)
            and measured['model_source_sha256'] == sha256(ROOT / 'experiments/r2d/models.py'),
            'Measured parameter and resource audit required')
    require(set(args.seeds) <= set(config['seeds']) and len(set(args.seeds)) == len(args.seeds),
            'Requested seeds differ from frozen seed set')
    require(not args.checkpoint_dir.is_relative_to(ROOT) and args.checkpoint_dir.name == 'sigma-r2d-runs',
            'Use dedicated external sigma-r2d-runs storage')
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (args.checkpoint_dir / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inputs = gpu_inputs(args)
        require(inputs[-1] == measured['runtime'], 'Measured and training runtime differ')
        widened = partial(fusion, widths=config['widths'])
        for seed in args.seeds:
            cfg, r2a, split, graph, signature = seed_inputs(inputs, seed, config, path)
            seed_args = SimpleNamespace(**vars(args))
            seed_args.evidence = args.evidence / f'seed{seed}'
            seed_args.checkpoint_dir = args.checkpoint_dir / f'seed{seed}'
            seed_args.checkpoint_dir.mkdir(exist_ok=True)
            data = (cfg, r2a, split, graph, signature)
            print(f'R2d seed {seed} started', flush=True)
            with (patch.object(train, 'gpu_inputs', return_value=data), patch.object(train, 'fusion', widened),
                  patch.object(train, 'emit', write_window), patch.object(audit, 'load_protocol',
                  return_value=(cfg, r2a, split))):
                seed_args.phase = 'first'
                if not (seed_args.evidence / 'audit-epoch1.json').exists():
                    train.run(seed_args)
                    audit.audit(seed_args.evidence, seed_args.checkpoint_dir, 1)
                seed_args.phase = 'finish'
                if not (seed_args.evidence / 'audit-epoch5.json').exists():
                    train.run(seed_args)
                    audit.audit(seed_args.evidence, seed_args.checkpoint_dir, 5)
            require(read(seed_args.evidence / 'audit-epoch5.json')['signature'] == signature, 'Seed audit differs')
            with patch.object(evaluate, 'gpu_inputs', return_value=data), patch.object(evaluate, 'fusion', widened):
                if not (seed_args.evidence / 'perturbation').exists():
                    evaluate.run(seed_args)
            with patch.object(analysis, 'load_protocol', return_value=(cfg, r2a, split)):
                analysis.analyze(seed_args.evidence)
            print(f'R2d seed {seed} complete', flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ('upstream', 'init', 'root', 'evidence', 'checkpoint-dir'):
        parser.add_argument('--' + name, type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
