"""Shared frozen inputs, runtime, model and optimizer checks."""
import hashlib
from pathlib import Path
import subprocess

import torch

from experiments.r1.assets import sha256
from experiments.r1c.data import canonical_sha, read_lists
from experiments.r1e.train import signatures
from experiments.r1g.partition import EVIDENCE, ROOT, TRAIN, UPDATES, VAL, read, references, require, selected_files, verify

FUSION_PREFIXES = ('backbone.cross_mamba.', 'backbone.channel_attn_mamba.')
ARMS = ('ssm', 'cnn', 'attention')


def load_protocol():
    cfg = read(ROOT / 'configs/R2c_protocol.json')
    r2a = read(ROOT / 'configs/R2a_exp0.json')
    split = read(EVIDENCE / 'split.json')
    require((cfg['version'], tuple(cfg['arms']), cfg['seed'], cfg['epochs'], cfg['primary_epoch'],
             cfg['precision'], cfg['microbatch'], cfg['accumulation'], cfg['tail_size'], cfg['updates_per_epoch'],
             cfg['bootstrap_replicates'], cfg['primary_case'], cfg['primary_rates'])
            == ('r2c-matched-fusion-v1', ARMS, 3407, 5, 5, 'fp32', 1, 8, 6, UPDATES, 10000,
                'primary', ['new', 'net']), 'Frozen R2c protocol differs')
    require(cfg['split_sha256'] == split['split_sha256'] == r2a['split_sha256']
            and cfg['selected_assets_sha256'] == split['selected_assets_sha256']
            and cfg['r2a_config_sha256'] == canonical_sha(r2a)
            and len(split['train_ids']) == TRAIN and len(split['validation_ids']) == VAL, 'Frozen inputs differ')
    return cfg, r2a, split


def gpu_inputs(args, *, validation_only=False):
    cfg, r2a, split = load_protocol()
    graph, prior = references()
    certificate = read(EVIDENCE / 'isolation.json')
    require(certificate['status'] == 'PASS'
            and verify(split, graph, prior, read_lists(args.root)) == certificate['class_support'],
            'R1g isolation audit differs')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expose one CUDA GPU')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime, _ = signatures(read(ROOT / 'configs/R1e_convergence.json'))
    fields = ('upstream_commit', 'init_sha256', 'seed', 'precision', 'microbatch',
              'accumulation', 'lr', 'weight_decay', 'betas')
    require(all(cfg[k] == runtime['config'][k] for k in fields) and sha256(args.init) == cfg['init_sha256'],
            'Runtime optimizer or initialization differs from pinned R1g recipe')
    git = ['git', '-C', str(args.upstream)]
    require(subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip() == cfg['upstream_commit'],
            'Pinned upstream source differs')
    subprocess.run(git + ['diff', '--quiet', 'HEAD', '--'], check=True)
    val = set(split['validation_ids'])
    for row in selected_files(split):
        if not validation_only or Path(row['path']).stem in val:
            require(sha256(args.root / row['path']) == row['sha256'], f'Frozen asset differs: {row["path"]}')
    sources = ('r2c/train.py', 'r2c/protocol.py', 'r2b/controls.py', 'r1g/train.py', 'r1d/state.py', 'r1e/state.py')
    signature = {'config_sha256': canonical_sha(cfg), 'runtime': runtime,
                 'sources': {f'experiments/{name}': sha256(ROOT / 'experiments' / name) for name in sources},
                 'split_sha256': split['split_sha256'], 'selected_assets_sha256': split['selected_assets_sha256'],
                 'graph_sha256': split['graph_sha256'], 'upstream_commit': cfg['upstream_commit'],
                 'init_sha256': cfg['init_sha256']}
    return cfg, r2a, split, graph, signature


def digest_common(model):
    """Include buffers; exclude only the two four-scale fusion module lists."""
    common = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not name.startswith(FUSION_PREFIXES):
            common.update(name.encode() + b'\0')
            common.update(str((list(value.shape), value.dtype)).encode() + b'\0')
            common.update(value.detach().cpu().contiguous().numpy().tobytes())
    return common.hexdigest()


def optimizer_steps(optimizer, scheduler, tensor_count, update, cfg):
    params = [p for group in optimizer.param_groups for p in group['params']]
    require(len(params) == len({id(p) for p in params}) == tensor_count == len(optimizer.state)
            and all(optimizer.state[p]['step'].item() == update for p in params)
            and scheduler.last_epoch == update
            and all(g['lr'] == cfg['lr'] and g['betas'] == tuple(cfg['betas']) for g in optimizer.param_groups),
            'AdamW or constant scheduler coverage/step differs')
