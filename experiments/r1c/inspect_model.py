"""Inspect the real pinned model/optimizer without consuming any dataset pixels."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess

import torch

from experiments.r1.assets import sha256
from experiments.r1c.data import write_json
from experiments.r1c.pilot import construct_model
from experiments.r1c.state import complete_groups


def main():
    parser = argparse.ArgumentParser()
    for name in ('upstream', 'init', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    upstream, init, output = args.upstream.resolve(), args.init.resolve(), args.output.resolve()
    assert sha256(init) == cfg['init_sha256']
    assert subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() == cfg['upstream_commit']
    subprocess.run(['git', '-C', str(upstream), 'diff', '--quiet', 'HEAD', '--'], check=True)
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    with construct_model(upstream, init) as (model, config, keys):
        from utils.init_func import group_weight
        groups, coverage = complete_groups(model, group_weight([], model, torch.nn.BatchNorm2d, cfg['lr']), cfg['lr'], cfg['weight_decay'])
        write_json(output / 'prepilot-coverage.json', coverage)
        versions = {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'timm', 'einops', 'numpy', 'opencv-python', 'Pillow')}
        result = {'scope': 'Actual CPU model construction and transfer initialization; no training/evaluation',
                  'initialization': keys, 'coverage': {k: v for k, v in coverage.items() if k != 'parameters'},
                  'versions': versions, 'gpu': torch.cuda.get_device_name(0), 'cuda_runtime': torch.version.cuda,
                  'upstream_commit': cfg['upstream_commit'], 'upstream_tracked_diff': 'clean'}
        write_json(output / 'prepilot-inspection.json', result)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
