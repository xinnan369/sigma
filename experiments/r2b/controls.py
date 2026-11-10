"""Interface-matched fusion controls and bounded, one-update resource measurements."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments.r1.assets import sha256
from experiments.r1c.data import read_lists, write_json
from experiments.r1c.pilot import construct_model, read_sample
from experiments.r1c.state import complete_groups
from experiments.r1d.state import EpochSampler
from experiments.r1d.train import train_window
from experiments.r1g.partition import EVIDENCE, ROOT, read, references, require, selected_files, verify
from experiments.r2a.run import model_digest

WIDTHS = (96, 192, 384, 768)
GRIDS = ((120, 160), (60, 80), (30, 40), (15, 20))


class CNN(nn.Module):
    def __init__(self, channels, *, cross):
        super().__init__()
        self.cross = cross
        self.mix = nn.Sequential(nn.Conv2d(2 * channels, channels, 1),
                                 nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                                 nn.GELU(), nn.Conv2d(channels, (2 if cross else 1) * channels, 1))

    def forward(self, rgb, thermal):
        y = self.mix(torch.cat((rgb, thermal), dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if self.cross:
            a, b = y.chunk(2, dim=-1)
            return rgb + a, thermal + b
        return rgb + thermal + y

    def macs(self, h, w, c):
        return h * w * (2 * c * c + 9 * c + (2 if self.cross else 1) * c * c)


class Attention(nn.Module):
    """Full-grid queries with 8x8 pooled keys/values per source modality."""
    def __init__(self, channels, *, cross):
        super().__init__()
        self.cross = cross
        inner = channels // 2
        if inner % 4:
            raise ValueError('Attention requires four equally sized heads')
        self.query = nn.Linear(channels, inner)
        self.key_value = nn.Linear(channels, 2 * inner)
        self.output = nn.Linear(inner, channels)

    def attend(self, query, sources):
        batch, height, width, channels = query.shape
        pooled = torch.cat([F.adaptive_avg_pool2d(s.permute(0, 3, 1, 2), (8, 8))
                            .flatten(2).transpose(1, 2) for s in sources], dim=1)
        q = self.query(query).reshape(batch, height * width, 4, -1).transpose(1, 2)
        k, v = (x.reshape(batch, pooled.shape[1], 4, -1).transpose(1, 2)
                for x in self.key_value(pooled).chunk(2, dim=-1))
        y = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(batch, height, width, -1)
        return self.output(y)

    def forward(self, rgb, thermal):
        if self.cross:
            return rgb + self.attend(rgb, (thermal,)), thermal + self.attend(thermal, (rgb,))
        joint = rgb + thermal
        return joint + self.attend(joint, (rgb, thermal))

    def macs(self, h, w, c):
        directions = 2 if self.cross else 1
        tokens = 64 if self.cross else 128
        inner = c // 2
        return directions * (2 * h * w * c * inner + 2 * tokens * c * inner + 2 * h * w * tokens * inner)


@contextmanager
def fusion(model, family):
    if family == 'ssm':
        yield
        return
    cls = {'cnn': CNN, 'attention': Attention}[family]
    backbone = model.backbone
    prior = backbone.cross_mamba, backbone.channel_attn_mamba
    backbone.cross_mamba = nn.ModuleList(cls(c, cross=True) for c in WIDTHS)
    backbone.channel_attn_mamba = nn.ModuleList(cls(c, cross=False) for c in WIDTHS)
    try:
        yield
    finally:
        backbone.cross_mamba, backbone.channel_attn_mamba = prior


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ssm_cost(blocks):
    macs, state_cells = 0, 0
    for block_type, group in zip(('cross', 'concat'), blocks, strict=True):
        for module, (height, width), channels in zip(group, GRIDS, WIDTHS, strict=True):
            op = module.op
            tokens = height * width
            inner, rank, n = op.d_inner, op.dt_rank, op.d_state
            scan_tokens = tokens * (2 if block_type == 'cross' else 4)
            macs += tokens * (4 * channels * inner + 2 * op.d_conv**2 * inner)
            macs += scan_tokens * inner * (2 * rank + 2 * n)
            state_cells += scan_tokens * inner * n
            if block_type == 'concat':
                macs += 4 * inner * (inner // 16)
    return macs, state_cells


def run(args):
    cfg = read(ROOT / 'configs/R2b_protocol.json')
    protocol = read(ROOT / 'configs/R1d_epoch.json')
    r2a = read(ROOT / 'configs/R2a_exp0.json')
    split = read(EVIDENCE / 'split.json')
    graph, prior = references()
    require(split['split_sha256'] == cfg['split_sha256']
            and verify(split, graph, prior, read_lists(args.root)) == read(EVIDENCE / 'isolation.json')['class_support'],
            'R1g split differs')
    git = ['git', '-C', str(args.upstream)]
    require(subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip() == cfg['upstream_commit']
            and sha256(args.init) == protocol['init_sha256']
            and sha256(args.checkpoint) == cfg['checkpoint_sha256'], 'Frozen assets differ')
    subprocess.run(git + ['diff', '--quiet', 'HEAD', '--'], check=True)
    require(torch.cuda.device_count() == 1, 'Expose one CUDA device')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed = cfg['seed']
    sampler = EpochSampler(len(split['train_ids']), seed)
    indices = sampler.next_window(8)
    ids = [split['train_ids'][i] for i in indices]
    sample = r2a['pilot_ids'][0]
    for asset in selected_files(split):
        if Path(asset['path']).stem in [*ids, sample]:
            require(sha256(args.root / asset['path']) == asset['sha256'], 'Resource pilot asset changed')
    args.evidence.mkdir(exist_ok=True, parents=True)
    rows = []
    for family in ('ssm', 'cnn', 'attention'):
        set_seed(seed)
        with construct_model(args.upstream, args.init) as (model, upstream_cfg, _):
            from utils.init_func import group_weight

            original_init = model_digest(model)
            with fusion(model, family):
                model.cuda().train()
                groups, coverage = complete_groups(
                    model, group_weight([], model, nn.BatchNorm2d, protocol['lr']),
                    protocol['lr'], protocol['weight_decay'])
                optimizer = torch.optim.AdamW(groups, lr=protocol['lr'], betas=tuple(protocol['betas']))
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
                torch.cuda.reset_peak_memory_stats()
                original_fusion = [model_digest(x) for x in
                                   (model.backbone.cross_mamba, model.backbone.channel_attn_mamba)]
                fusion_modules = [model.backbone.cross_mamba, model.backbone.channel_attn_mamba]
                params = sum(p.numel() for block in fusion_modules for p in block.parameters())
                if family == 'ssm':
                    macs, cells = ssm_cost(fusion_modules)
                else:
                    macs = sum(module.macs(h, w, c) for blocks in fusion_modules
                               for module, (h, w), c in zip(blocks, GRIDS, WIDTHS, strict=True))
                    cells = 0
                set_seed(seed)
                window = train_window(model, optimizer, scheduler, ids, args.root, upstream_cfg)
                require(len(optimizer.state) == coverage['tensor_count'] == window['gradient_tensor_count']
                        and all(model_digest(x) != prior for x, prior in zip(fusion_modules, original_fusion)),
                        'Incomplete resource-pilot gradient or update')
            rows.append({'family': family, 'scope': 'one training update; no validation score',
                         'seed': seed, 'common_initial_sigma_sha256': original_init,
                         'total_params': coverage['total_numel'], 'fusion_params': params,
                         'fusion_linear_conv_attention_macs': macs, 'fusion_ssm_scalar_state_updates': cells,
                         **window})
            print(f'{family}: {window["window_seconds"]:.3f}s, peak {window["peak_allocated_bytes"]} bytes', flush=True)
            del optimizer, scheduler, groups, model, fusion_modules
        torch.cuda.empty_cache()
    require(len({r['common_initial_sigma_sha256'] for r in rows}) == 1, 'Common initialization differed')
    write_json(args.evidence / 'control-trainability.json', rows)
    frozen = []
    set_seed(seed)
    with construct_model(args.upstream, args.init) as (model, upstream_cfg, _), torch.inference_mode():
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        frozen_hash = model_digest(model)
        rgb, thermal, _ = read_sample(args.root, sample, upstream_cfg, False)
        for family in ('cnn', 'attention'):
            with fusion(model, family):
                model.cuda().eval()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                y = model(rgb, thermal)
                require(y.shape == (1, 9, 480, 640) and torch.isfinite(y).all().item(),
                        'Frozen substitution produced invalid logits')
                frozen.append({'family': family, 'sample': sample, 'shape': list(y.shape),
                               'seconds': time.perf_counter() - start,
                               'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                               'scope': 'one engineering forward; untrained fusion; no performance score'})
        require(model_digest(model) == frozen_hash and sha256(args.checkpoint) == cfg['checkpoint_sha256'],
                'Frozen weights changed')
    write_json(args.evidence / 'control-frozen-interface.json', frozen)
    print(json.dumps({'status': 'PASS', 'resource_pilots': len(rows), 'frozen_substitution_forwards': len(frozen)}))


def main():
    parser = argparse.ArgumentParser()
    for name in ('upstream', 'init', 'root', 'checkpoint', 'evidence'):
        parser.add_argument('--' + name, required=True, type=lambda p: Path(p).resolve())
    run(parser.parse_args())


if __name__ == '__main__':
    main()
