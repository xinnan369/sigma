import argparse
from pathlib import Path
import statistics
import time

import torch

from experiments.r1.assets import sha256
from experiments.r1c.data import write_json
from experiments.r1c.pilot import construct_model, read_sample
from experiments.r1c.state import complete_groups
from experiments.r1d.state import EpochSampler, load, save
from experiments.r1d.train import train_window
from experiments.r1g.partition import ROOT, read, require
from experiments.r1g.train import write_rows
from experiments.r2a.run import model_digest
from experiments.r2b.controls import GRIDS, WIDTHS, set_seed
from experiments.r2c.protocol import gpu_inputs
from experiments.r2i.analysis import summarize
from experiments.r2i.augment import train_occlusion
from experiments.r2i.evaluate import evaluate
from experiments.r2i.gate import attach
from unittest.mock import patch

CONFIG = ROOT / 'configs/R2i_protocol.json'


def resources(model, arm):
    image = torch.zeros((1, 3, 480, 640), device='cuda')
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        times = []
        for i in range(12):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model(image, image)
            end.record()
            end.synchronize()
            if i >= 2:
                times.append(start.elapsed_time(end))
    tokens = sum(h*w for h,w in GRIDS)
    # Abs, square, adjacent-channel product, three reductions, two branch scales, linear
    # score and two scalar additions. tanh counted separately.
    extra = sum(h*w*(8*c+3) for (h,w),c in zip(GRIDS, WIDTHS, strict=True))
    return {'parameters': sum(p.numel() for p in model.parameters()),
            'gate_scalar_flops': extra if arm != 'baseline' else 0,
            'gate_tanh_calls': tokens if arm != 'baseline' else 0,
            'inference_median_ms': statistics.median(times),
            'inference_peak_bytes': torch.cuda.max_memory_allocated()}


def run(args):
    import selective_scan_cuda_core

    cfg = read(CONFIG)
    require(cfg['seed'] == 3407 and cfg['windows'] == 3 and cfg['accumulation'] == 8, 'Pilot budget changed')
    require(not args.output.is_relative_to(ROOT), 'Run directory must be external')
    args.output.mkdir(parents=True, exist_ok=False)
    base, r2a, split, graph, inherited = gpu_inputs(args)
    signature = {'protocol': sha256(CONFIG), 'source': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
                 'assets': inherited, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                 'gpu': torch.cuda.get_device_name(), 'extension_sha256': sha256(Path(selective_scan_cuda_core.__file__))}
    write_json(args.output / 'signature.json', signature)
    result = {}
    for track in cfg['tracks']:
        natural, initial_common, schedule, identity = None, None, None, None
        result[track] = {}
        for arm in cfg['arms']:
            set_seed(cfg['seed'])
            folder = args.output / f'{track}-{arm}'
            folder.mkdir()
            with construct_model(args.upstream, args.init) as (model, config, _):
                from utils.init_func import group_weight

                common = model_digest(model)
                require(initial_common is None or common == initial_common, 'Common initialization differs')
                initial_common = common
                with attach(model, arm):
                    # The zero-initialized head constructor consumes RNG even though
                    # its random weights are discarded. Align upstream train RNGs.
                    set_seed(cfg['seed'])
                    model.cuda().eval()
                    probe = torch.Generator(device='cuda').manual_seed(2026)
                    x, y = (torch.randn((1,3,480,640), generator=probe, device='cuda') for _ in range(2))
                    with torch.inference_mode():
                        logits = model(x, y)
                        require(torch.isfinite(logits).all().item(), 'Nonfinite identity probe')
                        now = logits.cpu()
                    require(identity is None or torch.equal(now, identity), 'Initialized logits changed')
                    identity = now
                    resource = resources(model, arm)
                    model.train()
                    lr = base['lr']
                    groups, coverage = complete_groups(model, group_weight([], model, torch.nn.BatchNorm2d, lr),
                                                       lr, base['weight_decay'])
                    optimizer = torch.optim.AdamW(groups, lr=lr, betas=tuple(base['betas']))
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
                    sampler = EpochSampler(len(split['train_ids']), cfg['seed'])
                    events, aug = [], []

                    def load_train(root, name, config, training):
                        rgb, thermal, label = read_sample(root, name, config, training)
                        thermal, roi = train_occlusion(thermal, config.norm_mean, config.norm_std,
                                                       cfg['seed'], name,
                                                       cfg['train_probability'] if track == 'augmented' else 0.,
                                                       cfg['train_roi'])
                        aug.append([name, roi])
                        return rgb, thermal, label

                    torch.cuda.reset_peak_memory_stats()
                    state_path = folder / 'local.pt'
                    state_sig = dict(signature, track=track, arm=arm)
                    with patch('experiments.r1d.train.read_sample', side_effect=load_train):
                        for step in range(cfg['windows']):
                            names = [split['train_ids'][i] for i in sampler.next_window(cfg['accumulation'])]
                            events.append(train_window(model, optimizer, scheduler, names, args.root, config))
                            if step == 0:
                                model_hash = model_digest(model)
                                next_indices = sampler.order[sampler.cursor:sampler.cursor+8]
                                rng_hash = tuple(t.cpu().numpy().tobytes() for t in
                                                 [torch.get_rng_state(), *torch.cuda.get_rng_state_all()])
                                save(state_path, model, optimizer, scheduler, sampler, {'updates': 1}, state_sig)
                                set_seed(1)
                                sampler.next_window(8)
                                progress, _ = load(state_path, model, optimizer, scheduler, sampler, state_sig)
                                require(progress['updates'] == 1 and model_digest(model) == model_hash
                                        and sampler.order[sampler.cursor:sampler.cursor+8] == next_indices
                                        and rng_hash == tuple(t.cpu().numpy().tobytes() for t in
                                                              [torch.get_rng_state(), *torch.cuda.get_rng_state_all()])
                                        and len(optimizer.state) == coverage['tensor_count'],
                                        'Resume boundary differs')
                    require(schedule is None or schedule == aug, 'Training order or augmentation differs')
                    schedule = aug
                    require(coverage['tensor_count'] == len(optimizer.state)
                            and all(s['step'].item() == cfg['windows'] for s in optimizer.state.values()),
                            'Incomplete optimizer updates')
                    save(state_path, model, optimizer, scheduler, sampler, {'updates': cfg['windows']}, state_sig)
                    write_json(folder / 'training.json', {'events': events, 'augmentation': aug, 'resume': 'PASS',
                               'coverage': coverage['tensor_count'], 'common_initial_sha256': common,
                               'checkpoint_sha256': sha256(state_path)})
                    start = time.perf_counter()
                    rows, refs = evaluate(model, args.root, config, cfg, split, graph, r2a, natural)
                    if natural is None:
                        natural = refs
                    write_rows(folder / 'validation.jsonl', rows)
                    result[track][arm] = {**resource, 'updates': len(events), 'training_seconds':
                        sum(e['window_seconds'] for e in events), 'training_peak_bytes':
                        max(e['peak_allocated_bytes'] for e in events), 'validation_seconds': time.perf_counter()-start,
                        'metrics': summarize(rows, cfg)}
                    write_json(args.output / 'progress.json', result)
                    print(track, arm, 'PASS', result[track][arm]['training_seconds'], flush=True)
            del model, optimizer, groups
            torch.cuda.empty_cache()
        require(all(result[track]['gated'][key] == result[track]['control'][key]
                    for key in ('parameters', 'gate_scalar_flops', 'gate_tanh_calls')), 'Control match failed')
    write_json(args.output / 'summary.json', {'status': 'PASS', 'tracks': result, 'signature': signature,
               'test_pixels_read': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('upstream', 'init', 'root', 'output'):
        parser.add_argument('--' + name, required=True, type=lambda p: Path(p).resolve())
    run(parser.parse_args())
