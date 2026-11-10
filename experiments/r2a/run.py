"""Gated clean reproduction, paired inference and fusion output replacement."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from experiments.exp0.perturbations import corrupt_thermal
from experiments.r1.assets import sha256
from experiments.r1c.data import canonical_sha, read_lists, write_json
from experiments.r1c.pilot import construct_model, read_sample
from experiments.r1e.train import signatures
from experiments.r1g.partition import ROOT, read, references, require, selected_files, verify
from experiments.r2a.hooks import modules, observe, replace
from experiments.r2a.metrics import confusion, digest, distance_bands, mask_record, pair, scores

R1G = ROOT / 'reports/evidence/r1g'


def append(stream, row):
    stream.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')


def records(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def cases(cfg):
    yield {'id': 'identity', 'kind': 'identity', 'position': 'center'}
    yield {'id': 'primary', **cfg['primary']}
    for pos in sorted(cfg['positions_y0_x0_y1_x1']):
        for seed in cfg['seeds']:
            yield {'id': f'gaussian.{pos}.{seed}', 'kind': 'gaussian', 'position': pos,
                   'seed': seed, 'sigma': cfg['gaussian_sigma']}
        yield {'id': f'shift.{pos}', 'kind': 'shift_local', 'position': pos, **cfg['shift']}


def masks(cfg):
    result = {'identity': np.zeros(cfg['image_shape'], dtype=bool)}
    for name, (y0, x0, y1, x1) in cfg['positions_y0_x0_y1_x1'].items():
        mask = np.zeros(cfg['image_shape'], dtype=bool)
        mask[y0:y1, x0:x1] = True
        require(mask.sum() == 4096, 'Unexpected ROI dimensions')
        result[name] = mask
    return result


def perturbed(raw, spec, sample, cfg):
    keys = ('fill', 'sigma', 'dx', 'dy')
    options = {key: spec[key] for key in keys if key in spec}
    seed = None
    if 'seed' in spec:
        pos = sorted(cfg['positions_y0_x0_y1_x1']).index(spec['position'])
        seed = int(np.random.SeedSequence([spec['seed'], int(sample[:-1]), int(sample[-1] == 'N'), pos])
                   .generate_state(1)[0])
        options['seed'] = seed
    result, mask = corrupt_thermal(raw, spec['kind'], tuple(cfg['positions_y0_x0_y1_x1'][spec['position']]), **options)
    require(np.array_equal(result[~mask], raw[~mask]), 'Thermal changed outside intended support')
    return result, mask, seed


def normalized_thermal(array, cfg):
    from utils.transforms import normalize

    replicated = np.repeat(array[..., None], 3, axis=-1)
    require(np.array_equal(replicated[..., 0], replicated[..., 1])
            and np.array_equal(replicated[..., 0], replicated[..., 2]), 'Invalid pre-normalization channels')
    normalized = normalize(replicated, cfg.norm_mean, cfg.norm_std).transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(normalized)).float().unsqueeze(0).cuda()


def predict(model, rgb, thermal):
    require(not model.training and rgb.shape == thermal.shape == (1, 3, 480, 640)
            and rgb.dtype == thermal.dtype == torch.float32, 'Paired forward protocol changed')
    logits = model(rgb, thermal)
    require(logits.shape == (1, 9, 480, 640) and torch.isfinite(logits).all().item(), 'Invalid logits')
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def model_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def clean_gate(model, cfg, root, split, assignment, signature, output):
    archive = records(R1G / 'gpu/images-epoch5.jsonl')
    require([x['id'] for x in archive] == split['validation_ids'], 'R1g clean archive differs')
    matrices = []
    path = output / 'clean.jsonl'
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        for original in archive:
            sample = original['id']
            rgb, thermal, label = read_sample(root, sample, cfg, False)
            pred = predict(model, rgb, thermal)
            matrix = confusion(label[0].cpu().numpy(), pred)
            require(assignment[sample] == original['group'] and matrix.tolist() == original['confusion_matrix'],
                    f'Clean gate differs from the R1g per-image confusion matrix: {sample}')
            matrices.append(matrix)
            append(stream, {'id': sample, 'group': assignment[sample], 'prediction_sha256': digest(pred),
                            'confusion_matrix': matrix.tolist(), **scores(matrix)})
    values = scores(np.sum(matrices, axis=0))
    require(abs(values['miou'] - .611498) <= 5e-7, 'Clean gate did not reproduce 61.1498%')
    return {'status': 'PASS', 'phase': 'clean', 'signature': signature, 'images': len(archive),
            'groups': len(set(assignment.values())), 'per_image_confusion_exact': True,
            'records_sha256': sha256(path), **values}


def paired_phase(model, upstream_cfg, root, split, assignment, cfg, phase, output, evidence, signature):
    import cv2

    clean_cert = read(evidence / 'clean/certificate.json')
    clean_path = evidence / 'clean/clean.jsonl'
    require(clean_cert['status'] == 'PASS' and clean_cert['phase'] == 'clean'
            and clean_cert['signature'] == signature and sha256(clean_path) == clean_cert['records_sha256'],
            'Missing matching clean gate')
    clean_refs = {r['id']: r for r in records(clean_path)}
    pilot = cfg['pilot_ids']
    require(len(pilot) == len(set(pilot)) == 12 and set(pilot) <= set(split['validation_ids'])
            and len({assignment[i] for i in pilot}) == 12, 'Pilot IDs changed')
    if phase == 'full':
        cert = read(evidence / 'pilot/certificate.json')
        require(cert['status'] == 'PASS' and cert['phase'] == 'pilot' and cert['signature'] == signature
                and cert['ids'] == pilot and cert['identity_replacement_passes'] == 96
                and cert['plain_forward_rechecks'] == 12, 'Missing matching engineering pilot gate')
    ids = pilot if phase == 'pilot' else split['validation_ids']
    all_masks = masks(cfg)
    bands = {k: distance_bands(m, cfg['distance_upper_bounds_px']) for k, m in all_masks.items()}
    tensor_masks = {k: torch.from_numpy(m).cuda().float()[None, None] for k, m in all_masks.items()}
    specs = list(cases(cfg))
    module_map = modules(model)
    write_json(output / 'support.json', {k: mask_record(v) for k, v in all_masks.items()})
    write_json(output / 'cases.json', specs)
    identity_replacements, plain_rechecks, full_predictions = 0, 0, 0
    with ExitStack() as stack:
        pair_logs = {name: stack.enter_context((output / f'pairs-{name}.jsonl').open('x', encoding='utf-8', newline='\n'))
                     for name in [s['id'] for s in specs] + [f'replace.{m}' for m in module_map]}
        response_logs = {s['id']: stack.enter_context((output / f'response-{s["id"]}.jsonl')
                                                      .open('x', encoding='utf-8', newline='\n')) for s in specs}
        for index, sample in enumerate(ids, 1):
            rgb, thermal, label = read_sample(root, sample, upstream_cfg, False)
            rgb_original, label_original = rgb.clone(), label.clone()
            gt = label[0].cpu().numpy()
            raw = cv2.imread(str(root / 'Modal' / (sample + '.png')), cv2.IMREAD_GRAYSCALE)
            require(raw.shape == tuple(cfg['image_shape']) and raw.dtype == np.uint8
                    and torch.equal(normalized_thermal(raw, upstream_cfg), thermal),
                    'Raw thermal does not reproduce clean preprocessing')
            with observe(module_map, sample) as captured:
                clean = predict(model, rgb, thermal)
            ref = clean_refs[sample]
            require(digest(clean) == ref['prediction_sha256'], f'Hooked clean changed: {sample}')
            clean_scores = {k: ref[k] for k in ('miou', 'foreground_miou', 'iou_0_to_8')}
            metadata = {'id': sample, 'group': assignment[sample], 'clean_prediction_sha256': digest(clean)}
            for spec in specs:
                raw_out, mask, seed = perturbed(raw, spec, sample, cfg)
                key = 'identity' if spec['kind'] == 'identity' else spec['position']
                require(np.array_equal(mask, all_masks[key]), 'Perturbation support changed')
                changed = normalized_thermal(raw_out, upstream_cfg)
                inverse = ~tensor_masks[key][0, 0].bool()
                require(torch.equal(changed[:, :, inverse], thermal[:, :, inverse]), 'Normalized thermal changed outside ROI')
                with observe(module_map, sample, reference=captured, mask=tensor_masks[key]) as response:
                    pred = predict(model, rgb, changed)
                require(len(response) == 28, 'Incomplete fusion records')
                if spec['kind'] == 'identity':
                    require(np.array_equal(clean, pred), f'Identity prediction changed: {sample}')
                if spec['id'] == 'primary':
                    primary_x, primary_pred = changed, pred
                append(pair_logs[spec['id']], {**metadata, 'case': spec['id'], 'support': key,
                       'effective_seed': seed, 'changed_input_pixels': int(np.count_nonzero(raw_out != raw)),
                       'prediction_sha256': digest(pred), **pair(clean, pred, gt, mask, bands[key], clean_scores)})
                append(response_logs[spec['id']], {**metadata, 'case': spec['id'], 'response': response})
                full_predictions += 1
            for name, module in module_map.items():
                if phase == 'pilot':
                    with replace(module, name, captured, sample):
                        same = predict(model, rgb, thermal)
                    require(np.array_equal(clean, same), f'Identity replacement changed prediction: {name}/{sample}')
                    identity_replacements += 1
                with replace(module, name, captured, sample):
                    pred = predict(model, rgb, primary_x)
                case = f'replace.{name}'
                append(pair_logs[case], {**metadata, 'case': case, 'support': 'center',
                       'prediction_sha256': digest(pred),
                       **pair(clean, pred, gt, all_masks['center'], bands['center'], clean_scores)})
                full_predictions += 1
            if phase == 'pilot':
                require(np.array_equal(primary_pred, predict(model, rgb, primary_x))
                        and np.array_equal(clean, predict(model, rgb, thermal)), 'Post-hook plain forward changed')
                plain_rechecks += 1
            require(torch.equal(rgb, rgb_original) and torch.equal(label, label_original), 'RGB or GT changed in pair')
            del captured, primary_x
            if index % 10 == 0 or index == len(ids):
                print(f'{phase}: {index}/{len(ids)} samples', flush=True)
    return {'status': 'PASS', 'phase': phase, 'signature': signature, 'ids': ids,
            'identity_inputs_passed': len(ids), 'identity_replacement_passes': identity_replacements,
            'plain_forward_rechecks': plain_rechecks, 'paired_predictions': full_predictions}


def run(args):
    cfg = read(ROOT / 'configs/R2a_exp0.json')
    require(cfg['version'] == 'r2a-exp0-v1' and cfg['image_shape'] == [480, 640]
            and cfg['seeds'] == [3407, 3408, 3409], 'Frozen experiment configuration differs')
    graph, prior = references()
    split = read(R1G / 'split.json')
    cert = read(R1G / 'isolation.json')
    require(split['split_sha256'] == cfg['split_sha256'] == cert['split_sha256']
            and verify(split, graph, prior, read_lists(args.root)) == cert['class_support'],
            'R1g visual split differs')
    val_files = set(split['validation_ids'])
    for entry in selected_files(split):
        if Path(entry['path']).stem in val_files:
            require(sha256(args.root / entry['path']) == entry['sha256'], f'Validation file changed: {entry["path"]}')
    assignment = {i: group[0] for group in graph['groups'] for i in group if i in val_files}
    require(len(assignment) == 157 and len(set(assignment.values())) == 54, 'Unexpected validation membership')
    require(sha256(args.checkpoint) == cfg['checkpoint_sha256'], 'Requested checkpoint digest differs')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expose one CUDA GPU')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime, _ = signatures(read(ROOT / 'configs/R1e_convergence.json'))
    require(sha256(args.init) == runtime['config']['init_sha256'], 'VMamba initializer digest differs')
    git = ['git', '-C', str(args.upstream)]
    require(subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip()
            == runtime['config']['upstream_commit'], 'Upstream source differs')
    subprocess.run(git + ['diff', '--quiet', 'HEAD', '--'], check=True)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    archived = read(R1G / 'gpu/epochs.json')
    require(state['format'] == 'r1g-expanded-scene-v1' and state['history'] == archived
            and state['sampler']['epoch'] == 5 and state['sampler']['cursor'] == 1254
            and state['signature']['config']['split_sha256'] == cfg['split_sha256']
            and state['signature']['runtime'] == runtime, 'R1g checkpoint protocol differs')
    signature = {'config_sha256': canonical_sha(cfg), 'checkpoint_sha256': cfg['checkpoint_sha256'],
                 'split_sha256': cfg['split_sha256'], 'asset_sha256': split['selected_assets_sha256'],
                 'graph_sha256': split['graph_sha256'], 'runtime': runtime,
                 'sources': {p: sha256(ROOT / p) for p in ('experiments/r2a/run.py', 'experiments/r2a/hooks.py',
                             'experiments/r2a/metrics.py', 'experiments/exp0/perturbations.py')}}
    output = args.output / args.phase
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    with construct_model(args.upstream, args.init) as (model, upstream_cfg, _):
        keys = model.load_state_dict(state['model'], strict=True)
        del state
        model = model.cuda().eval()
        before = model_digest(model)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            if args.phase == 'clean':
                result = clean_gate(model, upstream_cfg, args.root, split, assignment, signature, output)
            else:
                result = paired_phase(model, upstream_cfg, args.root, split, assignment, cfg, args.phase,
                                      output, args.output, signature)
        require(model_digest(model) == before and not model.training, 'Model state or mode changed')
        result.update(seconds=time.perf_counter() - start, strict_model_keys=str(keys), model_state_sha256=before,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), test_pixels_read=0)
    require(sha256(args.checkpoint) == cfg['checkpoint_sha256'], 'Checkpoint file changed')
    write_json(output / 'certificate.json', result)
    print({k: v for k, v in result.items() if k not in ('signature', 'ids')}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('clean', 'pilot', 'full'))
    for name in ('upstream', 'init', 'root', 'checkpoint', 'output'):
        parser.add_argument('--' + name, type=lambda p: Path(p).resolve(), required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
