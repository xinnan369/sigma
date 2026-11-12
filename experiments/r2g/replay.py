"""Clean-context factorial replay using the accepted R2f CUDA state recovery."""
from collections import Counter
import math

import torch

from experiments.r1g.partition import require
from experiments.r2a.metrics import confusion, digest, pair, scores
from experiments.r2b.state import Scan
from experiments.r2e.paths import geometry, intercept
from experiments.r2e.run import predict
from experiments.r2f.replay import validated_factor


def scopes(arm):
    result = {'encoder': ['encoder'], 'decoder': ['decoder'],
              'shared': ['encoder', 'decoder']}
    if arm == 'ssm':
        fused = [f'{name}.{i}' for name in ('cross', 'concat') for i in range(4)]
        result.update({key: [key] for key in fused})
        result['fusion'] = fused
    result['all'] = list(dict.fromkeys(x for names in result.values() for x in names))
    return result


def compact(row):
    return {name: row[name] for name in ('outside', 'by_distance', 'corrupt_matrix')}


def numerical_summary(audits):
    result = {}
    for key, detail in audits.items():
        native = detail['native']
        for kind, entries in (('readout', [x['readout'] for x in native]),
                              ('prefix', [x['chunk_end'] for x in native]),
                              ('hybrid', detail['hybrid'])):
            part = result.setdefault(kind, {'count': 0, 'maxima': {}, 'worst_call': None})
            for entry in entries:
                require(all(math.isfinite(v) for v in entry.values()) and entry['max_mixed_ratio'] <= 1
                        and entry['rms_relative'] <= 2e-5 and entry['peak_relative'] <= 3e-5,
                        f'Numerical gate failed at {key} {kind}: {entry}')
                part['count'] += 1
                maxima = part['maxima']
                if entry['max_mixed_ratio'] > maxima.get('max_mixed_ratio', -1):
                    part['worst_call'] = list(key)
                for name, value in entry.items():
                    maxima[name] = max(value, maxima.get(name, 0))
    return result


def image(model, vmamba, arm, rgb, clean, corrupt, label, refs, roi, bands):
    captured, layouts, replacements, audits = {}, {}, {}, {}

    def capture(key, name, grid, native, args):
        require(key not in layouts, 'Duplicate native clean scan')
        scan = Scan(*args)
        layouts[key] = (name, geometry(name, key[2], grid, scan))
        out, prefix = native(*args)
        captured[key] = scan, out
        return out, prefix

    def compare(key, name, grid, native, args):
        scan = Scan(*args)
        require(layouts[key] == (name, geometry(name, key[2], grid, scan)), 'Paired scan layout differs')
        out, prefix = native(*args)
        prior, base = captured.pop(key)
        require(scan.nrows == prior.nrows and scan.softplus == prior.softplus
                and all(torch.equal(getattr(scan, x), getattr(prior, x)) for x in ('a', 'd', 'bias')),
                f'Frozen scan parameters differ at {key}')
        try:
            value = validated_factor(native, prior, scan, base, out)
        except (ValueError, RuntimeError) as exc:
            raise ValueError(f'Scan {key} {layouts[key]}: {exc}') from exc
        replacements[key], audits[key] = value, value.audit
        return out, prefix

    with intercept(model, vmamba, arm, capture):
        clean_logits = model(rgb, clean)
    with intercept(model, vmamba, arm, compare):
        corrupt_logits = model(rgb, corrupt)
    expected = {'encoder': 30, 'decoder': 12}
    if arm == 'ssm':
        expected.update({f'{name}.{i}': n for name, n in (('cross', 2), ('concat', 1)) for i in range(4)})
    require(not captured and set(layouts) == set(replacements) and
            Counter(name for name, _ in layouts.values()) == expected, 'Incomplete paired scans')
    a, b = predict(clean_logits), predict(corrupt_logits)
    hashes = {'clean': digest(a), 'natural_corrupt': digest(b)}
    require(list(hashes.values()) == [r['prediction_sha256'] for r in refs], 'Natural prediction differs from R2d')
    gt = label[0].cpu().numpy()
    matrix = confusion(gt, a).tolist()
    require(matrix == refs[0]['confusion_matrix'], 'Clean confusion differs from R2d')
    baseline = scores(matrix)
    natural = pair(a, b, gt, roi, bands, baseline)
    require(all(natural[k] == refs[1][k] for k in natural), 'Natural perturbation metrics differ from R2d')
    outcomes = {'natural_corrupt': compact(natural)}
    for scope, names in scopes(arm).items():
        for mask in range(8):
            seen = set()

            def substitute(key, name, grid, native, args):
                require(key not in seen and layouts[key][0] == name and
                        tuple(layouts[key][1]['grid']) == grid, 'Replay scan identity differs')
                seen.add(key)
                if name not in names:
                    return native(*args)
                output = replacements[key].output(mask)
                require(output.shape == args[0].shape and output.dtype == args[0].dtype, 'Replay shape differs')
                return output, None

            with intercept(model, vmamba, arm, substitute):
                logits = model(rgb, clean)
            require(seen == set(layouts), 'Replay scan coverage differs')
            if mask == 0:
                require(torch.equal(logits, clean_logits), f'Identity logits differ for {scope}')
            else:
                key = f'{scope}/{mask}'
                pred = predict(logits)
                hashes[key] = digest(pred)
                outcomes[key] = compact(pair(a, pred, gt, roi, bands, baseline))
    require(torch.equal(clean_logits, model(rgb, clean)) and torch.equal(corrupt_logits, model(rgb, corrupt)),
            'Hooks leaked after interventions')
    layout = [{'call': list(key), 'scope': name, **geo} for key, (name, geo) in layouts.items()]
    return {'clean_matrix': matrix, 'outcomes': outcomes, 'prediction_sha256': hashes,
            'numeric': numerical_summary(audits), 'identity_count': len(scopes(arm)), 'cleanup': True}, layout

