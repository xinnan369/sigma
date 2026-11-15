"""Replay accepted CUDA readouts inside the natural corrupt input context."""
from collections import Counter

import torch

from experiments.r1g.partition import require
from experiments.r2a.metrics import confusion, digest, pair, scores
from experiments.r2b.state import Scan, check_close
from experiments.r2e.paths import direct, geometry, intercept
from experiments.r2e.run import predict
from experiments.r2f.numerics import basis_states
from experiments.r2f.replay import reconstruct, rounded_reference, validated_factor
from experiments.r2g.replay import numerical_summary, scopes
from experiments.r2h.metrics import repair


def image(model, vmamba, arm, rgb, clean, corrupt, label, refs, roi, bands, donor):
    captured, layouts, replacements, audits = {}, {}, {}, {}

    donor_rgb, donor_thermal, donor_ref = donor
    mismatched = {}
    wrong_scans = {}

    def capture_donor(key, name, grid, native, args):
        if name == 'encoder':
            wrong_scans[key] = Scan(*args)
        return native(*args)

    with intercept(model, vmamba, arm, capture_donor):
        donor_logits = model(donor_rgb, donor_thermal)
    require(digest(predict(donor_logits)) == donor_ref, 'Mismatch donor Clean prediction differs from R2d')
    del donor_logits

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
        if name == 'encoder':
            wrong = wrong_scans.pop(key)
            require(geometry(name, key[2], grid, wrong) == layouts[key][1], 'Mismatch donor layout differs')
            h = basis_states(native, wrong)
            skip = direct(scan)
            mixed = reconstruct(h, scan.c, skip)
            audit = check_close(mixed, rounded_reference(h, scan.c, skip), f'Mismatch {key}')
            mismatched[key] = (mixed, audit)
        replacements[key], audits[key] = value, value.audit
        return out, prefix

    with intercept(model, vmamba, arm, capture):
        clean_logits = model(rgb, clean)
    with intercept(model, vmamba, arm, compare):
        corrupt_logits = model(rgb, corrupt)
    expected = {'encoder': 30, 'decoder': 12}
    if arm == 'ssm':
        expected.update({f'{name}.{i}': n for name, n in (('cross', 2), ('concat', 1)) for i in range(4)})
    require(not captured and not wrong_scans and len(mismatched) == 30 and set(layouts) == set(replacements) and
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
    outcomes = {'natural_corrupt': repair(a, b, b, gt, roi, bands)}
    conditions = [(scope, names, mask) for scope, names in scopes(arm).items()
                  for mask in range(8)]
    conditions.append(('mismatch', ['encoder'], 6))
    for scope, names, mask in conditions:
        seen = set()

        def substitute(key, name, grid, native, args):
            require(key not in seen and layouts[key][0] == name and
                    tuple(layouts[key][1]['grid']) == grid, 'Repair scan identity differs')
            seen.add(key)
            if name not in names:
                return native(*args)
            output = mismatched[key][0] if scope == 'mismatch' else replacements[key].output(mask)
            require(output.shape == args[0].shape and output.dtype == args[0].dtype, 'Repair shape differs')
            return output, None

        with intercept(model, vmamba, arm, substitute):
            logits = model(rgb, corrupt)
        require(seen == set(layouts), 'Repair scan coverage differs')
        if mask == 7:
            require(torch.equal(logits, corrupt_logits), f'Corrupt Identity logits differ for {scope}')
        else:
            key = f'{scope}/{mask}'
            pred = predict(logits)
            hashes[key] = digest(pred)
            outcomes[key] = repair(a, b, pred, gt, roi, bands)
    require(torch.equal(clean_logits, model(rgb, clean)) and torch.equal(corrupt_logits, model(rgb, corrupt)),
            'Hooks leaked after interventions')
    layout = [{'call': list(key), 'scope': name, **geo} for key, (name, geo) in layouts.items()]
    return {'clean_matrix': matrix, 'outcomes': outcomes, 'prediction_sha256': hashes,
            'numeric': numerical_summary(audits), 'mismatch_numeric': {field: max(v[1][field] for v in mismatched.values()) for field in next(iter(mismatched.values()))[1]}, 'identity_count': len(scopes(arm)), 'cleanup': True}, layout
