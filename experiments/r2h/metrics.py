import numpy as np

from experiments.r2a.metrics import confusion

FIELDS = ('pixels', 'induced', 'recovered', 'introduced', 'corrected', 'new_change', 'net_change', 'disagree')


def repair(clean, corrupt, restored, gt, roi, bands):
    valid = (gt >= 0) & (gt < 9) & ~roi
    clean_wrong = clean != gt
    corrupt_wrong = corrupt != gt
    repaired_wrong = restored != gt
    induced = ~clean_wrong & corrupt_wrong
    recovered = induced & ~repaired_wrong
    introduced = ~corrupt_wrong & repaired_wrong
    corrected = corrupt_wrong & ~repaired_wrong
    new_change = ~clean_wrong & introduced

    def counts(region):
        region = valid & region
        values = [region, region & induced, region & recovered, region & introduced,
                  region & corrected, region & new_change, region & (restored != corrupt)]
        n, base, fix, harm, all_fix, new, disagree = [int(x.sum()) for x in values]
        return dict(zip(FIELDS, (n, base, fix, harm, all_fix, new - fix, harm - all_fix, disagree)))

    return {'outside': counts(valid), 'bands': {k: counts(v) for k, v in bands.items()},
            'classes': [counts(gt == k) for k in range(9)],
            'matrix': confusion(gt, restored).tolist()}
