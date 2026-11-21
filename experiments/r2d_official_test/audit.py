"""Read-only verification of all persisted MFNet test metrics from per-image counts."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (3407, 3408, 3409, 3410, 3411)
ARMS = ('ssm', 'cnn', 'attention')
METRICS = ('miou', 'foreground_miou')
CLASS_COUNT = 9
PIXELS_PER_IMAGE = 480 * 640


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def equivalent(observed, expected):
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            equivalent(observed[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            equivalent(x, y) for x, y in zip(observed, expected))
    if isinstance(expected, float):
        return math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)
    return observed == expected


def inspect_events(path, names):
    counts = [[0]*CLASS_COUNT for _ in range(CLASS_COUNT)]
    rows = path.read_bytes().splitlines()
    require(len(rows) == len(names), f'{path}: unexpected event count')
    for pos, (raw, name) in enumerate(zip(rows, names, strict=True), 1):
        row = json.loads(raw)
        require((row['position'], row['id']) == (pos, name) and
                re.fullmatch(r'[0-9a-f]{64}', row['prediction_sha256']),
                f'{path}: invalid identity or prediction hash at row {pos}')
        matrix = row['confusion_matrix']
        require(len(matrix) == CLASS_COUNT and all(len(line) == CLASS_COUNT for line in matrix) and
                all(type(v) is int and v >= 0 for line in matrix for v in line) and
                sum(map(sum, matrix)) == PIXELS_PER_IMAGE,
                f'{path}: invalid pixel counts at row {pos}')
        for i, line in enumerate(matrix):
            for j, v in enumerate(line):
                counts[i][j] += v
    return counts


def calculate(counts):
    support = list(map(sum, counts))
    predictions = list(map(sum, zip(*counts)))
    union = [a+b-counts[i][i] for i, (a,b) in enumerate(zip(support, predictions))]
    require(all(u > 0 for u in union), 'Expected all nine classes in the official test')
    iou = [counts[i][i]/u for i,u in enumerate(union)]
    return dict(miou=mean(iou),foreground_miou=mean(iou[1:]),iou_0_to_8=iou,
                class_present_count=9,foreground_class_present_count=8,
                valid_pixel_count=sum(support),confusion_matrix=counts)


def describe(values):
    return dict(values=values,mean=mean(values),sample_std=stdev(values))


def audit(output):
    contract = read(output/'input-contract.json')
    summary = read(output/'summary.json')
    official = read(ROOT/'reports/evidence/r1c/lists.json')['lists']['test.txt']
    data = read(ROOT/'reports/evidence/r1b/data-manifest.json')
    profile = read(ROOT/'reports/evidence/r2d/resource-profile.json')
    names = official['ids']
    support = [data['class_pixel_counts'][str(i)] for i in range(CLASS_COUNT)]
    require(len(names) == len(set(names)) == 393 and
            contract['frozen_393']['test_txt_sha256'] == official['sha256'] == data['test_txt_sha256'] and
            contract['frozen_393']['class_pixel_counts'] == support and
            contract['frozen_393']['ignore_255_pixel_count'] == data['ignore_255_pixel_count'] == 0,
            'Original data contract differs')
    require(contract['frozen_393']['test_records_sha256'] == data['file_records_sha256'] and
            contract['frozen_393']['file_count'] == data['valid_file_records'] == 3 * len(names) and
            contract['frozen_393']['training_validation_overlap_count'] == 0,
            'Original inventory or ID isolation differs')
    source = Path(__file__).resolve().parent
    require(contract['runner_sha256'] == sha(source/'run.py') and
            contract['metrics_sha256'] == sha(source/'metrics.py') and
            contract['runtime'] == profile['runtime'] and
            profile['model_source_sha256'] == sha(ROOT/'experiments/r2d/models.py'),
            'Execution source or original R2d runtime fingerprint differs')
    require(contract['status']=='INPUTS_VERIFIED' and summary['status']=='COMPLETE' and
            summary['data_contract_sha256']==sha(output/'input-contract.json') and
            not summary['cross_dataset_validation'] and summary['official_test_pixel_reads_per_run']==len(names)
            and not contract['test_time_training'] and not contract['test_time_adaptation'],
            'Official inference scope differs')
    require(tuple(summary['seeds'])==tuple(contract['seeds'])==SEEDS and
            tuple(summary['arms'])==tuple(contract['arms'])==ARMS and
            len(contract['checkpoint_sha256'])==len(summary['run_file_sha256'])==
            len(summary['per_image_file_sha256'])==15, 'Incomplete or unexpected cohort')

    scores = {}
    verified = {}
    for seed in SEEDS:
        for arm in ARMS:
            key = f'{arm}/{seed}'
            rel = f'seed{seed}/{arm}'
            jsonl = f'{rel}.jsonl'
            path = f'{rel}.json'
            expected = read(ROOT/f'reports/evidence/r2d/seed{seed}/training/{arm}/checkpoint-ledger.json')[-1]
            row = read(output/path)
            require((row['status'],row['seed'],row['arm'],row['sample_count'])==
                    ('COMPLETE',seed,arm,len(names)) and
                    row['checkpoint_sha256']==contract['checkpoint_sha256'][key]==expected['last_sha256']
                    and row['model_sha256']==expected['model_sha256'] and not row['test_time_training']
                    and not row['test_time_adaptation'], f'{key}: checkpoint or scope differs')
            events_sha = sha(output/jsonl)
            require(events_sha == row['per_image_sha256'] == summary['per_image_file_sha256'][jsonl]
                    and sha(output/path)==summary['run_file_sha256'][path], f'{key}: artifact SHA256 differs')
            counts = inspect_events(output/jsonl,names)
            require(list(map(sum,counts))==support, f'{key}: class support differs')
            actual = calculate(counts)
            require(equivalent(row['metrics'],actual), f'{key}: persisted metrics differ')
            scores[seed,arm] = actual
            verified[key] = dict(sample_count=len(names), event_sha256=events_sha,
                                 checkpoint_sha256=row['checkpoint_sha256'])

    for arm in ARMS:
        expected = {metric:describe([scores[s,arm][metric] for s in SEEDS]) for metric in METRICS}
        expected['iou_0_to_8'] = [describe([scores[s,arm]['iou_0_to_8'][i] for s in SEEDS])
                                  for i in range(CLASS_COUNT)]
        require(equivalent(summary['arm_statistics'][arm],expected), f'{arm}: seed statistics differ')
    for left,right in (('ssm','cnn'),('ssm','attention'),('cnn','attention')):
        expected = {metric:describe([scores[s,left][metric]-scores[s,right][metric] for s in SEEDS])
                    for metric in METRICS}
        require(equivalent(summary['paired_statistics'][f'{left}_minus_{right}'],expected),
                f'{left}/{right}: paired differences differ')
    return dict(status='PASS', verified_runs=verified, total_per_image_records=15*len(names),
                class_support=support, count_per_run=sum(support), summary_sha256=sha(output/'summary.json'),
                input_contract_sha256=sha(output/'input-contract.json'),
                auditor_sha256=sha(Path(__file__).resolve()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='Completed run directory')
    parser.add_argument('--audit-file', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.output)
    with args.audit_file.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result,stream,indent=2,sort_keys=True)
        stream.write('\n')
    print(f'PASS {len(result["verified_runs"])} frozen runs and {result["total_per_image_records"]} prediction/count records')


if __name__ == '__main__':
    main()
