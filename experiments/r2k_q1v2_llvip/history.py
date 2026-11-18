"""Repeat the archived cross-dataset screen without reading historic Holdout pixels."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .scan import dump, phash

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ('reports/evidence/r1c/manifest.json', 'reports/evidence/r1b/data-manifest.json')
Q1 = 'configs/R2k_Q1_protocol.json'
R1G = 'reports/evidence/r1g/split.json'
R1F = 'reports/evidence/r1f/split.json'


def load(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_calibration_ids(config, validation):
    p = config['partitions']
    fit, cal, holdout = (set(p[k]['images']) for k in ('fit', 'calibration', 'holdout'))
    if fit & cal or (fit | cal) & (holdout | set(validation)):
        raise ValueError('Historical partition overlap: refusing all historic pixel reads')
    return sorted(fit | cal)


def compare(rows, historical_root, archived):
    paths = (*MANIFESTS, Q1, R1G, R1F)
    declared = {**archived['manifests'], **archived['partition_metadata_sha256']}
    if any(sha(ROOT/p) != declared[p] for p in paths):
        raise ValueError('Archived historical source SHA256 differs')
    catalog = {}
    for p in MANIFESTS:
        for r in load(ROOT/p)['files']:
            if r['path'].startswith(('RGB/','Modal/')):
                prior = catalog.setdefault(r['path'],r['sha256'])
                if prior != r['sha256']:
                    raise ValueError('Conflicting archived historical digest')
    index = defaultdict(list)
    for p,h in catalog.items():
        index[h].append(p)
    imgs = [r for r in rows if 'modality' in r and 'error' not in r]
    matches = [[r['path'],index[r['sha256']]] for r in imgs if r['sha256'] in index]
    ids = fit_calibration_ids(load(ROOT/Q1),load(ROOT/R1G)['validation_ids'])
    screens = {}
    for mod,sub in [('visible','RGB'),('infrared','Modal')]:
        hist = []
        for i in ids:
            rel = f'{sub}/{i}.png'
            path = historical_root/rel
            if sha(path) != catalog[rel]:
                raise ValueError('Historical Fit/Calibration file differs: '+rel)
            with Image.open(path) as im:
                hist.append(int(phash(im),16))
        vec = np.array(hist,dtype=np.uint64)
        pairs = left = 0
        minimum = 64
        for r in imgs:
            if r['modality'] != mod:
                continue
            dist = np.bitwise_count(vec ^ np.uint64(int(r['phash63'],16)))
            hits = int((dist<=6).sum())
            pairs += hits
            left += bool(hits)
            minimum = min(minimum,int(dist.min()))
        screens[mod] = {'eligible_originals':len(ids),'hashed':len(vec),
                        'pairs_at_hamming_le_6':pairs,'left_images_with_hit':left,
                        'minimum_distance':minimum}
    return {'records':len(catalog),'sha256_hits':matches,
            'fit_calibration_only_perceptual':screens,'holdout_pixel_reads':0,
            'official_mfnet_test_pixel_reads':0}


def main():
    p=argparse.ArgumentParser()
    for arg in ('manifest','historical-root','output'):
        p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args()
    archive=load(ROOT/'reports/evidence/llvip-precheck/data_audit.json')
    if sha(a.manifest)!=archive['manifest_sha256']:
        raise ValueError('LLVIP committed manifest differs')
    expected=archive['historical']
    result=compare(load(a.manifest)['files'],a.historical_root,expected)
    for key in ('records','sha256_hits'):
        if result[key]!=expected[key]:
            raise ValueError('Historical audit differs: '+key)
    for arm,actual in result['fit_calibration_only_perceptual'].items():
        prior=expected['fit_calibration_only_perceptual'][arm]
        if any(actual[k]!=(prior[k] if k in prior else prior['screen'][k]) for k in actual):
            raise ValueError('Historical perceptual summary differs: '+arm)
    dump(a.output,{'status':'PASS',**result,'manifest_sha256':archive['manifest_sha256'],
                   'historical_metadata_sha256':{**expected['manifests'],
                                                  **expected['partition_metadata_sha256']},
                   'implementation_sha256_lf': hashlib.sha256(
                       Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()})
    print('Historical audit PASS:',result['records'],'digests; 500 Fit/Calibration images per mode')


if __name__=='__main__':
    main()
