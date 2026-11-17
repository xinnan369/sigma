"""Freeze original training image groups and data-independent settings."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/R2k_Q1_protocol.json'
SPLIT = ROOT / 'reports/evidence/r1g/split.json'
GRAPH = ROOT / 'reports/evidence/r1f/scene-audit.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def build():
    split, graph = read(SPLIT), read(GRAPH)
    require(digest({k:v for k,v in split.items() if k!='split_sha256'})==split['split_sha256']
            and digest(graph)==split['graph_sha256'], 'Parent split/graph hash differs')
    raw=set(split['train_ids'])
    ids=sorted(i for i in raw if i.endswith(('D','N')))
    require(len(ids)==627 and raw==set(ids)|{i+'_flip' for i in ids}
            and not set(ids)&set(split['validation_ids']), 'Parent training cohort differs')
    assignment={i:g[0] for g in graph['groups'] for i in g}
    groups=sorted({assignment[i] for i in ids})
    require(not set(groups)&{assignment[i] for i in split['validation_ids']}, 'Parent groups overlap')
    counts=np.array([[sum(assignment[i]==g and i.endswith(t) for i in ids) for t in ('D','N')] for g in groups])
    target=np.array([.6,.2,.2])
    cut=[round(.6*len(groups)),round(.8*len(groups))]
    rng=np.random.default_rng(3407)
    best=None
    for _ in range(256):
        parts=np.split(rng.permutation(len(groups)),cut)
        observed=np.stack([counts[p].sum(0) for p in parts])/counts.sum(0)
        error=float(np.square(observed-target[:,None]).sum())
        if best is None or error<best[0]:
            best=(error,parts)
    partitions={}
    for name,p in zip(('fit','calibration','holdout'),best[1],strict=True):
        gset={groups[j] for j in p}
        partitions[name]={'groups':sorted(gset),'images':{i:assignment[i] for i in ids if assignment[i] in gset}}
    return {'version':'r2k-q1-v1','split_sha256':split['split_sha256'],'graph_sha256':split['graph_sha256'],
            'seed':3407,'partitions':partitions,'split_search_permutations':256,
            'grid':{'stride':8,'offset':4},'descriptors':['local mean','local variance','local range',
            'mean absolute gradient','local saturation proportion','center minus local mean magnitude'],
            'descriptor_window':9,'parameters_per_head':7,'fit_samples_per_view_class':64,
            'optimizer':{'method':'L-BFGS-B','maxiter':200,'l2':.001},
            'roi':{'size':96,'margin':12},'fit_calibration_families':['occlusion','gaussian'],
            'gaussian_sigmas':[12,24],'blur':{'tiers':[[7,1.6],[17,4.0]],'holdout_only':True},
            'calibration':['identity','logit_platt'],'probability_bins':256,'ece_bins':10,
            'health_target_fpr':.05,'health_fpr_limit':.07,'low_texture_fpr_limit':.10,
            'low_texture_calibration_quantile':.20,'bootstrap_draws':10000,
            'threshold':'lowest quantized threshold with healthy calibration FPR <= 0.05',
            'holdout_family':'blur','primary_brier_rule':'favorable point; report paired CI',
            'success':'AP lower CI > 0, Brier difference < 0, healthy FPR <= .07, low-texture FPR <= .10, AP advantages positive within D/N, deep interior, and each ROI-center quadrant',
            'inconclusive':'fewer than 10 holdout groups, fewer than 5 per D/N stratum, or AP interval includes 0 without other failure'}


if __name__=='__main__':
    print(json.dumps(build(),indent=2))
