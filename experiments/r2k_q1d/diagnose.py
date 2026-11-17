"""Trace the accepted Q1 estimator without fitting or retuning it."""
import argparse
from collections import defaultdict
from pathlib import Path
import platform
import resource
import time

import numpy as np

from experiments.r2k_q1.model import degrade, describe, logits, probabilities, rng
from experiments.r2k_q1.protocol import CONFIG, ROOT, read, require, sha
from experiments.r2k_q1.run import ARMS, images, signature, write


def rank_metrics(y, p):
    """Non-interpolated AP and midrank AUROC, invariant to tie ordering."""
    require(y.shape == p.shape and 0 < y.sum() < len(y) and np.isfinite(p).all(),
            'Invalid ranking cohort')
    order = np.argsort(-p, kind='stable')
    y = y[order]
    p = p[order]
    ends = np.r_[np.flatnonzero(p[1:] != p[:-1]), len(p)-1]
    tp = np.cumsum(y)[ends]
    fp = ends + 1 - tp
    dpos = np.diff(np.r_[0,tp])
    dneg = np.diff(np.r_[0,fp])
    return {'ap':float(np.sum(tp/(ends+1)*dpos)/tp[-1]),
            'auc':float(np.sum((tp-dpos/2)*dneg)/(tp[-1]*fp[-1])),
            'brier':float(np.mean((p-y)**2)), 'levels':len(ends)}


def quantiles(x):
    if not len(x):
        return None
    return np.percentile(x,[10,50,90],axis=0).tolist()


def pixel_change(before, after, box):
    values = np.abs(after[box].astype(np.int16)-before[box].astype(np.int16)).ravel()
    return np.array([values.size,*[np.count_nonzero(values<=d) for d in (0,1,2,4)],
                     values.sum()],dtype=np.int64)


def strata_records():
    return defaultdict(lambda: {'points':0,'before':[],'after':[],'delta':[],
                                'contribution':[],'raw_delta':[],'pixels':np.zeros(6,np.int64)})


def trace(root, local, output):
    import cv2
    from scipy.special import expit

    start = time.perf_counter()
    cv2.setNumThreads(1)
    cfg = read(CONFIG)
    frozen = read(local/'frozen-fit.json')
    archive = ROOT/'reports/evidence/r2k-q1'
    commitment = read(archive/'calibration-freeze.json')
    prior = read(archive/'holdout-results.json')
    npz = local/'holdout.npz'
    index = local/'holdout-index.json'
    require(frozen['signature']==signature() and sha(local/'frozen-fit.json')==commitment['fit_sha256']
            and sha(npz)==prior['fingerprints']['predictions'] and sha(index)==prior['fingerprints']['index']
            and sha(local/'calibration.npz')==frozen['calibration_input_sha256'],
            'Original data or model fingerprint differs')
    forder = cfg['descriptors']
    head = frozen['heads']['thermal']
    raw_weight, scale = np.asarray(head['weights'][:-1]), np.asarray(head['scale'])
    slope, bias = frozen['calibration']['thermal']['calibration']
    threshold = frozen['calibration']['thermal']['threshold_bin']
    require(slope>0,'Unexpected nonmonotonic calibration')
    stats = strata_records()
    # Separate source roles. Probe the seen families only on the calibration role.
    plans = {'calibration':(('occlusion',0),('gaussian',0),('gaussian',1)),
             'holdout':(('blur',0),('blur',1))}
    pred, provenance = defaultdict(list), {}
    low = frozen['low_gradient']
    with np.load(npz,allow_pickle=False) as old:
        indices = {(r['id'],r['family'],r['tier']):j for j,r in enumerate(read(index))}
        for role,families in plans.items():
            for i,_,thermal,rgb in images(root,cfg['partitions'][role]['images']):
                clean = describe(thermal)
                raw = logits(clean,head)
                texture = clean[...,3] <= low
                t = i[-1]
                if role=='holdout':
                    j=indices[i,'healthy',0]
                    for arm in ARMS:
                        current=rgb if arm=='rgb' else thermal
                        q=probabilities(logits(describe(current),frozen['heads'][arm]),
                                        frozen['calibration'][arm]['calibration'])
                        require(np.array_equal(q.ravel(),old[arm][j]), 'Healthy archived predictions differ')
                for family,tier in families:
                    changed, roi, interior, _ = degrade(thermal,family,cfg,i,tier)
                    dirty = describe(changed)
                    z = logits(dirty,head)
                    calibrated = expit(slope*z+bias)
                    quantized = probabilities(z,[slope,bias])
                    origin = rng(cfg['seed'],i,family,tier)
                    size,margin = cfg['roi']['size'],cfg['roi']['margin']
                    yy,xx = (int(origin.integers(margin,d-size-margin+1)) for d in thermal.shape)
                    box = np.s_[yy:yy+size,xx:xx+size]
                    abs_counts = pixel_change(thermal,changed,box)
                    if role=='holdout':
                        j=indices[i,family,tier]
                        clean_j=indices[i,'healthy',0]
                        require(np.array_equal(roi.ravel(),old['target'][j])
                                and np.array_equal(quantized.ravel(),old['thermal'][j])
                                and np.array_equal(old['rgb'][j],old['rgb'][clean_j]),
                                'Frozen ROI or modality score differs')
                        pred['target'].append(roi.ravel())
                        for key,a in (('raw',z),('uncalibrated',expit(z)),('platt',calibrated),
                                      ('quantized',quantized/255),('rgb',old['rgb'][j]/255),
                                      ('shuffle',old['shuffle'][j]/255)):
                            pred[key].append(a.ravel())
                    delta = dirty.astype(np.float64)-clean
                    effect = delta / scale * raw_weight
                    require(np.allclose(effect.sum(-1),z-raw,atol=2e-8,rtol=2e-7),'Logit accounting differs')
                    label = f'{role}/{family}/{tier}'
                    selectors = {'all':roi,'interior':interior,'D' if t=='D' else 'N':roi,
                                 'low_texture':roi&texture,'higher_texture':roi&~texture,
                                 t+'/low_texture':roi&texture,t+'/higher_texture':roi&~texture}
                    for key,mask in selectors.items():
                        entry=stats[label+'/'+key]
                        entry['points']+=int(mask.sum())
                        for n,a in (('before',clean),('after',dirty),('delta',delta),('contribution',effect),
                                    ('raw_delta',z-raw)):
                            entry[n].append(a[mask])
                        if key in ('all','D','N'):
                            entry['pixels']+=abs_counts
    encoded={}
    for key,entry in stats.items():
        n=entry['points']
        values={k:quantiles(np.concatenate(entry[k])) for k in
                ('before','after','delta','contribution','raw_delta')}
        if n:
            changes=np.concatenate(entry['raw_delta'])
            values['raw_logit_decrease_fraction']=float(np.mean(changes<0))
        c=entry['pixels']
        if c[0]:
            values['source_pixels']={'n':int(c[0]),'fraction_abs_change_le':{
                str(d):float(c[j+1]/c[0]) for j,d in enumerate((0,1,2,4))},'mae_8bit':float(c[-1]/c[0])}
        encoded[key]={'points':n,**values}
    y=np.concatenate(pred.pop('target'))
    data={k:np.concatenate(v) for k,v in pred.items()}
    rankings={k:rank_metrics(y,v) for k,v in data.items() if k!='raw'}
    rankings['raw_logit']={k:v for k,v in rank_metrics(y,data['raw']).items() if k!='brier'}
    # Constant value is obtained from the original Calibration target prevalence.
    with np.load(local/'calibration.npz',allow_pickle=False) as cal:
        constant=float(cal['target'].mean())
    rankings['calibration_prevalence_constant']=rank_metrics(y,np.full(len(y),constant))
    binned=data['quantized']
    cutoff=threshold/255
    recall=float(np.mean(binned[y]>=cutoff))
    require(np.isclose(recall,prior['strata']['blur']['thermal']['recall']['point'],atol=1e-14)
            and np.isclose(rankings['quantized']['ap'],prior['strata']['blur']['thermal']['ap']['point'],atol=1e-14),
            'Accepted headline metrics did not reproduce')
    report={'status':'POST-HOC','no_training_or_retuning':True,'frozen_protocol_sha256':sha(CONFIG),
            'frozen_fit_sha256':sha(local/'frozen-fit.json'),'holdout_sha256':sha(npz),
            'source_sha256':sha(Path(__file__)),'feature_order':forder,'operating_threshold_bin':threshold,
            'feature_weight_signs':np.sign(raw_weight).astype(int).tolist(),
            'feature_scale_order_small_to_large':np.argsort(scale).tolist(),
            'ranking':rankings,'positive_fraction':float(y.mean()),'calibration_class_prior':constant,
            'threshold_recall':recall,'positive_zero_bin':float(np.mean(binned[y]==0)),
            'negative_zero_bin':float(np.mean(binned[~y]==0)),
            'positive_below_threshold_count':int(np.sum(binned[y]<cutoff)),
            'strata':encoded,'seconds':time.perf_counter()-start,
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'opencv':cv2.__version__,
                           'opencv_threads':cv2.getNumThreads()}}
    write(output,report)
    print('POST-HOC PASS: accepted hashes, aligned logits and archived Holdout scores reproduced',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','local','output'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    trace(a.root,a.local,a.output)
