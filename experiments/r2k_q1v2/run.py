"""Fit and calibrate frozen train-side heads. There is no evaluation entry point."""
import argparse
from collections import defaultdict
import platform
from pathlib import Path
import resource
import statistics
import time

import numpy as np

from experiments.r2k_q1 import metrics
from experiments.r2k_q1.model import describe, logistic, logits, operating_threshold, probabilities, rng
from experiments.r2k_q1.protocol import ROOT, read, require, sha
from experiments.r2k_q1.run import HEALTH, SEEN, images, views, write
from experiments.r2k_q1v2.features import describe_multiscale

CONFIG = ROOT/'configs/R2k_Q1v2_protocol.json'
ARMS=('q1','thermal','rgb','ablated','shuffle')


def frozen_inputs():
    cfg=read(CONFIG)
    parent=ROOT/cfg['parent']
    require(sha(parent)==cfg['parent_sha256'] and cfg['roles']==['fit','calibration']
            and cfg['families']==['healthy','occlusion','gaussian'] and cfg['blur_entirely_held_out']
            and cfg['scales']==[3,9,21] and cfg['grid']=={'stride':8,'offset':4}
            and cfg['seed']==3407, 'Frozen protocol differs')
    old=read(parent)
    a,b,h=[old['partitions'][k] for k in ('fit','calibration','holdout')]
    require(not set(a['groups'])&set(b['groups']) and not (set(a['groups'])|set(b['groups']))&set(h['groups'])
            and not set(a['images'])&set(b['images']) and not (set(a['images'])|set(b['images']))&set(h['images']),
            'Frozen image/group roles overlap')
    return cfg,old


def features(image, arm, cfg):
    if arm=='q1':
        f=describe(image)
        return f, np.ones(f.shape[:2],dtype=bool)
    return describe_multiscale(image,cfg,ablated=arm=='ablated')


def predict(image, arm, cfg, head, calibration):
    f, identifiable=features(image,arm,cfg)
    return probabilities(logits(f,head),calibration),identifiable


def signed_sources():
    return {p.relative_to(ROOT).as_posix():sha(p) for package in ('r2k_q1','r2k_q1v2')
            for p in sorted((ROOT/'experiments'/package).glob('*.py'))}


def sample_positions(target,cfg,i,family,tier):
    out=[]
    for value in (False,True):
        choices=np.flatnonzero(target.ravel()==value)
        if len(choices):
            out.extend(rng(cfg['seed'],'fit',i,family,tier,value).choice(
                choices,cfg['fit_samples_per_view_class'],replace=False))
    return out


def extract(root,part,cfg,old,fit):
    arrays=defaultdict(list)
    index=[]
    profiling=defaultdict(list)
    for i,g,t,r in images(root,part['images']):
        rgb,rc=features(r,'rgb',cfg)
        for family,tier,changed,mask,_,_ in views(t,old,i,(HEALTH,*SEEN)):
            require(family in cfg['families'], 'Disallowed Fit or Calibration view')
            idx=sample_positions(mask,old,i,family,tier) if fit else slice(None)
            tf,tc=features(changed,'thermal',cfg)
            af,ac=features(changed,'ablated',cfg)
            q1,_=features(changed,'q1',cfg)
            arrays['target'].append(mask.ravel()[idx])
            arrays['healthy'].append(np.full(len(arrays['target'][-1]),family=='healthy',bool))
            if not fit:
                arrays['low'].append((describe(t)[...,3].ravel() <= old['diagnostic_low_gradient']))
                for arm,args in (('q1',(changed,'q1')),('thermal',(changed,'thermal')),
                                  ('rgb',(r,'rgb')),('ablated',(changed,'ablated'))):
                    start=time.perf_counter()
                    features(*args,cfg)
                    profiling[arm].append(1000*(time.perf_counter()-start))
            for arm,data,coverage in (('rgb',rgb,rc),('thermal',tf,tc),('ablated',af,ac),
                                      ('q1',q1,np.ones(mask.shape,bool))):
                require(data.shape==(*mask.shape,6) and np.isfinite(data).all(), 'Descriptors differ')
                arrays[arm].append(data.reshape(-1,6)[idx])
                arrays[arm+'_coverage'].append(coverage.ravel()[idx])
            if not fit:
                index.append({'id':i,'group':g,'family':family,'tier':tier,'points':int(mask.size),
                              'positive':int(mask.sum())})
            else:
                index.append({'id':i,'group':g,'family':family,'tier':tier,
                              'points':len(arrays['target'][-1]),'positive':int(arrays['target'][-1].sum())})
    return {k:np.concatenate(v) for k,v in arrays.items()},index,{a:statistics.median(v)
                                                               for a,v in profiling.items()}


def fit_heads(x,old):
    y=x['target']
    heads={}
    for arm in ARMS:
        f=x['thermal' if arm=='shuffle' else arm].astype(np.float64)
        target=rng(old['seed'],'label_permutation').permutation(y) if arm=='shuffle' else y
        mean=f.mean(0)
        std=np.maximum(f.std(0),1e-6)
        w,n=logistic((f-mean)/std,target,old)
        require(len(w)==7 and np.isfinite(w).all(), 'Fit parameter mismatch')
        heads[arm]={'weights':w.tolist(),'mean':mean.tolist(),'scale':std.tolist(),'iterations':n}
    return heads


def calibrate(draws,old):
    healthy=draws['healthy']
    y=draws['target']
    calibrations, outputs={},{}
    for arm in ARMS:
        z=draws[arm]
        w,n=logistic(z[:,None],y,old)
        choices={'identity':[1.,0.],'logit_platt':w.tolist()}
        errors={k:float(np.mean((probabilities(z,v)/255-y)**2)) for k,v in choices.items()}
        method=min(choices,key=errors.get)
        bins=probabilities(z,choices[method])
        threshold=operating_threshold(bins[healthy],old['health_target_fpr'])
        outputs[arm]=bins
        calibrations[arm]={'choices':choices,'method':method,'candidate_brier':errors,
                           'threshold_bin':threshold,'iterations':n}
    constant=np.rint(255*y.mean()).astype(np.uint8)
    outputs['constant']=np.full(len(y),constant,np.uint8)
    calibrations['constant']={'method':'calibration_prior','threshold_bin':int(constant)+1}
    return calibrations,outputs


def results(draws, outputs, calibration, index):
    out={}
    y=draws['target']
    healthy=draws['healthy']
    family=np.repeat([r['family'] for r in index],[r['points'] for r in index])
    for arm,p in outputs.items():
        t=calibration[arm]['threshold_bin']
        m=metrics.metrics(metrics.distribution(y,p))
        cov=draws.get(('thermal' if arm=='shuffle' else arm)+'_coverage')
        coverage=np.ones_like(y) if cov is None else cov
        out[arm]={**{k:float(v) for k,v in m.items()},
                  'healthy_fpr':float(np.mean(p[healthy]>=t)),
                  'low_texture_healthy_fpr':float(np.mean(p[healthy&draws['low']]>=t)),
                  'flag_coverage':float(np.mean(coverage)), 'coverage_by_family':{
                      f:float(np.mean(coverage[family==f])) for f in ('healthy','occlusion','gaussian')},
                  'evaluated_points':len(p),'recall':{
                      f:float(np.mean(p[(family==f)&y]>=t)) for f in ('occlusion','gaussian')}}
    return out


def run(root,local,evidence):
    import cv2
    import scipy

    begin=time.perf_counter()
    cv2.setNumThreads(1)
    cfg,old=frozen_inputs()
    local.mkdir(exist_ok=False)
    old=dict(old,diagnostic_low_gradient=read(ROOT/'reports/evidence/r2k-q1/calibration-freeze.json')['low_gradient'])
    f,fit_index,_=extract(root,old['partitions']['fit'],cfg,old,True)
    heads=fit_heads(f,old)
    del f
    c,cal_index,timing=extract(root,old['partitions']['calibration'],cfg,old,False)
    draws={k:v for k,v in c.items() if k in ('healthy','target','low') or k.endswith('_coverage')}
    for a in ARMS:
        draws[a]=logits(c['thermal' if a=='shuffle' else a],heads[a])
    del c
    cal,p=calibrate(draws,old)
    stats=results(draws,p,cal,cal_index)
    _,_,thermal,rgb=next(images(root,old['partitions']['calibration']['images']))
    latency={}
    for arm in ARMS:
        image=rgb if arm=='rgb' else thermal
        start=[]
        for k in range(21):
            t=time.perf_counter()
            scores,flag=predict(image,arm,cfg,heads[arm],cal[arm]['choices'][cal[arm]['method']])
            if k:
                start.append(1000*(time.perf_counter()-t))
        require(scores.shape==flag.shape==(60,80), 'Inference coverage differs')
        latency[arm]=statistics.median(start)
    np.savez_compressed(local/'calibration.npz',**draws,**{f'bins_{a}':v for a,v in p.items()})
    sources=signed_sources()
    write(local/'frozen-parameters.json',{'heads':heads,'calibration':cal,'source_sha256':sources,
                                          'protocol_sha256':sha(CONFIG)})
    public={a:{k:v for k,v in entry.items() if k!='choices'} for a,entry in cal.items()}
    write(evidence/'fit-calibration.json',{'scope':'Fit/Calibration only; no independent evaluation',
          'protocol_sha256':sha(CONFIG),'source_sha256':sources,'frozen_parameters_sha256':
          sha(local/'frozen-parameters.json'),'calibration_arrays_sha256':sha(local/'calibration.npz'),
          'fit_index':fit_index,'calibration_index':cal_index,'calibration':public,'metrics':stats,
          'fit_iterations':{a:h['iterations'] for a,h in heads.items()},'parameters_each':7,
          'feature_passes':{'q1':{'box':4,'sobel':2,'min_max_morphology':2},
                            'multiscale':{'box':10,'sobel':4,'max_morphology':1}},
          'feature_only_latency_median_ms':timing,'single_frame_latency_median_ms':latency,'seconds':time.perf_counter()-begin,
          'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'environment':{'python':platform.python_version(),'numpy':np.__version__,
                         'scipy':scipy.__version__,'opencv':cv2.__version__,'opencv_threads':cv2.getNumThreads()},
          'holdout_image_reads':0,'blur_views':0,'exit':0})
    print('TRAIN-SIDE CPU COMPLETE', flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for a in ('root','local','evidence'):
        p.add_argument('--'+a,type=Path,required=True)
    a=p.parse_args()
    run(a.root,a.local,a.evidence)
