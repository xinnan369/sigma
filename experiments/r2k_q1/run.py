"""Separate image access for fit/calibration and final holdout."""
import argparse
import platform
import resource
import time
from pathlib import Path

import numpy as np

from experiments.r2k_q1.model import (degrade, describe, logistic, logits, operating_threshold, predict,
                                      probabilities, rng)
from experiments.r2k_q1.protocol import CONFIG, ROOT, build, read, require, sha
from experiments.r2k_q1 import metrics

ARMS=('thermal','rgb','shuffle')
SEEN=(('occlusion',0),('gaussian',0),('gaussian',1))
HEALTH=('healthy',0)


def write(path, value):
    import json
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def signature():
    return {'protocol_sha256':sha(CONFIG),'source_sha256':{
        p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}


def images(root, roster):
    import cv2

    manifest=read(ROOT/'reports/evidence/r1c/manifest.json')
    expected={r['path']:r['sha256'] for r in manifest['files']}
    for i,group in roster.items():
        views=[]
        for folder in ('Modal','RGB'):
            name=f'{folder}/{i}.png'
            path=root/name
            require(sha(path)==expected[name],f'Input bytes differ: {name}')
            view=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            require(view is not None and view.shape==(480,640),f'Image dimensions differ: {name}')
            views.append(view)
        yield i,group,*views


def views(thermal, cfg, i, families):
    grid_shape=thermal[4::8,4::8].shape
    for family,tier in families:
        if family=='healthy':
            mask=np.zeros(grid_shape,bool)
            yield family,tier,thermal,mask,mask,-1
        else:
            yield family,tier,*degrade(thermal,family,cfg,i,tier)


def extract_fit(root, roles, cfg):
    data={a:[] for a in ARMS[:2]}
    labels=[]
    log=[]
    for i,g,t,r in images(root,roles['fit']['images']):
        rf=describe(r).reshape(-1,6)
        for family,tier,changed,target,_,_ in views(t,cfg,i,(HEALTH,*SEEN)):
            require(family in ('healthy',*cfg['fit_calibration_families']), 'Fitting received held-out family')
            indices=[]
            for value in (False,True):
                candidates=np.flatnonzero(target.ravel()==value)
                if len(candidates):
                    indices.extend(rng(cfg['seed'],'fit',i,family,tier,value).choice(
                        candidates,cfg['fit_samples_per_view_class'],replace=False))
            data['thermal'].append(describe(changed).reshape(-1,6)[indices])
            data['rgb'].append(rf[indices])
            labels.append(target.ravel()[indices])
            log.append({'id':i,'group':g,'family':family,'tier':tier,'sampled':len(indices),
                        'positives':int(labels[-1].sum())})
    y=np.concatenate(labels)
    shuffled=rng(cfg['seed'],'label_permutation').permutation(y)
    heads={}
    for arm,key,target in (('thermal','thermal',y),('rgb','rgb',y),('shuffle','thermal',shuffled)):
        x=np.concatenate(data[key],dtype=np.float64)
        mean=x.mean(0)
        scale=np.maximum(x.std(0),1e-6)
        w,n=logistic((x-mean)/scale,target,cfg)
        heads[arm]={'mean':mean.tolist(),'scale':scale.tolist(),'weights':w.tolist(),'iterations':n}
    return heads, {'labels':len(y),'positives':int(y.sum()),'views_per_image':4,'parameters_each':7,'rows':log}


def calibrate(root, roles, cfg, heads, local):
    z={a:[] for a in ARMS}
    labels,health,gradient=[],[],[]
    log=[]
    for i,g,t,r in images(root,roles['calibration']['images']):
        rf=describe(r)
        for family,tier,changed,target,_,_ in views(t,cfg,i,(HEALTH,*SEEN)):
            require(family in ('healthy',*cfg['fit_calibration_families']), 'Calibration received held-out family')
            tf=describe(changed)
            labels.append(target.ravel())
            health.append(np.full(target.size,family=='healthy'))
            if family=='healthy':
                gradient.extend(tf[...,3].ravel().tolist())
            for arm in ARMS:
                z[arm].append(logits(rf if arm=='rgb' else tf,heads[arm]).ravel())
            log.append({'id':i,'group':g,'family':family,'tier':tier,'grid_points':target.size,
                        'positives':int(target.sum())})
    y=np.concatenate(labels)
    healthy=np.concatenate(health)
    np.savez_compressed(local/'calibration.npz',target=y,healthy=healthy,
                        **{a:np.concatenate(z[a]) for a in ARMS})
    result={}
    for arm in ARMS:
        flat=np.concatenate(z[arm])
        w,n=logistic(flat[:,None],y,cfg)
        choices={'identity':[1.,0.],'logit_platt':w.tolist()}
        scores={k:float(np.mean((probabilities(flat,v)/255-y)**2)) for k,v in choices.items()}
        selected=min(choices,key=scores.get)
        prob=probabilities(flat,choices[selected])
        threshold=operating_threshold(prob[healthy],cfg['health_target_fpr'])
        result[arm]={'chosen':selected,'calibration':choices[selected],'candidate_brier':scores,'candidates':choices,
                     'threshold_bin':threshold,'calibration_healthy_fpr':float(np.mean(prob[healthy]>=threshold)),
                     'iterations':n}
    return result, float(np.quantile(gradient,cfg['low_texture_calibration_quantile'])), log


def fit(root, local, cfg):
    local.mkdir(exist_ok=False)
    heads,fit_log=extract_fit(root,cfg['partitions'],cfg)
    cal,low,log=calibrate(root,cfg['partitions'],cfg,heads,local)
    artifact={'signature':signature(),'heads':heads,'calibration':cal,'low_gradient':low,
              'calibration_input_sha256':sha(local/'calibration.npz'),'fit_log':fit_log,'calibration_log':log,'split_roles_read':['fit','calibration'],
              'families_read':['healthy','occlusion','gaussian']}
    write(local/'frozen-fit.json',artifact)
    print('FIT PASS',{k:v for k,v in fit_log.items() if k!='rows'},flush=True)


def finish(root, local, cfg, evidence):
    frozen=read(local/'frozen-fit.json')
    require(frozen['signature']==signature(), 'Frozen pre-holdout artifact or sources differ')
    require(not (local/'holdout.npz').exists(), 'Holdout already executed')
    require(frozen['families_read']==['healthy','occlusion','gaussian'], 'Holdout family leaked')
    require(sha(local/'frozen-fit.json')==read(evidence/'calibration-freeze.json')['fit_sha256'],
            'Pre-holdout calibration commitment differs')
    roles=cfg['partitions']
    rows=[]
    arrays={k:[] for k in (*ARMS,'target','interior','low')}
    for i,g,t,r in images(root,roles['holdout']['images']):
        low=describe(t)[...,3]<=frozen['low_gradient']
        for family,tier,changed,target,interior,q in views(t,cfg,i,(HEALTH,('blur',0),('blur',1))):
            rows.append({'id':i,'group':g,'family':family,'tier':tier,'quadrant':q})
            for arm in ARMS:
                c=frozen['calibration'][arm]['calibration']
                arrays[arm].append(predict(r if arm=='rgb' else changed,frozen['heads'][arm],c).ravel())
            arrays['target'].append(target.ravel())
            arrays['interior'].append(interior.ravel())
            arrays['low'].append(low.ravel())
    np.savez_compressed(local/'holdout.npz',**arrays)
    write(local/'holdout-index.json',rows)
    report=score(rows,arrays,frozen,cfg)
    report['fingerprints']={'fit':sha(local/'frozen-fit.json'),'predictions':sha(local/'holdout.npz'),
                            'index':sha(local/'holdout-index.json'),**signature()}
    write(evidence/'holdout-results.json',report)
    print('HOLDOUT',report['decision'],flush=True)


def score(rows, arrays, frozen, cfg):
    groups=cfg['partitions']['holdout']['groups']
    n=len(groups)
    weights=np.vstack((np.ones(n),rng(cfg['seed'],'bootstrap').multinomial(
        n,np.full(n,1/n),size=cfg['bootstrap_draws'])))
    thresholds={a:frozen['calibration'][a]['threshold_bin'] for a in ARMS}
    tables={}
    for j,row in enumerate(rows):
        i=groups.index(row['group'])
        y=np.asarray(arrays['target'][j])
        all_points=np.ones(len(y),bool)
        areas={}
        if row['family']=='healthy':
            areas={'healthy':all_points,'healthy/'+row['id'][-1]:all_points,
                   'healthy/low_texture':arrays['low'][j]}
        else:
            areas={'blur':all_points,'blur/'+row['id'][-1]:all_points,
                   'blur/tier'+str(row['tier']):all_points,'blur/q'+str(row['quadrant']):all_points,
                   'blur/interior':~y|arrays['interior'][j]}
        for stratum,mask in areas.items():
            block=tables.setdefault(stratum,{a:np.zeros((n,2,256),np.int64) for a in ARMS})
            for a in ARMS:
                block[a][i]+=metrics.distribution(y[mask],arrays[a][j][mask])
    results={}
    for key, table in tables.items():
        draw=metrics.bootstrap(weights,table,thresholds)
        out={a:{k:metrics.summary(v) for k,v in data.items()} for a,data in draw.items()}
        out['thermal_minus_rgb']={k:metrics.summary(draw['thermal'][k]-draw['rgb'][k]) for k in draw['rgb']}
        out['reliability_curve']={a:metrics.curve(t.sum(0)) for a,t in table.items()}
        if key in ('blur','healthy'):
            out['visual_groups']={g:{a:{'points':int(t[i].sum()),'positives':int(t[i,1].sum()),
                         'flagged_positive':int(t[i,1,thresholds[a]:].sum()),
                         'flagged_negative':int(t[i,0,thresholds[a]:].sum())}
                         for a,t in table.items()} for i,g in enumerate(groups)}
        results[key]=out
    primary=results['blur']['thermal_minus_rgb']
    health=results['healthy']['thermal']['fpr']['point']
    low=results['healthy/low_texture']['thermal']['fpr']['point']
    conditional=all(results[k]['thermal_minus_rgb']['ap']['point']>0 for k in
                    ('blur/D','blur/N','blur/interior',*(f'blur/q{i}' for i in range(4))))
    sufficient=n>=10 and all(len({r['group'] for r in rows if r['id'].endswith(t)})>=5 for t in 'DN')
    if not sufficient:
        decision='INCONCLUSIVE'
    elif health>cfg['health_fpr_limit'] or low>cfg['low_texture_fpr_limit'] or primary['brier']['point']>=0 or not conditional:
        decision='FAIL'
    elif primary['ap']['ci95'][0]>0:
        decision='PASS'
    elif primary['ap']['ci95'][1]<0:
        decision='FAIL'
    else:
        decision='INCONCLUSIVE'
    healthy_flags=[]
    for j,r in enumerate(rows):
        if r['family']=='healthy':
            flags=np.asarray(arrays['thermal'][j])>=thresholds['thermal']
            healthy_flags.append({'id':r['id'],'fpr':float(np.mean(flags))})
    return {'decision':decision,'draws':cfg['bootstrap_draws'],'thresholds':thresholds,'strata':results,
            'healthy_failures':sorted(healthy_flags,key=lambda x:x['fpr'],reverse=True),
            'holdout_rows':rows,'support_sufficient':sufficient,'conditional_ap_positive':conditional}


if __name__=='__main__':
    import cv2
    import scipy

    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('fit','holdout'))
    for option in ('root','local','evidence'):
        p.add_argument('--'+option,type=Path,required=True)
    args=p.parse_args()
    cv2.setNumThreads(1)
    cfg=read(CONFIG)
    require(cfg==build(),'Frozen protocol or data partitions differ')
    initial=time.perf_counter()
    if args.phase=='fit':
        fit(args.root,args.local,cfg)
    else:
        finish(args.root,args.local,cfg,args.evidence)
    write(args.evidence/f'{args.phase}-runtime.json',
          {'phase':args.phase,'seconds':time.perf_counter()-initial,
           'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'python':platform.python_version(),
           'numpy':np.__version__,'scipy':scipy.__version__,'opencv':cv2.__version__,
           'opencv_threads':cv2.getNumThreads(),'cuda':'not imported','exit':0})
