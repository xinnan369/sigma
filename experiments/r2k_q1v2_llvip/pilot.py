"""Frozen LLVIP exploration, with roster selection separated from evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from experiments.r2k_q1 import metrics, model
from experiments.r2k_q1v2.features import describe_multiscale
from .scan import dump, pair_inventory, sha

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/'configs/R2k_Q1v2_LLVIP_PILOT.json'
SCOPE = 'EXPLORATORY_LLVIP_PILOT'


def digest(path):
    return sha(path.read_bytes())


def read(path):
    return json.loads(path.read_bytes())


def source_digest(path):
    return sha(path.read_bytes().replace(b'\r\n',b'\n'))


def adapt(bgr):
    if bgr is None or bgr.dtype != np.uint8 or bgr.shape != (1024,1280,3):
        raise ValueError('Expected LLVIP uint8 BGR 1280x1024')
    crop = bgr[32:992,0:1280]
    small = cv2.resize(crop,(640,480),interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)


def chosen_roster(rows, seed, count):
    _, pairs = pair_inventory(rows,'test')
    ids = [i for i,(v,t) in pairs.items()
           if all((r['width'],r['height'],r['mode'],r['bits_per_sample'],r['format']) ==
                  (1280,1024,'RGB',8,'JPEG') for r in (v,t))]
    ids.sort(key=lambda i:(sha(f'{seed}/llvip-test/{i}'.encode()),i))
    if len(ids)<count:
        raise ValueError('Insufficient eligible paired images')
    return [{'id':i, **{k:{n:r[n] for n in ('path','sha256')}
                        for k,r in zip(('rgb','thermal'),pairs[i])}} for i in ids[:count]]


def transformed(root, row):
    b=(root/row['path']).read_bytes()
    if sha(b)!=row['sha256']:
        raise ValueError('Source changed: '+row['path'])
    return adapt(cv2.imdecode(np.frombuffer(b,np.uint8),
                               cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION))


def load_heads(config, q1_path, p0_path):
    paths = {'q1':q1_path,'p0':p0_path}
    assets = {}
    for name,path in paths.items():
        if digest(path)!=config['assets_sha256'][name]:
            raise ValueError(f'{name} frozen parameters differ')
        assets[name]=read(path)
    q1,p0=assets['q1'],assets['p0']
    for name,entry,key in [('q1',q1['signature'],'protocol_sha256'),('p0',p0,'protocol_sha256')]:
        if entry[key]!=config['parent_protocol_sha256'][name]:
            raise ValueError('Frozen protocol commitment differs')
    a=q1['calibration']['thermal']
    result={'q1':(q1['heads']['thermal'],a['calibration'],a['threshold_bin'])}
    for arm in ('thermal','rgb'):
        c=p0['calibration'][arm]
        result[arm]=(p0['heads'][arm],c['choices'][c['method']],c['threshold_bin'])
    return result


def predict(frame, arm, cfg, heads):
    head,cal,_=heads[arm]
    if arm=='q1':
        features=model.describe(frame)
    else:
        features,_=describe_multiscale(frame,cfg)
    if features.shape!=(60,80,6) or not np.isfinite(features).all():
        raise ValueError('Invalid descriptor grid')
    return model.probabilities(model.logits(features,head),cal)


def timed_predict(frame, arm, cfg, heads):
    start=time.perf_counter_ns()
    scores=predict(frame,arm,cfg,heads)
    return scores,time.perf_counter_ns()-start


def roi_coordinates(cfg,i,tier):
    gen=model.rng(cfg['seed'],i,'blur',tier)
    size,m=cfg['roi']['size'],cfg['roi']['margin']
    y,x=int(gen.integers(m,480-size-m+1)),int(gen.integers(m,640-size-m+1))
    return [x,y,x+size,y+size]


def evaluate(root, roster, old, p0, heads):
    result=[]
    for sample in roster:
        i=sample['id']
        rgb, thermal=(transformed(root,sample[k]) for k in ('rgb','thermal'))
        clean={a:timed_predict(rgb if a=='rgb' else thermal,a,p0,heads) for a in heads}
        healthy={a:pair[0] for a,pair in clean.items()}
        record={'id':i,'adapted_sha256':{'rgb':sha(rgb.tobytes()),'thermal':sha(thermal.tobytes())},
                'healthy_flagged':{a:int((p>=heads[a][2]).sum()) for a,p in healthy.items()},
                'clean_inference_ns':{a:pair[1] for a,pair in clean.items()},
                'healthy_points':int(thermal[model.grid(thermal)].size),'blur':[]}
        for tier in range(2):
            changed, mask, _, _=model.degrade(thermal,'blur',old,i,tier)
            x,y,x1,y1=roi_coordinates(old,i,tier)
            support=np.zeros_like(thermal,bool)
            support[y:y1,x:x1]=True
            if not np.array_equal(support[model.grid(thermal)],mask) or np.any(changed[~support]!=thermal[~support]):
                raise ValueError('ROI mapping or support mismatch')
            arms={}
            for a in heads:
                scores,latency=timed_predict(rgb if a=='rgb' else changed,a,p0,heads)
                if a=='rgb' and not np.array_equal(scores,healthy[a]):
                    raise ValueError('RGB-only prediction changed')
                difference=scores[mask].astype(int)-healthy[a][mask]
                arms[a]={'inference_ns':latency,'histogram':metrics.distribution(mask,scores).tolist(),
                         'delta_bins_sum':int(difference.sum()),
                         'delta_bins_histogram':np.bincount(difference+255,minlength=511).tolist(),
                         'changed_output_bins':int((scores!=healthy[a]).sum()),
                         'prediction_sha256':sha(scores.tobytes())}
            record['blur'].append({'tier':tier,'roi_xyxy': [x,y,x1,y1],
                                   'positive_points':int(mask.sum()),'arms':arms})
        result.append(record)
        print(f'PAIR {len(result)}/{len(roster)} {i}: clean, weak, strong PASS',flush=True)
    return result


def summarize(rows, heads):
    result={}
    for a,(_,_,threshold) in heads.items():
        out={}
        for label,tiers in [('all',(0,1)),('tier0',(0,)),('tier1',(1,))]:
            selected=[b['arms'][a] for r in rows for b in r['blur'] if b['tier'] in tiers]
            h=np.sum([b['histogram'] for b in selected],axis=0)
            delta=np.sum([b['delta_bins_histogram'] for b in selected],axis=0)
            cum=delta.cumsum()
            n=int(h[1].sum())
            assert int(delta.sum())==n
            med=(np.searchsorted(cum,(n+1)//2)+np.searchsorted(cum,n//2+1))/2-255
            m=metrics.metrics(h)
            out[label]={**{k:float(m[k]) for k in ('ap','brier','balanced_brier','positive_fraction')},
                       'roi_points':n,'points':int(h.sum()),
                       'roi_recall':float(h[1,threshold:].sum()/n),
                       'mean_roi_bad_probability_change':sum(b['delta_bins_sum'] for b in selected)/(255*n),
                       'median_roi_bad_probability_change':float(med/255),
                       'positive_difference_fraction':float(delta[256:].sum()/n),
                       'tied_difference_fraction':float(delta[255]/n),
                       'prediction_bins_changed':sum(b['changed_output_bins'] for b in selected)}
        result[a]={'threshold_bin':threshold,
                   'healthy_fpr':sum(r['healthy_flagged'][a] for r in rows)/sum(r['healthy_points'] for r in rows),
                   'blur':out}
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=('freeze-roster','evaluate'))
    for name in ('root','manifest','output','q1','p0'):
        p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    begin=time.perf_counter()
    cpu_begin=time.process_time()
    cfg=read(CONFIG)
    cv2.setNumThreads(1)
    if cv2.__version__ != cfg['opencv_version']:
        raise ValueError('OpenCV version differs from frozen pilot')
    if digest(a.manifest)!=cfg['manifest_sha256']:
        raise ValueError('Precheck manifest changed')
    if any(source_digest(ROOT/path)!=h for path,h in cfg['source_sha256_lf'].items()):
        raise ValueError('Execution source changed')
    roster=chosen_roster(read(a.manifest)['files'],cfg['seed'],cfg['pairs'])
    descriptor={'scope':SCOPE,'protocol_sha256':digest(CONFIG),'pairs':roster}
    roster_path=a.output/'roster.json'
    if a.stage=='freeze-roster':
        a.output.mkdir(parents=True,exist_ok=False)
        dump(roster_path,descriptor)
        return
    if read(roster_path)!=descriptor or (a.output/'results.json').exists():
        raise ValueError('Frozen roster differs or results already exist')
    paths={'q1':ROOT/'configs/R2k_Q1_protocol.json','p0':ROOT/'configs/R2k_Q1v2_protocol.json'}
    for key,path in paths.items():
        if digest(path)!=cfg['parent_protocol_sha256'][key]:
            raise ValueError('Parent protocol changed')
    old,p0=(read(paths[n]) for n in ('q1','p0'))
    heads=load_heads(cfg,a.q1,a.p0)
    print(SCOPE, 'preflight PASS: frozen roster, source, protocol, and parameter digests', flush=True)
    rows=evaluate(a.root,roster,old,p0,heads)
    with (a.output/'per-image.jsonl').open('x',encoding='utf-8',newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r,separators=(',',':'))+'\n')
    import resource  # Pilot runs under WSL/Linux for process peak RSS.
    latency={}
    for arm in heads:
        ns={'clean':[r['clean_inference_ns'][arm] for r in rows]}
        ns.update({label:[r['blur'][tier]['arms'][arm]['inference_ns'] for r in rows]
                   for tier,label in enumerate(('weak','strong'))})
        latency[arm]={k:{'median_ms':float(np.median(v)/1e6),
                         'p95_ms':float(np.percentile(v,95)/1e6)} for k,v in ns.items()}
    dump(a.output/'results.json',{'scope':SCOPE,'pairs':len(rows),'metrics':summarize(rows,heads),
                                  'roster_sha256':digest(roster_path),
                                  'protocol_sha256':digest(CONFIG),
                                  'per_image_sha256':digest(a.output/'per-image.jsonl'),
                                  'opencv':cv2.__version__,'numpy':np.__version__,
                                  'python':platform.python_version(),'platform':platform.platform(),
                                  'opencv_threads':cv2.getNumThreads(),
                                  'seconds':time.perf_counter()-begin,
                                  'process_cpu_seconds':time.process_time()-cpu_begin,
                                  'inference_latency':latency,
                                  'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    print(SCOPE, len(rows), 'pairs completed', flush=True)


if __name__=='__main__':
    main()
