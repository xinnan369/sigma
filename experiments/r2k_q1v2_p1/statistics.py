"""Outcome rules and paired group statistics. No image or inference interface."""
import numpy as np

from experiments.r2k_q1v2_p1.admission import ARMS, require

DRAWS = 10000
SEED = 3407


def quantities(hist):
    neg,pos=hist[...,0,:],hist[...,1,:]
    p,n=pos.sum(-1),neg.sum(-1)
    tp=np.cumsum(pos[...,::-1],axis=-1)
    seen=np.cumsum((pos+neg)[...,::-1],axis=-1)
    score=np.arange(256)/255
    with np.errstate(divide='ignore',invalid='ignore'):
        precision=np.divide(tp,seen,out=np.zeros_like(tp,dtype=float),where=seen>0)
        ap=(precision*pos[...,::-1]).sum(-1)/p
        brier_pos=pos@((1-score)**2)
        brier_neg=neg@(score**2)
        return {'ap':ap,'balanced_brier':.5*(brier_pos/p+brier_neg/n),
                'brier':(brier_pos+brier_neg)/(p+n),'prevalence':p/(p+n)}


def summarize(x):
    valid=x[1:][np.isfinite(x[1:])]
    return {'point':float(x[0]) if np.isfinite(x[0]) else None,
            'ci95':np.percentile(valid,[2.5,97.5]).tolist() if len(valid) else None,'valid_draws':len(valid)}


def fraction(a,b):
    return float(a/b) if b else None


def decision(support, ap, brier, guards, integrity):
    if not integrity:
        return 'FAIL'
    if any(support[k]<v for k,v in {'groups':40,'day':15,'night':15,'blur_positive':15}.items()):
        return 'INCONCLUSIVE'
    if any(g is False for g in guards):
        return 'FAIL'
    if ap is None or brier is None or any(g is None for g in guards):
        return 'INCONCLUSIVE'
    if ap[1]<0 or brier[0]>0:
        return 'FAIL'
    if ap[0]>0 and brier[1]<0:
        return 'PASS'
    return 'INCONCLUSIVE'


def views(row, a):
    v=row['scores'][a]
    require(set(v)=={'healthy','blur0','blur1'}, 'Missing frozen view')
    require(all(z.dtype==np.uint8 and z.shape==row['low_texture'].shape for z in v.values()),
            'Scores must be aligned uint8 bins')
    return v


def summarize_pairs(rows, thresholds):
    """One healthy and two paired blur views per original. P1 invokes this on fixtures only."""
    require(rows and set(thresholds)==set(ARMS)
            and all(type(t) is int and 0<=t<=256 for t in thresholds.values()), 'Invalid paired input')
    groups=sorted({r['group'] for r in rows})
    lookup={g:i for i,g in enumerate(groups)}
    n=len(groups)
    sections=('all','day','night','tier0','tier1','low_texture')
    hist={s:{a:np.zeros((n,2,256),np.int64) for a in ARMS} for s in sections}
    # Healthy rows are counted once per original, independent of blur tiers.
    healthy={a:np.zeros((n,4,2),np.int64) for a in ARMS}
    flags={a:np.zeros(2,np.int64) for a in ARMS}
    support={'groups':n,'day':0,'night':0,'blur_positive':0}
    positive=set()
    thermal_delta=[]
    rgb_identity=True
    require(len({r['id'] for r in rows})==len(rows), 'Duplicate original')
    for r in rows:
        i=lookup[r['group']]
        t=r['illumination']
        low=r['low_texture']
        require(t in ('day','night') and low.dtype==bool and low.size>0 and set(r['scores'])==set(ARMS),
                'Invalid stratum or arm roster')
        require(set(r['targets'])=={'blur0','blur1'}
                and all(y.dtype==bool and y.shape==low.shape for y in r['targets'].values()),
                'Blur targets differ from paired grid')
        if any(y.any() for y in r['targets'].values()):
            positive.add(r['group'])
        for a in ARMS:
            v=views(r,a)
            mark=r['observable'][a]
            require(set(mark)==set(v) and all(m.dtype==bool and m.shape==low.shape for m in mark.values()),
                    'Observability flags must align')
            flags[a]+=[sum(int(m.sum()) for m in mark.values()),sum(m.size for m in mark.values())]
            flagged=v['healthy'].astype(np.int16)>=thresholds[a]
            counts=(low.size,int(flagged.sum()))
            healthy[a][i,0]+=counts
            healthy[a][i,1 if t=='day' else 2]+=counts
            healthy[a][i,3]+=[int(low.sum()),int(flagged[low].sum())]
            for tier in (0,1):
                key=f'blur{tier}'
                y=r['targets'][key]
                if a=='thermal':
                    thermal_delta.append((v[key].astype(np.int16)-v['healthy'])[y])
                if a=='rgb':
                    rgb_identity=rgb_identity and np.array_equal(v[key],v['healthy'])
                encoded=256*y.astype(np.int32)+v[key]
                table=np.bincount(encoded.ravel(),minlength=512).reshape(2,256)
                for s in ('all',t,f'tier{tier}'):
                    hist[s][a][i]+=table
                hist['low_texture'][a][i]+=np.bincount(encoded[low],minlength=512).reshape(2,256)
    for t in ('day','night'):
        support[t]=len({r['group'] for r in rows if r['illumination']==t})
    support['blur_positive']=len(positive)
    weights=np.vstack([np.ones(n),np.random.Generator(np.random.PCG64(SEED)).multinomial(
        n,np.full(n,1/n),size=DRAWS)])
    primary={}
    bootstrap={}
    for a,h in hist['all'].items():
        bootstrap[a]=quantities((weights@h.reshape(n,512)).reshape(-1,2,256))
        primary[a]={k:summarize(v) for k,v in bootstrap[a].items()}
    delta={k:summarize(bootstrap['thermal'][k]-bootstrap['rgb'][k]) for k in ('ap','balanced_brier')}
    operators={}
    strata={}
    for a in ARMS:
        th=thresholds[a]
        operators[a]={'healthy_fpr':{name:fraction(k[1],k[0]) for name,k in zip(
                      ('all','day','night','low_texture'),healthy[a].sum(0),strict=True)},
                      'observability_coverage':fraction(*flags[a]),'retained_coverage':1.,
                      'group_counts':{g:{'healthy_total':int(healthy[a][i,0,0]),
                                         'healthy_false_positives':int(healthy[a][i,0,1]),
                                         'blur_positives':int(hist['all'][a][i,1].sum()),
                                         'blur_true_positives':int(hist['all'][a][i,1,th:].sum()),
                                         'blur_false_negatives':int(hist['all'][a][i,1,:th].sum()),
                                         'blur_false_positives':int(hist['all'][a][i,0,th:].sum()),
                                         'blur_true_negatives':int(hist['all'][a][i,0,:th].sum())}
                                      for i,g in enumerate(groups)}}
        strata[a]={}
        for section in sections:
            h=hist[section][a].sum(0)
            q=quantities(h[None,...])
            strata[a][section]={k:float(v[0]) if np.isfinite(v[0]) else None for k,v in q.items()}
            strata[a][section]['recall']=fraction(h[1,th:].sum(),h[1].sum())
    median=np.concatenate(thermal_delta)
    direction=float(np.median(median))/255 if len(median) else None
    fpr=operators['thermal']['healthy_fpr']
    recall=strata['thermal']['all']['recall']
    def within(value, bound):
        return None if value is None else value<=bound
    guards=[within(fpr['all'],.07),*(within(fpr[k],.10) for k in ('day','night','low_texture')),
            None if recall is None else recall>=.20, None if direction is None else direction>0]
    return {'decision':decision(support,delta['ap']['ci95'],delta['balanced_brier']['ci95'],guards,rgb_identity),
            'support':support,'draws':DRAWS,'seed':SEED,'primary':primary,'thermal_minus_rgb':delta,
            'operators':operators,'strata':strata,'median_roi_bad_score_change':direction,
            'rgb_paired_identity':rgb_identity}
