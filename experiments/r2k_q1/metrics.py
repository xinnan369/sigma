"""Exact scores of the frozen 256-level probabilities with shared group draws."""
import numpy as np


def metrics(h):
    h=np.asarray(h,dtype=np.float64)
    neg,pos=h[...,0,:],h[...,1,:]
    n,p=neg.sum(-1),pos.sum(-1)
    total=p+n
    tp=np.cumsum(pos[...,::-1],axis=-1)
    fp=np.cumsum(neg[...,::-1],axis=-1)
    result={}
    with np.errstate(divide='ignore',invalid='ignore'):
        precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=tp+fp>0)
        result['ap']=(precision*pos[...,::-1]).sum(-1)/p
        result['auc']=((tp-pos[...,::-1]/2)*neg[...,::-1]).sum(-1)/(p*n)
        score=np.arange(256)/255
        bn=(neg*(score**2)).sum(-1)
        bp=(pos*((1-score)**2)).sum(-1)
        result['brier']=(bn+bp)/total
        result['balanced_brier']=.5*(bn/n+bp/p)
        result['positive_fraction']=p/total
        edges=np.minimum(9,np.floor(score*10)).astype(int)
        result['ece']=sum(abs((pos[...,edges==b]-(pos+neg)[...,edges==b]*score[edges==b]).sum(-1))
                          for b in range(10))/total
    return result


def curve(h):
    score=np.arange(256)/255
    edges=np.minimum(9,np.floor(score*10)).astype(int)
    out=[]
    for b in range(10):
        subset=h[:,edges==b]
        n=int(subset.sum())
        if n:
            out.append({'bin':b,'count':n,'predicted':float(subset.sum(0)@score[edges==b]/n),
                        'observed':float(subset[1].sum()/n)})
    return out


def distribution(target,bins):
    return np.bincount(256*target.astype(np.int64).ravel()+bins.ravel(),minlength=512).reshape(2,256)


def bootstrap(weights, group_hist, thresholds):
    results={}
    for arm, groups in group_hist.items():
        h=weights@np.asarray(groups).reshape(len(groups),-1)
        h=h.reshape(len(weights),2,256)
        m=metrics(h)
        with np.errstate(divide='ignore',invalid='ignore'):
            m['recall']=h[:,1,thresholds[arm]:].sum(-1)/h[:,1].sum(-1)
            m['fpr']=h[:,0,thresholds[arm]:].sum(-1)/h[:,0].sum(-1)
        results[arm]=m
    return results


def summary(draws):
    sample=np.asarray(draws[1:])
    valid=sample[np.isfinite(sample)]
    return {'point':float(draws[0]) if np.isfinite(draws[0]) else None,
            'ci95':np.percentile(valid,[2.5,97.5]).tolist() if len(valid) else None,
            'valid_draws':int(len(valid))}
