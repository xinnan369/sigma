"""Point-aligned descriptors and equal-size logistic quality heads."""
import hashlib

import numpy as np


def rng(seed, *keys):
    h=hashlib.sha256('/'.join(map(str,(seed,*keys))).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:16],'big'))


def grid(image):
    return np.s_[4:image.shape[0]:8,4:image.shape[1]:8]


def describe(image):
    import cv2

    x=image.astype(np.float32)/255
    mean=cv2.blur(x,(9,9))
    variance=np.maximum(0,cv2.blur(x*x,(9,9))-mean*mean)
    kernel=np.ones((9,9),np.uint8)
    span=cv2.dilate(x,kernel)-cv2.erode(x,kernel)
    gradient=cv2.blur((abs(cv2.Sobel(x,-1,1,0))+abs(cv2.Sobel(x,-1,0,1)))/8,(9,9))
    sat=cv2.blur(((x<.01)|(x>.99)).astype(np.float32),(9,9))
    return np.stack([v[grid(image)] for v in (mean,variance,span,gradient,sat,abs(x-mean))],axis=-1)


def degrade(image, family, cfg, sample, tier):
    import cv2

    gen=rng(cfg['seed'],sample,family,tier)
    h,w=image.shape
    size=cfg['roi']['size']
    m=cfg['roi']['margin']
    y,x=int(gen.integers(m,h-size-m+1)),int(gen.integers(m,w-size-m+1))
    box=np.s_[y:y+size,x:x+size]
    mask=np.zeros_like(image,bool)
    mask[box]=True
    out=image.copy()
    if family=='occlusion':
        out[box]=0
    elif family=='gaussian':
        sigma=cfg['gaussian_sigmas'][tier]
        out[box]=np.clip(np.rint(image[box]+gen.normal(0,sigma,(size,size))),0,255).astype(np.uint8)
    elif family=='blur':
        k,sigma=cfg['blur']['tiers'][tier]
        out[box]=cv2.GaussianBlur(image,(k,k),sigma)[box]
    else:
        raise ValueError('Unknown corruption')
    interior=np.zeros_like(mask)
    margin=12
    interior[y+margin:y+size-margin,x+margin:x+size-margin]=True
    quadrant=int(y+size/2>=h/2)*2+int(x+size/2>=w/2)
    return out,mask[grid(image)],interior[grid(image)],quadrant


def logistic(x, y, cfg):
    from scipy.optimize import minimize
    from scipy.special import expit

    def objective(w):
        z=x@w[:-1]+w[-1]
        loss=np.mean(np.logaddexp(0,z)-y*z)
        error=expit(z)-y
        l2=cfg['optimizer']['l2']
        return loss+l2*(w[:-1]@w[:-1])/2, np.r_[x.T@error/len(y)+l2*w[:-1],error.mean()]

    r=minimize(objective,np.zeros(x.shape[1]+1),jac=True,method='L-BFGS-B',
               options={'maxiter':cfg['optimizer']['maxiter'],'ftol':1e-12,'gtol':1e-8})
    if not r.success:
        raise RuntimeError(f'Logistic fit failed: {r.message}')
    return r.x, int(r.nit)


def logits(features, head):
    return ((features-head['mean'])/head['scale'])@head['weights'][:-1]+head['weights'][-1]


def probabilities(z, calibration):
    from scipy.special import expit

    return np.rint(255*expit(calibration[0]*z+calibration[1])).astype(np.uint8)


def predict(image, head, calibration):
    return probabilities(logits(describe(image),head),calibration)


def operating_threshold(healthy_bins, target):
    counts=np.bincount(healthy_bins.ravel(),minlength=256)
    tail=np.r_[np.cumsum(counts[::-1])[::-1],0]
    return int(np.flatnonzero(tail <= target*counts.sum())[0])
