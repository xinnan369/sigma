"""Fixed-scale point-aligned descriptors from one current grayscale frame."""
import numpy as np

from experiments.r2k_q1.model import grid


def describe_multiscale(image, cfg, ablated=False):
    import cv2

    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError('Expected single-frame grayscale uint8 image')
    x=image.astype(np.float32)/255
    fine,mid,coarse=cfg['scales']
    eps=cfg['epsilon']

    def smooth(a,k):
        return cv2.blur(a,(k,k),borderType=cv2.BORDER_REFLECT_101)

    def grad(a):
        return (abs(cv2.Sobel(a,-1,1,0))+abs(cv2.Sobel(a,-1,0,1)))/8

    f,m,c=(smooth(x,k) for k in (fine,mid,coarse))
    fine_edge=grad(f)
    g,h=smooth(fine_edge,mid),smooth(grad(c),mid)
    high=smooth((x-f)**2,mid)
    band=smooth((f-m)**2,mid)
    mean_sq=smooth(x*x,mid)
    saturation=smooth(((x<.01)|(x>.99)).astype(np.float32),mid)
    fill=smooth((image==0).astype(np.float32),mid)
    grad_ratio=np.log((g+eps)/(h+eps))
    band_ratio=np.log((high+eps**2)/(band+eps**2))
    # Inverse peak-to-local edge-strength ratio is a width proxy, not physical PSF.
    edge_width=g/(cv2.dilate(fine_edge,np.ones((mid,mid),np.uint8))+eps)
    head=(m,np.maximum(0,mean_sq-m*m)) if ablated else (grad_ratio,band_ratio)
    tensor=np.stack([v[grid(x)] for v in (*head,edge_width,h,saturation,fill)],axis=-1)
    return tensor, h[grid(x)]>=cfg['minimum_coarse_gradient']
