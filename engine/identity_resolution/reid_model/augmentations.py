"""Native-resolution appearance transforms. Fragment views copy detector artifacts, not identity labels."""
import random
import cv2
import numpy as np
from PIL import Image

FRAGMENT_VIEWS=('original','light','heavy')


class RandomNativeDegradation:
    def __init__(self,motion_probability=.2,downsample_probability=.1):
        if min(motion_probability,downsample_probability)<0 or motion_probability+downsample_probability>1:
            raise ValueError('Invalid degradation probabilities')
        self.motion_probability=motion_probability;self.downsample_probability=downsample_probability

    def __call__(self,image):
        choice=random.random()
        if choice<self.motion_probability:
            limit=min(7,max(1,int(min(image.size)/8)))
            sizes=[k for k in (3,5,7) if k<=limit]
            if not sizes:return image
            k=random.choice(sizes);kernel=np.zeros((k,k),np.float32);center=(k-1)/2
            angle=random.uniform(0,np.pi);dx=np.cos(angle)*center;dy=np.sin(angle)*center
            cv2.line(kernel,(round(center-dx),round(center-dy)),(round(center+dx),round(center+dy)),1.,1)
            kernel/=kernel.sum()
            return Image.fromarray(cv2.filter2D(np.asarray(image),-1,kernel,borderType=cv2.BORDER_REFLECT_101))
        if choice<self.motion_probability+self.downsample_probability:
            scale=random.uniform(.65,.9);w,h=image.size
            return image.resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.BILINEAR).resize((w,h),Image.Resampling.BICUBIC)
        return image


def _rgb(image):
    return np.asarray(image.convert('RGB'))


def _pil(arr):
    return Image.fromarray(np.clip(arr,0,255).astype(np.uint8),'RGB')


def _jpeg(arr,quality,rng):
    quality=int(np.clip(quality,25,95))
    ok,buf=cv2.imencode('.jpg',cv2.cvtColor(arr,cv2.COLOR_RGB2BGR),[int(cv2.IMWRITE_JPEG_QUALITY),quality])
    if not ok:return arr
    decoded=cv2.imdecode(buf,cv2.IMREAD_COLOR)
    if decoded is None:return arr
    return cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB)


def _motion_blur(arr,rng):
    h,w=arr.shape[:2]
    limit=min(7,max(1,int(min(h,w)/8)))
    sizes=[k for k in (3,5,7) if k<=limit]
    if not sizes:return arr
    k=rng.choice(sizes);kernel=np.zeros((k,k),np.float32);center=(k-1)/2
    angle=rng.uniform(0,np.pi);dx=np.cos(angle)*center;dy=np.sin(angle)*center
    cv2.line(kernel,(round(center-dx),round(center-dy)),(round(center+dx),round(center+dy)),1.,1)
    kernel/=max(float(kernel.sum()),1e-6)
    return cv2.filter2D(arr,-1,kernel,borderType=cv2.BORDER_REFLECT_101)


def _downsample(arr,scale):
    h,w=arr.shape[:2]
    small=cv2.resize(arr,(max(1,int(round(w*scale))),max(1,int(round(h*scale)))),interpolation=cv2.INTER_AREA)
    return cv2.resize(small,(w,h),interpolation=cv2.INTER_LINEAR)


def _partial_body(arr,top,bottom,left,right):
    h,w=arr.shape[:2]
    y0=int(round(h*top));y1=int(round(h*(1-bottom)))
    x0=int(round(w*left));x1=int(round(w*(1-right)))
    y1=max(y0+1,y1);x1=max(x0+1,x1)
    crop=arr[y0:y1,x0:x1]
    if crop.size==0:return arr
    return cv2.resize(crop,(w,h),interpolation=cv2.INTER_LINEAR)


def _fatter_box(arr,pad_frac):
    h,w=arr.shape[:2]
    pad=max(1,int(round(w*pad_frac)))
    padded=cv2.copyMakeBorder(arr,0,0,pad,pad,cv2.BORDER_REPLICATE)
    return cv2.resize(padded,(w,h),interpolation=cv2.INTER_LINEAR)


def _wash(arr,contrast,brightness,green):
    x=arr.astype(np.float32)
    x=(x-127.5)*contrast+127.5+brightness
    x[:,:,1]=x[:,:,1]+green
    return np.clip(x,0,255).astype(np.uint8)


def _other_edge(arr,other,rng):
    if other is None:return arr
    overlay=cv2.resize(_rgb(other),(arr.shape[1],arr.shape[0]),interpolation=cv2.INTER_LINEAR)
    bw=max(2,int(round(arr.shape[1]*rng.uniform(.12,.30))))
    alpha=rng.uniform(.4,.75)
    out=arr.copy()
    strip=overlay[:,:bw] if rng.random()<0.5 else overlay[:,-bw:]
    if rng.random()<0.5:
        out[:,:bw]=(alpha*strip+(1-alpha)*out[:,:bw]).astype(np.uint8)
    else:
        out[:,-bw:]=(alpha*strip+(1-alpha)*out[:,-bw:]).astype(np.uint8)
    return out


def apply_fragment_view(image,view,rng=None,other=None):
    """Detector-like box jitter and a real other-person edge. Original is identity."""
    if view not in FRAGMENT_VIEWS:raise ValueError(f'Unknown fragment view {view}')
    if view=='original':return image
    rng=rng or random
    arr=_rgb(image)
    if min(arr.shape[:2])<8:return image
    if view=='light':
        cut=rng.uniform(.06,.14)
        side=rng.choice(('top','bottom','left','right'))
        kwargs=dict(top=0,bottom=0,left=0,right=0);kwargs[side]=cut
        arr=_partial_body(arr,**kwargs)
        if rng.random()<0.5:arr=_fatter_box(arr,rng.uniform(.04,.12))
        if rng.random()<0.4:arr=_other_edge(arr,other,rng)
        return _pil(arr)
    # heavy: incomplete detector box, wider box, another player on the edge. No blur-as-identity.
    if rng.random()<0.5:
        arr=_partial_body(arr,top=0,bottom=rng.uniform(.16,.34),left=rng.uniform(0,.08),right=rng.uniform(0,.08))
    else:
        arr=_partial_body(arr,top=rng.uniform(.10,.20),bottom=rng.uniform(.08,.20),left=rng.uniform(0,.10),right=rng.uniform(0,.10))
    arr=_fatter_box(arr,rng.uniform(.10,.24))
    arr=_other_edge(arr,other,rng)
    return _pil(arr)


def view_schedule(k,step=0):
    """At least half original. Remaining slots rotate light/heavy so both appear across steps."""
    if k<1:raise ValueError('k must be positive')
    n_orig=max(1,(k+1)//2)
    rest=[('light','heavy')[(i+step)%2] for i in range(k-n_orig)]
    return ['original']*n_orig+rest


def expand_train_views(rows,config):
    """Train rows stay one copy per crop. Human identity groups never enter training."""
    train=[r for r in rows if r.get('split')=='train']
    appearance=config.get('fragment_appearance') if config else None
    if not appearance:return train
    views=tuple(appearance.get('views',FRAGMENT_VIEWS))
    if 'original' not in views:
        raise ValueError('fragment_appearance must keep original views')
    allowed={None,'','algorithm_seed_only'}
    if any(r.get('label_source') not in allowed for r in train):
        raise ValueError('Human correspondence entered training')
    return train
