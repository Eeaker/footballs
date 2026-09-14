"""Checkpoint-specific RGB crop extraction and versioned feature caches."""
import copy
import csv
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def require_bf16(device):
    """Production ReID never silently falls back to FP32/FP16."""
    import torch
    if device != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('ReID inference requires a CUDA GPU; CPU inference is disabled by the BF16 production policy.')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('This GPU does not support CUDA BF16. Use an Ampere-or-newer NVIDIA GPU.')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def backend():
    sys.path.insert(0,str(ROOT/'reid_model'))
    import train_adapt
    return train_adapt


def load_model(config,weight,device):
    import torch
    model=backend().model_for(config,pretrained=False)
    model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
    return model.to(device).eval()


def build_reference_bank(paths,model_config,out,device='cpu',batch_size=8):
    """Frozen training-gallery prototypes, never query-track IDs or held-out images."""
    from selftraining import load_manifest,validate_dataset
    import torch
    require_bf16(device)
    rows=load_manifest(paths['manifest'])
    if not paths.get('allow_new_match_references'):validate_dataset(paths,rows)
    gallery=[r for r in rows if r['split']=='train' and r['enrolment_gallery']]
    holdout=not paths.get('allow_new_match_references')
    if holdout and (not gallery or max(r['frame'] for r in gallery)>1049):raise ValueError('Invalid training gallery')
    if not gallery:raise ValueError('Invalid training gallery')
    c=dict(model_config,eval_batch=batch_size,workers=0)
    model=load_model(c,paths['reid_weights'],device)
    try:features=backend().encode(model,gallery,c,device)
    finally:
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    pids=sorted({r['pid'] for r in gallery})
    prototypes=np.array([features[[i for i,r in enumerate(gallery) if r['pid']==pid]].mean(0) for pid in pids])
    prototypes/=np.maximum(np.linalg.norm(prototypes,axis=1,keepdims=True),1e-9)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez(out,prototypes=prototypes,checkpoint=sha(paths['reid_weights']),manifest=sha(paths['manifest']),
             gallery_count=len(gallery),max_frame=max(r['frame'] for r in gallery))
    return out


def restrict_audit(audit,windows):
    if windows is None:return copy.deepcopy(audit)
    def allowed(f):return any(lo<=f<=hi for lo,hi in windows)
    d=copy.deepcopy(audit);ns=[]
    for n in d['nodes']:
        n['rows']=[r for r in n['rows'] if allowed(r[0])]
        if not n['rows']:continue
        n['start'],n['end']=n['rows'][0][0],n['rows'][-1][0]
        keep={r[0] for r in n['rows']}
        n['positions']={f:xy for f,xy in n['positions'].items() if f in keep}
        n['samples']=[s for s in n.get('samples',[]) if s[0] in keep];ns.append(n)
    d['nodes']=ns;d['rows']=[r for r in d['rows'] if allowed(r[0])]
    d['selected']=[]
    return d


def overlap(row,others):
    _,local,x,y,w,h,*_=row;best=0.
    for other in others:
        if other[1]==local:continue
        _,_,xx,yy,ww,hh,*_=other
        area=max(0,min(x+w,xx+ww)-max(x,xx))*max(0,min(y+h,yy+hh)-max(y,yy))
        best=max(best,area/max(w*h,1e-9))
    return best


def select_observations(node,by_frame,quality):
    """Training retains endpoint guards; inference may recover clean short-track crops."""
    if node.get('quarantine') or node.get('exclusion_reason') or node.get('training_allowed') is False:return []
    pool=[];selected={};guard=quality.get('endpoint_guard_frames',3)
    for r in node['rows']:
        f,local,x,y,w,h,confidence=r;ov=overlap(r,by_frame[f])
        if confidence<quality.get('confidence',.55) or ov>quality.get('max_overlap',.15) or w<16 or h<40:continue
        if min(f-node['start'],node['end']-f)<guard:continue
        score=confidence*(1-ov)*min(1,h/100);pool.append((score,r,ov))
        key=f//quality.get('sample_every_frames',15)
        if key not in selected or score>selected[key][0]:selected[key]=(score,r,ov)
    result={p[1][0]:p for p in selected.values()}
    if node['end']-node['start']+1<=quality.get('short_track_frames',0) and pool:
        target=min(quality.get('short_track_samples',4),len(pool))
        for frame in np.linspace(pool[0][1][0],pool[-1][1][0],target):
            item=min(pool,key=lambda p:(abs(p[1][0]-frame),-p[0]));result[item[1][0]]=item
    return [result[f] for f in sorted(result)]


def apply_observation_policy(audit,policy):
    """Apply reviewed observation exclusions without injecting identity labels."""
    excluded={(r['source_local_id'],f) for r in policy['invalid_detections'] for f in r['frames']}
    for n in audit['nodes']:
        hits=[r for r in n['rows'] if (n['source_local'],r[0]) in excluded]
        if not hits:continue
        if len(hits)!=len(n['rows']):raise ValueError('Partial invalid node requires explicit splitting before association')
        n.update(quarantine=True,invalid_detection=True,training_allowed=False,
                 exclusion_reason='reviewed_mixed_person_box',samples=[],positions={})
    return audit


def extract_features(audit_path,video,weight,model_config,out,quality,device='cpu',batch_size=8,windows=None):
    import torch
    from PIL import Image
    require_bf16(device)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fingerprint=dict(version=3,audit=sha(audit_path),video=sha(video),checkpoint=sha(weight),
                     model_config=model_config,quality=quality,windows=windows)
    cache=out/'features.pkl';metadata=out/'feature_metadata.json'
    if cache.exists() and metadata.exists() and json.loads(metadata.read_text())['fingerprint']==fingerprint:return cache
    if cache.exists():raise ValueError('Feature cache provenance changed; use a fresh run directory')
    with Path(audit_path).open('rb') as f:audit=restrict_audit(pickle.load(f),windows)
    policy=quality.get('observation_policy')
    if policy:
        if policy['audit_sha256']!=fingerprint['audit']:raise ValueError('Observation policy belongs to a different audit')
        apply_observation_policy(audit,policy)
    by_frame=defaultdict(list)
    for n in audit['nodes']:
        if n.get('invalid_detection'):continue
        for r in n['rows']:by_frame[r[0]].append(r)
    requested=defaultdict(list)
    for n in audit['nodes']:
        n['samples']=[]
        if n.get('quarantine'):continue
        for _,r,ov in select_observations(n,by_frame,quality):requested[r[0]].append((n,r,ov))
    model=load_model(model_config,weight,device);transform=backend().Crops([],model_config,False).transform
    pending=[];manifest=[];dimension=None
    def flush():
        nonlocal dimension
        if not pending:return
        with torch.inference_mode():
            batch=torch.stack([p[0] for p in pending]).to(device)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=True):
                encoded=model(batch).float()
            encoded=torch.nn.functional.normalize(encoded,dim=1).cpu().numpy()
        if encoded.shape[1]!=3840 or not np.isfinite(encoded).all():raise ValueError('Invalid TransReID embeddings')
        dimension=encoded.shape[1]
        for feature,(_,n,f,colour,record) in zip(encoded,pending):
            n['samples'].append((f,feature,colour));manifest.append(record)
        pending.clear()
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise ValueError('Cannot open input video')
    if cap.get(cv2.CAP_PROP_FPS)<=0:
        cap.release();raise ValueError('Input video has invalid FPS metadata')
    width,height=int(cap.get(3)),int(cap.get(4))
    try:
        for f in range(max(requested,default=-1)+1):
            ok,im=cap.read()
            if not ok:raise ValueError('Video ended before requested frame')
            for n,r,ov in requested[f]:
                _,local,x,y,w,h,confidence=r
                if x<2 or y<2 or x+w>width-2 or y+h>height-2:continue
                crop=im[int(y):int(y+h),int(x):int(x+w)]
                path=out/'crops'/f'n{n["id"]}_f{f}.jpg';path.parent.mkdir(exist_ok=True)
                ok,jpeg=cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok;jpeg.tofile(str(path))
                # Encode the same JPEG that later enters self-training, not a different raw crop.
                rgb=Image.open(path).convert('RGB');tensor=transform(rgb)
                hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
                torso=hsv[max(0,int(h*.15)):max(1,int(h*.6)),int(w*.2):max(int(w*.2)+1,int(w*.8))]
                colour=cv2.calcHist([torso],[0,1],None,[16,8],[0,180,0,256]).reshape(-1)
                colour=colour/max(np.linalg.norm(colour),1e-9)
                record=dict(node=n['id'],algorithm_id=n['output_id'],source_local_id=local,frame=f,
                            kit=n.get('kit',-1),overlap=ov,confidence=confidence,image=str(path.resolve()),sha256=sha(path))
                pending.append((tensor,n,f,colour,record))
                if len(pending)>=batch_size:flush()
        flush()
    finally:
        cap.release();del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    if dimension is None:raise ValueError('No eligible image crops')
    info=dict(fingerprint=fingerprint,dimension=dimension,crops=len(manifest),encoder='TransReID',windows=windows)
    audit['feature_metadata']=info
    with cache.open('wb') as f:pickle.dump(audit,f)
    metadata.write_text(json.dumps(info,indent=2),encoding='utf-8')
    with (out/'crops.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(manifest[0]));writer.writeheader();writer.writerows(manifest)
    return cache
