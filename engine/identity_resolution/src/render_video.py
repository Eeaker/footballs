"""Render standard one-based MOT frames; preserve duration when subsampling."""
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np


def load_tracking_data(path):
    frames=defaultdict(list);seen=set()
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            r=line.split(',');frame,identity=int(r[0]),int(r[1])
            if frame<1:raise ValueError('MOT frame numbers must be one-based')
            if (frame,identity) in seen:raise ValueError('Duplicate frame/identity')
            seen.add((frame,identity));frames[frame].append((identity,*map(float,r[2:7])))
    return frames


def render_video_func(video_path,tracking_path,output_path,fps=30,sample_rate=1,preview_only=False,num_samples=10,
                      quarantine_ids=(),show_confidence=True,show_legend=True,source_ids=None,review_ids=()):
    if sample_rate<1:raise ValueError('sample_rate must be positive')
    frames=load_tracking_data(tracking_path);cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened():raise ValueError('Cannot open video')
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));source_fps=cap.get(cv2.CAP_PROP_FPS)
    if frames and max(frames)>total:raise ValueError('MOT frame exceeds video length')
    if abs(fps-source_fps)>.1:raise ValueError('Configured FPS disagrees with source video')
    out=Path(output_path);out.parent.mkdir(parents=True,exist_ok=True)
    previews=set(np.linspace(0,total-1,min(num_samples,total)).astype(int))
    writer=None
    if not preview_only:
        writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),source_fps/sample_rate,(int(cap.get(3)),int(cap.get(4))))
        if not writer.isOpened():raise ValueError('Cannot open output writer')
    written=0
    try:
        for f in range(total):
            ok,im=cap.read()
            if not ok:raise ValueError('Truncated video')
            if f%sample_rate and f not in previews:continue
            for identity,x,y,w,h,conf in frames.get(f+1,[]):
                q=identity in quarantine_ids
                color=(145,145,145) if q else tuple(map(int,np.random.default_rng(identity).integers(60,240,3)))
                label='OCC - no sample' if q else f'ID {identity}'
                source=(source_ids or {}).get((f,identity),identity)
                if not q and source!=identity:label+=f' <- {source}'
                if not q and source in review_ids:label+=' REVIEW';color=(0,165,255)
                if show_confidence and not q:label+=f' {conf:.2f}'
                cv2.rectangle(im,(round(x),round(y)),(round(x+w),round(y+h)),color,2)
                cv2.putText(im,label,(round(x),max(20,round(y)-4)),0,.52,color,2)
            cv2.putText(im,f'clip {f} | MOT {f+1}',(12,28),0,.7,(255,255,255),2)
            if show_legend:cv2.putText(im,'Algorithm IDs <- source IDs; grey OCC; orange REVIEW',(12,55),0,.6,(255,255,255),2)
            if f in previews:
                folder=out.parent/'samples';folder.mkdir(exist_ok=True)
                ok,b=cv2.imencode('.jpg',im);assert ok;b.tofile(str(folder/f'frame_{f:04d}.jpg'))
            if writer and f%sample_rate==0:writer.write(im);written+=1
    finally:
        cap.release()
        if writer:writer.release()
    return dict(source_frames=total,rendered_frames=written,output_fps=source_fps/sample_rate)
