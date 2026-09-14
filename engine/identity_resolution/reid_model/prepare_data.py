"""Build immutable time-disjoint, user-confirmed identity data from latest output."""
import csv
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
PURE=HERE.parent
WINDOWS={'train':(0,1049),'val':(1110,1379),'test':(1440,1799)}


def split_for(frame):
    return next((name for name,(lo,hi) in WINDOWS.items() if lo<=frame<=hi),None)


def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    out=HERE/'data';out.mkdir(exist_ok=True)
    audit_path=PURE/'最新纯净ID源码/output/refine/audit_data.pkl'
    dense_path=PURE/'长轨迹优化_20260908/fresh_conf_0.10_dense.pkl'
    d=pickle.load(audit_path.open('rb'));e=pickle.load(dense_path.open('rb'))
    original=pickle.load((PURE/'长轨迹优化_20260908/results_v3/audit_data.pkl').open('rb'))
    origin={(r[0],r[1]):n['output_id'] for n in original['nodes'] for r in n['rows']}
    groups=json.loads((HERE/'identity_groups.json').read_text(encoding='utf-8'))['groups']
    owner={i:(pid,name) for pid,(name,ids) in enumerate(groups.items()) for i in ids}
    assert len(owner)==sum(map(len,groups.values()))
    seen=set();counts=Counter();requested=defaultdict(list);kitvotes=defaultdict(Counter)
    for n in d['nodes']:
        if n['output_id'] not in owner:continue
        assert not n['quarantine'],'User identity unexpectedly names quarantine'
        pid,name=owner[n['output_id']]
        for r in n['rows']:
            f,local,x,y,w,h,conf=r
            assert (pid,f) not in seen,('same-frame identity collision',pid,f)
            seen.add((pid,f));counts['user_confirmed_observations']+=1
            split=split_for(f)
            if split is None:counts['embargo']+=1;continue
            ev=e[(f,local)]
            if conf<.55 or ev['overlap']>.15 or w<16 or h<40 or x<2 or y<2 or x+w>1918 or y+h>1078:
                counts['quality_rejected']+=1;continue
            # Appearance colours cannot reject a manually confirmed goalkeeper.
            if name!='goalkeeper' and n['kit']>=0 and ev['label']>=0 and ev['label']!=n['kit']:
                counts['colour_rejected']+=1;continue
            if min(f-n['start'],n['end']-f)<3:counts['endpoint_rejected']+=1;continue
            if n['kit']>=0:kitvotes[pid][n['kit']]+=1
            requested[f].append(dict(pid=pid,identity=name,algorithm_id=n['output_id'],original_v3_id=origin[(f,local)],
                source_local_id=local,node=n['id'],frame=f,source_frame=f+3150,split=split,
                x=x,y=y,w=w,h=h,confidence=conf,overlap=float(ev['overlap'])))
    # One quality-ranked native crop per identity per half second; no upsampled evidence.
    best={};cap=cv2.VideoCapture(str(PURE/'1分钟片段原视频.mp4'))
    for f in range(1800):
        ok,im=cap.read();assert ok
        for r in requested[f]:
            x,y,w,h=(r[k] for k in ['x','y','w','h']);crop=im[int(y):int(y+h),int(x):int(x+w)]
            sharp=float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
            quality=r['confidence']*(1-r['overlap'])*min(1,h/100)*(.5+.5*min(1,sharp/100))
            key=(r['pid'],r['split'],f//15)
            if key not in best or quality>best[key][0]:best[key]=(quality,dict(r,sharpness=sharp),crop.copy())
    cap.release();rows=[]
    for quality,r,crop in sorted(best.values(),key=lambda v:(v[1]['pid'],v[1]['frame'])):
        # Role is not jersey colour: a blue goalkeeper is also a blue hard negative.
        r['kit']=kitvotes[r['pid']].most_common(1)[0][0] if kitvotes[r['pid']] else -1
        path=out/'images'/r['split']/f'p{r["pid"]}_f{r["frame"]:04d}_id{r["algorithm_id"]}.jpg'
        path.parent.mkdir(parents=True,exist_ok=True)
        ok,b=cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok;b.tofile(str(path))
        r['image']=path.relative_to(out).as_posix();r['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        r['quality_score']=quality;rows.append(r)
    assert len({r['sha256'] for r in rows})==len(rows),'Duplicate images across data'
    # Enrolment gallery is an explicitly declared subset of train, never held-out queries.
    for r in rows:r['enrolment_gallery']=False
    for pid in range(len(groups)):
        rr=[r for r in rows if r['pid']==pid and r['split']=='train']
        assert len(rr)>=8,(pid,'insufficient train')
        for k in np.linspace(0,len(rr)-1,min(12,len(rr))).astype(int):rr[k]['enrolment_gallery']=True
        for split in ['val','test']:assert any(r['pid']==pid and r['split']==split for r in rows),(pid,split)
    with (out/'manifest.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    stats={name:{split:sum(r['identity']==name and r['split']==split for r in rows) for split in WINDOWS} for name in groups}
    dump(out/'dataset_report.json',dict(windows=WINDOWS,source_offset=3150,counts=stats,filter_counts=dict(counts),
        total=len(rows),same_frame_conflicts=0,minimum_train_query_gap_frames=61,
        label_source='human_confirmed_identity_groups; algorithm_boxes',
        evaluation_scope='known players, single clip chronological development holdout; not unseen-match test',
        audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((out/'manifest.csv').read_bytes()).hexdigest()))
    print(json.dumps(stats,ensure_ascii=False,indent=2));print('total',len(rows))


if __name__=='__main__':main()
