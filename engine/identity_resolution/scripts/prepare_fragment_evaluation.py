"""Freeze algorithm-only training manifest BEFORE opening human evaluation labels."""
import argparse,csv,json,pickle,sys
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_backbone_train import sha,dump,quality_ok,overlap


def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def prepare(data,audit_path,video,review_path):
    data=Path(data);manifest=data/'manifest.csv'
    if (data/'fragment_protocol.json').exists():raise ValueError('Protocol already frozen')
    train=list(csv.DictReader(manifest.open(encoding='utf-8')))
    assert all(r['split']=='train' for r in train)
    for r in train:
        for k in ['pid','frame','source_local_id','algorithm_id','node','original_v3_id','kit']:r[k]=int(r[k])
        r['source_frame']=r['frame']+3150
        r['label_source']='algorithm_seed_only'
        r['nearest_training_gap']=-1
    trainpath=data/'train_manifest.csv';write_csv(trainpath,train);frozen=sha(trainpath)
    # This is the first read of user identity correspondence. Train set is immutable.
    review=json.loads(Path(review_path).read_text(encoding='utf-8'))
    mapping={};seed_ids={r['algorithm_id'] for r in train}
    for pid in sorted({r['pid'] for r in train}):
        trunk=next(r['algorithm_id'] for r in train if r['pid']==pid)
        groups=[g for g in review['groups'].values() if trunk in g]
        if len(groups)!=1:raise ValueError(('Selected algorithm identity lacks evaluation mapping',trunk))
        for oid in groups[0]:
            if oid in mapping and mapping[oid]!=pid:raise ValueError('Two selected classes belong to same reviewed person')
            mapping[oid]=pid
    audit=pickle.load(Path(audit_path).open('rb'));byframe=defaultdict(list);requests=defaultdict(list);rejected=defaultdict(int)
    for n in audit['nodes']:
        for r in n['rows']:byframe[r[0]].append(r)
    cap=cv2.VideoCapture(str(video));width,height=int(cap.get(3)),int(cap.get(4));total=int(cap.get(7))
    for n in audit['nodes']:
        oid=n['output_id']
        if oid in seed_ids or oid not in mapping:continue
        if oid in review['invalid_ids'] or n['quarantine']:continue
        for row in n['rows']:
            ov=overlap(row,byframe[row[0]])
            if not quality_ok(row,ov,width,height):rejected[oid]+=1;continue
            requests[row[0]].append((n,row,ov))
    best={}
    for frame in range(total):
        ok,image=cap.read();assert ok
        for n,row,ov in requests[frame]:
            oid=n['output_id'];pid=mapping[oid];_,local,x,y,w,h,conf=row
            crop=image[int(y):int(y+h),int(x):int(x+w)].copy()
            sharp=float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
            score=conf*(1-ov)*(.5+.5*min(1,sharp/100))
            interval=2 if len(n['rows'])<45 else 15
            key=(oid,frame//interval)
            prototype=next(r for r in train if r['pid']==pid)
            r={k:'' for k in prototype}
            r.update(pid=pid,identity=prototype['identity'],algorithm_id=oid,original_v3_id=oid,source_local_id=local,node=n['id'],frame=frame,
                     source_frame=frame+3150,x=x,y=y,w=w,h=h,confidence=conf,overlap=ov,sharpness=sharp,
                     kit=prototype['kit'],quality_score=score,enrolment_gallery=False,label_source='user_review_evaluation_only',
                     nearest_training_gap=min(abs(frame-t['frame']) for t in train if t['pid']==pid))
            if key not in best or score>best[key][0]:best[key]=(score,r,crop)
    cap.release();byid=defaultdict(list)
    for _,r,im in best.values():byid[r['algorithm_id']].append((r,im))
    partitions={};held=[]
    for pid in sorted({r['pid'] for r in train}):
        ids=sorted((i for i,rr in byid.items() if rr[0][0]['pid']==pid),key=lambda i:min(r['frame'] for r,_ in byid[i]))
        if not ids:partitions[pid]={'val':[],'test':[]};continue
        test=ids[-1];partitions[pid]={'val':ids[:-1],'test':[test]}
        for oid in ids:
            split='test' if oid==test else 'val'
            for r,im in sorted(byid[oid],key=lambda x:x[0]['frame']):
                dest=data/'images'/split/f'p{pid}_f{r["frame"]:06d}_id{oid}.jpg';dest.parent.mkdir(parents=True,exist_ok=True)
                ok,b=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok;b.tofile(str(dest))
                r.update(split=split,image=dest.relative_to(data).as_posix(),sha256=sha(dest));held.append(r)
    rows=train+held
    sets={s:{r['algorithm_id'] for r in rows if r['split']==s} for s in ['train','val','test']}
    assert not sets['train']&sets['val'] and not sets['train']&sets['test'] and not sets['val']&sets['test']
    assert len({(r['source_local_id'],r['frame']) for r in rows})==len(rows)
    assert len({r['sha256'] for r in rows})==len(rows)
    assert sha(trainpath)==frozen
    write_csv(manifest,rows)
    report=dict(seed_ids=sorted(seed_ids),train_manifest_sha256=frozen,manifest_sha256=sha(manifest),
                review_sha256=sha(review_path),review_used_after_training_freeze=True,whole_id_disjoint=True,
                counts={s:sum(r['split']==s for r in rows) for s in sets},
                coverage={s:len({r['pid'] for r in rows if r['split']==s}) for s in sets},
                partitions=partitions,quality_rejected=dict(rejected),invalid_ids_excluded=review['invalid_ids'],
                no_identity_expansion_into_training=True,
                caveat='Other algorithm IDs only; very short held-out fragments and near-time neighbours are reported, not independent new matches.')
    dump(data/'fragment_protocol.json',report)
    c=json.loads((ROOT/'reid_model/config.json').read_text())
    c.update(num_class=len(seed_ids),manifest=str(manifest.resolve()),data_root=str(data.resolve()),run_dir=str((data.parent/'training').resolve()),
             weight_file=str((ROOT/'reid_model/weights/jx_vit_base_p16_224-80ecf9dd.pth').resolve()),
             workers=0,eval_batch=8,autotune_k=[2],epochs=20,patience=5,steps_per_epoch=20,
             seed=20260910,train_only=False,fragment_protocol=str((data/'fragment_protocol.json').resolve()))
    dump(data/'training_config.json',c);print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    prepare(a.data,ROOT/'data/audit_data.pkl',ROOT/'data/video.mp4',ROOT/'configs/review_20260910.json')
