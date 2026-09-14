"""Read-only fragment review: actual overlapping crops, geometry and ReID ablations."""
import argparse
import csv
from collections import defaultdict
import html
import json
from pathlib import Path
import pickle
import sys
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from scripts.run_pipeline import load_config
from features import backend,load_model,overlap,sha
from trajectory_reassociation import group_tracks
from second_stage import DEFAULTS,describe,geometry,pair_appearance


def bracket_error(query,anchor):
    before=[f for f in anchor['positions'] if f<query['start']]
    after=[f for f in anchor['positions'] if f>query['end']]
    if not before or not after:return None
    lo,hi=max(before),min(after)
    if query['start']-lo>60 or hi-query['end']>60:return None
    if any(r[0] not in query['positions'] for r in query['rows']):return None
    errors=[float(np.linalg.norm(query['positions'][r[0]]-(anchor['positions'][lo]+
        (anchor['positions'][hi]-anchor['positions'][lo])*(r[0]-lo)/(hi-lo)))) for r in query['rows']]
    return dict(residual=max(errors),bracket=[lo,hi])


def rank_candidates(query,anchors):
    c=dict(DEFAULTS,min_samples=1,local_matching=True);x=describe(query,c);out=[]
    for anchor in anchors:
        if len(anchor['rows'])<30 or len(anchor['samples'])<2:continue
        geo,reason=geometry(query,anchor,c)
        if reason:continue
        aff,cent,col,source=pair_appearance(query,anchor,x,describe(anchor,c),geo,c)
        if aff is None:continue
        bracket=bracket_error(query,anchor)
        residual=bracket['residual'] if bracket else None
        fused=.7*aff+.3*np.exp(-residual/1.5) if residual is not None else None
        out.append(dict(target=anchor['id'],affinity=float(aff),centroid=float(cent),
            appearance_source=source,bracket=bracket,fused_score=float(fused) if fused is not None else None))
    return out


def decide(candidates,samples):
    """All outputs are review proposals, never training labels or fabricated detections."""
    reid=sorted(candidates,key=lambda r:-r['affinity'])
    bracket=[r for r in candidates if r['bracket'] is not None]
    motion=sorted(bracket,key=lambda r:r['bracket']['residual'])
    fused=sorted(bracket,key=lambda r:-r['fused_score'])
    result=dict(interpolation=None,reid=None,fusion=None,automatic_training_label=False)
    if motion and motion[0]['bracket']['residual']<=1.0:
        if len(motion)==1 or motion[1]['bracket']['residual']-motion[0]['bracket']['residual']>=.3:
            result['interpolation']=motion[0]['target']
    if samples>=2 and reid and reid[0]['affinity']>=.85:
        if len(reid)==1 or reid[0]['affinity']-reid[1]['affinity']>=.05:result['reid']=reid[0]['target']
    if samples>=2 and fused:
        best=fused[0]
        if best['bracket']['residual']<=2. and best['affinity']>=.80 and best['fused_score']>=.75:
            if len(fused)==1 or best['fused_score']-fused[1]['fused_score']>=.05:
                result['fusion']=best['target']
    return result


def extract_probe(audit,ids,video,config,weight,out,device):
    import cv2
    import torch
    from PIL import Image
    by_frame=defaultdict(list);queries=defaultdict(list)
    for n in audit['nodes']:
        for r in n['rows']:
            by_frame[r[0]].append(r)
            if not n['quarantine'] and n['output_id'] in ids:queries[r[0]].append((n,r))
    model=load_model(config,weight,device);transform=backend().Crops([],config,False).transform
    records=[];pending=[];sample_map=defaultdict(list)
    cap=cv2.VideoCapture(str(video));assert cap.isOpened()
    def flush():
        if not pending:return
        with torch.inference_mode():f=torch.nn.functional.normalize(model(torch.stack([v[0] for v in pending]).to(device)).float(),dim=1).cpu().numpy()
        for vector,(_,record,colour) in zip(f,pending):
            sample_map[record['id']].append((record['frame'],vector,colour));records.append(record)
        pending.clear()
    try:
        for frame in range(max(queries,default=-1)+1):
            ok,im=cap.read()
            if not ok:raise ValueError('Video ended early')
            for node,row in queries.get(frame,[]):
                _,local,x,y,w,h,confidence=row
                x0,y0=max(0,int(x)),max(0,int(y));x1,y1=min(im.shape[1],int(x+w)),min(im.shape[0],int(y+h))
                if x1<=x0 or y1<=y0:continue
                crop=im[y0:y1,x0:x1];folder=out/'images'/f'ID{node["output_id"]}';folder.mkdir(parents=True,exist_ok=True)
                path=folder/f'f{frame:04d}.jpg';ok,jpg=cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok;jpg.tofile(str(path))
                context=im.copy();cv2.rectangle(context,(x0,y0),(x1,y1),(0,0,255),3)
                cv2.putText(context,f'ID {node["output_id"]}  clip {frame}  source {frame+3150}',(15,30),0,.8,(255,255,255),2)
                context=cv2.resize(context,(960,540));context_path=folder/f'f{frame:04d}_context.jpg'
                ok,jpg=cv2.imencode('.jpg',context);assert ok;jpg.tofile(str(context_path))
                hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV);hh,ww=hsv.shape[:2]
                torso=hsv[int(hh*.15):max(int(hh*.15)+1,int(hh*.6)),int(ww*.2):max(int(ww*.2)+1,int(ww*.8))]
                colour=cv2.calcHist([torso],[0,1],None,[16,8],[0,180,0,256]).reshape(-1);colour/=max(np.linalg.norm(colour),1e-9)
                record=dict(id=node['output_id'],frame=frame,source_frame=frame+3150,confidence=confidence,
                    overlap=overlap(row,by_frame[frame]),width=w,height=h,image=str(path.relative_to(out)).replace('\\','/'),
                    context=str(context_path.relative_to(out)).replace('\\','/'),inference_probe_only=True)
                with Image.open(path) as rgb:tensor=transform(rgb.convert('RGB'))
                pending.append((tensor,record,colour))
                if len(pending)>=8:flush()
        flush()
    finally:
        cap.release();del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    return records,sample_map


def walls(records,reports,out):
    from PIL import Image,ImageOps,ImageDraw,ImageFont
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    tiles=[];articles=[]
    for oid,report in reports.items():
        rr=sorted([r for r in records if r['id']==oid],key=lambda r:r['frame'])
        picks=[rr[i] for i in np.unique(np.linspace(0,len(rr)-1,min(3,len(rr))).astype(int))] if rr else []
        tile=Image.new('RGB',(720,470),'#f5f6f8');d=ImageDraw.Draw(tile)
        d.text((12,8),f"ID{oid} | {report['observations']}条 | {report['start']/30:.2f}–{report['end']/30:.2f}s",font=font,fill='#17202a')
        for i,r in enumerate(picks):
            with Image.open(out/r['image']) as im:crop=ImageOps.contain(im.convert('RGB'),(215,270))
            tile.paste(crop,(12+i*235,45));d.text((12+i*235,325),f"帧{r['frame']}  重叠{r['overlap']:.0%}",font=small,fill='black')
        d.text((12,360),f"插值候选 {report['decision']['interpolation']} | ReID候选 {report['decision']['reid']} | 融合 {report['decision']['fusion']}",font=small,fill='#17202a')
        d.text((12,390),'候选不是人工确认；单帧不自动定身份；不进入训练。',font=small,fill='#8b4513')
        filename=f'ID{oid}_wall.jpg';tile.save(out/filename,quality=95);tiles.append((oid,tile))
        items=''.join(f'<figure><a href="{html.escape(r["context"])}"><img src="{html.escape(r["image"])}"></a><figcaption>clip {r["frame"]} / source {r["source_frame"]} · 重叠 {r["overlap"]:.0%}</figcaption></figure>' for r in picks)
        context=f'<a href="{picks[len(picks)//2]["context"]}"><img class="context" src="{picks[len(picks)//2]["context"]}"></a>' if picks else ''
        articles.append(f'<article id="id{oid}"><h2>ID{oid} · {report["observations"]}条观测</h2><p>{html.escape(json.dumps(report["decision"],ensure_ascii=False))}</p><div class="crops">{items}</div>{context}<details><summary>所有帧及算法证据</summary><pre>{html.escape(json.dumps(report,ensure_ascii=False,indent=2))}</pre>'+''.join(f'<a href="{r["context"]}">帧{r["frame"]}上下文</a> ' for r in rr)+'</details></article>')
    for page,start in enumerate(range(0,len(tiles),6),1):
        subset=tiles[start:start+6];sheet=Image.new('RGB',(1440,((len(subset)+1)//2)*470),'white')
        for i,(_,tile) in enumerate(subset):sheet.paste(tile,((i%2)*720,(i//2)*470))
        sheet.save(out/f'图片墙_{page}.jpg',quality=95)
    document='<!doctype html><meta charset="utf-8"><title>短轨迹人工核对</title><style>body{max-width:1100px;margin:30px auto;font:16px sans-serif;background:#eef1f4;color:#17202a}article{background:white;padding:24px;margin:24px 0;border-radius:10px}.crops{display:flex;gap:20px}.crops img{height:260px;max-width:250px;object-fit:contain}.context{max-width:960px;width:100%}figure{margin:0}pre{white-space:pre-wrap}</style><h1>短轨迹与遮挡裁剪核对</h1><p>点击人物图片查看原帧红框上下文。数字是原输出ID；候选为算法实验结果，尚未人工确认。没有删除原始检测，没有把插值坐标当成真实观测或训练样本。</p>'+''.join(articles)
    (out/'index.html').write_text(document,encoding='utf-8')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ids',required=True)
    p.add_argument('--features',type=Path,required=True);p.add_argument('--mapping',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    args=p.parse_args(argv);out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('Use a new review directory')
    ids=set(map(int,args.ids.split(',')));c=load_config(ROOT/'configs/pipeline_config.json')
    audit=pickle.loads(Path(c['paths']['audit_data']).read_bytes());clean=pickle.loads(args.features.read_bytes())
    if sha(c['paths']['reid_weights'])!=clean['feature_metadata']['fingerprint']['checkpoint']:raise ValueError('Checkpoint mismatch')
    mapping={int(i):j for i,j in json.loads(args.mapping.read_text()).items()}
    (out/'protocol.json').write_text(json.dumps(dict(ids=sorted(ids),checkpoint=sha(c['paths']['reid_weights']),
        source_audit=sha(c['paths']['audit_data']),probe='all actual target crops, including overlaps; no training',
        fusion=dict(min_samples=2,min_affinity=.80,max_residual=2.,min_score=.75,margin=.05)),indent=2),encoding='utf-8')
    records,samples=extract_probe(audit,ids,c['paths']['video'],json.loads(Path(c['paths']['reid_config']).read_text()),c['paths']['reid_weights'],out,args.device)
    queries={t['id']:t for t in group_tracks(audit['nodes'])};reports={}
    for oid in sorted(ids):
        query=dict(queries[oid],samples=samples[oid])
        # Query fragment excluded from every target's appearance evidence and interpolation anchors.
        anchors=group_tracks([dict(n,output_id=mapping.get(n['output_id'],n['output_id'])) for n in clean['nodes'] if n['output_id']!=oid])
        candidates=rank_candidates(query,anchors);rr=[r for r in records if r['id']==oid]
        reports[oid]=dict(start=query['start'],end=query['end'],observations=len(query['rows']),
            original_clean_samples=sum(len(n['samples']) for n in clean['nodes'] if n['output_id']==oid),probe_samples=len(samples[oid]),
            overlap_min=min((r['overlap'] for r in rr),default=None),overlap_max=max((r['overlap'] for r in rr),default=None),
            decision=decide(candidates,len(samples[oid])),candidates=sorted(candidates,key=lambda r:-r['affinity']))
    (out/'review.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'crop_records.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    np.savez(out/'probe_features.npz',**{f'id{i}':np.array([s[1] for s in samples[i]]) for i in ids})
    walls(records,reports,out)
    print(json.dumps({i:dict(samples=r['probe_samples'],decision=r['decision']) for i,r in reports.items()},indent=2))


if __name__=='__main__':main()
