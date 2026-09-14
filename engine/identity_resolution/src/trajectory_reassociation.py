"""Label-blind association; no training or evaluation identity map is consulted."""
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
import numpy as np


def cosine_sim(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.ndim!=1 or b.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():return None
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>1e-9 else None


def get_track_features(node):
    samples=node.get('samples',[])
    if not samples:return None,None
    app=[np.asarray(s[1]) for s in samples];col=[np.asarray(s[2]) for s in samples]
    if len({a.shape for a in app})!=1 or len({c.shape for c in col})!=1:return None,None
    return np.mean(app,axis=0),np.mean(col,axis=0)


def group_tracks(nodes):
    groups={}
    for n in nodes:
        if n.get('quarantine'):continue
        oid=n['output_id']
        g=groups.setdefault(oid,dict(id=oid,output_id=oid,rows=[],positions={},samples=[],kits=set()))
        for r in n['rows']:
            g['rows'].append(r)
            if r[0] in n.get('positions',{}):g['positions'][r[0]]=np.asarray(n['positions'][r[0]])
        g['samples'].extend(n.get('samples',[]))
        if n.get('kit',-1)>=0:g['kits'].add(n['kit'])
    for g in groups.values():
        g['rows'].sort(key=lambda r:r[0]);g['samples'].sort(key=lambda s:s[0])
        fs=[r[0] for r in g['rows']]
        if len(fs)!=len(set(fs)):raise ValueError('Input same-frame identity collision')
        g['start'],g['end']=fs[0],fs[-1]
        g['kit']=next(iter(g['kits'])) if len(g['kits'])==1 else -1
    return list(groups.values())


def velocity(node,side):
    rows=node['rows'][-20:] if side=='end' else node['rows'][:20]
    if len(rows)<2:return np.zeros(2)
    t=np.array([r[0] for r in rows],float);t-=t.mean()
    pts=np.array([node['positions'][r[0]] for r in rows])
    return (t[:,None]*pts).sum(0)/max(float(t@t),1e-9)


def build_trajectory_graph(nodes,config,max_gap=60):
    tracks=group_tracks(nodes);edges=[]
    hard=config.get('hard_constraints',{});weights=config.get('scoring_weights',{'motion':.3,'appearance':.5,'colour':.2})
    for l in tracks:
        for r in tracks:
            gap=r['start']-l['end']
            if l['id']==r['id'] or not 1<=gap<=max_gap:continue
            if len(l['kits'])>1 or len(r['kits'])>1:continue
            if hard.get('kit_match_required',True) and l['kit']>=0 and r['kit']>=0 and l['kit']!=r['kit']:continue
            try:
                a,b=l['positions'][l['end']],r['positions'][r['start']]
                distance=float(np.linalg.norm(a-b));lv,rv=velocity(l,'end'),velocity(r,'start')
                residual=max(float(np.linalg.norm(a+lv*gap-b)),float(np.linalg.norm(b-rv*gap-a)))
            except (KeyError,ValueError):continue
            if not np.isfinite([distance,residual]).all():continue
            fps=config.get('fps',30)
            if distance>hard.get('distance_slack_m',1.)+hard.get('max_speed_m_s',9.)*gap/fps:continue
            if residual>hard.get('residual_slack_m',.7)+hard.get('residual_growth_m_s',3.)*gap/fps:continue
            la,lc=get_track_features(l);ra,rc=get_track_features(r)
            if la is None or ra is None:continue
            app,col=cosine_sim(la,ra),cosine_sim(lc,rc)
            if app is None or col is None or app<hard.get('min_app_sim',.9) or col<hard.get('min_col_sim',.35):continue
            motion=math.exp(-residual/(.6+2*gap/fps))
            score=weights['motion']*motion+weights['appearance']*app+weights['colour']*col
            edges.append(dict(from_oid=l['id'],to_oid=r['id'],gap=gap,distance=distance,residual=residual,
                              app_sim=app,col_sim=col,motion_score=motion,combined=score))
    return edges


def greedy_associate(candidates,nodes,threshold=.85,use_gt_filter=False,margin=.05):
    if use_gt_filter:raise ValueError('Ground-truth filtering is forbidden')
    tracks=group_tracks(nodes);parent={g['id']:g['id'] for g in tracks}
    frames={g['id']:{r[0] for r in g['rows']} for g in tracks};kits={g['id']:set(g['kits']) for g in tracks}
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    outgoing,incoming=defaultdict(list),defaultdict(list)
    for e in candidates:outgoing[e['from_oid']].append(e);incoming[e['to_oid']].append(e)
    for table in [outgoing,incoming]:
        for values in table.values():values.sort(key=lambda e:(-e['combined'],e['from_oid'],e['to_oid']))
    accepted=[];successors={};predecessors={}
    for e in sorted(candidates,key=lambda x:-x['combined']):
        a,b=e['from_oid'],e['to_oid']
        if a==b or a in successors or b in predecessors or e['combined']<threshold:continue
        if outgoing[a][0] is not e or incoming[b][0] is not e:continue
        if any(len(v)>1 and e['combined']-v[1]['combined']<margin for v in [outgoing[a],incoming[b]]):continue
        x,y=root(a),root(b)
        if x==y or frames[x]&frames[y] or len(kits[x]|kits[y])>1:continue
        parent[y]=x;frames[x]|=frames[y];kits[x]|=kits[y]
        successors[a]=b;predecessors[b]=a;accepted.append(dict(e,accepted=True))
    return accepted,successors,predecessors


def expand_to_full_mapping(accepted_edges,nodes):
    ids={n['output_id'] for n in nodes if not n.get('quarantine')};parent={i:i for i in ids}
    frames=defaultdict(set)
    for n in nodes:
        if not n.get('quarantine'):frames[n['output_id']].update(r[0] for r in n.get('rows',[]))
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    for e in accepted_edges:
        a,b=root(e['from_oid']),root(e['to_oid'])
        if a==b:continue
        if frames[a]&frames[b]:raise ValueError('Association would join simultaneous observations')
        lo,hi=min(a,b),max(a,b);parent[hi]=lo;frames[lo]|=frames[hi]
    return {i:root(i) for i in sorted(ids)}


def write_result(audit,mapping,out):
    quarantine_obs=set()
    for node in audit.get('nodes') or []:
        if not node.get('quarantine'):continue
        for row in node['rows']:quarantine_obs.add((int(row[0]),int(node['output_id'])))
    rows=[];keys=set()
    def append(row,allow_skip=False):
        identity=mapping.get(row[1],row[1]);key=(row[0],identity)
        if key in keys:
            if allow_skip:return
            raise ValueError('Duplicate frame/identity after association')
        keys.add(key);rows.append((row[0],identity,*row[2:]))
    for row in audit['rows']:
        if (int(row[0]),int(row[1])) in quarantine_obs:continue
        append(row,False)
    for row in audit['rows']:
        if (int(row[0]),int(row[1])) not in quarantine_obs:continue
        append(row,True)
    with (out/'tracking_reid_trajectory.txt').open('w',encoding='utf-8') as f:
        for frame,i,x,y,w,h,confidence in sorted(rows):
            f.write(f'{frame+1},{i},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{confidence:.6f},-1,-1,-1\n')
    return rows


def run_reassociation(audit_path,output_dir,config=None,gt_filter=False):
    if gt_filter:raise ValueError('Ground-truth filtering is forbidden')
    config=config or {};out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    with Path(audit_path).open('rb') as f:audit=pickle.load(f)
    if audit.get('feature_metadata',{}).get('dimension')!=3840:
        raise ValueError('Extract checkpoint-specific TransReID features first; legacy cache is not the trained model')
    nodes=audit['nodes'];candidates=build_trajectory_graph(nodes,config,config.get('max_gap',60))
    accepted,_,_=greedy_associate(candidates,nodes,config.get('threshold',.85),False,config.get('ambiguity_margin',.05))
    mapping=expand_to_full_mapping(accepted,nodes);rows=write_result(audit,mapping,out)
    for name,value in [('trajectory_mapping.json',mapping),('accepted_edges.json',accepted),('candidates.json',candidates)]:
        (out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    report=dict(input_rows=len(audit['rows']),output_rows=len(rows),normal_ids_before=len(mapping),
                normal_ids_after=len(set(mapping.values())),accepted_edges=len(accepted),
                same_frame_conflicts=0,labels_used_for_association=False,feature_metadata=audit['feature_metadata'])
    report['candidate_edges']=len(candidates)
    report['below_score_threshold']=sum(e['combined']<config.get('threshold',.85) for e in candidates)
    report['unresolved_passing_edges']=len(candidates)-report['below_score_threshold']-len(accepted)
    (out/'association_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return dict(mapping=mapping,accepted_edges=accepted,report=report,output_tracking=str(out/'tracking_reid_trajectory.txt'))
