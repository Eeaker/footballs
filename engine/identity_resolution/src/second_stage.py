"""Second-pass recovery. Query identity maps are forbidden; frozen training references are optional."""
import json
import math
import pickle
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np
from trajectory_reassociation import group_tracks,cosine_sim,velocity,write_result


DEFAULTS=dict(max_gap=1800,fps=30,min_samples=2,max_samples=16,
              min_affinity=.85,min_centroid=.70,min_colour=.25,
              component_min_centroid=.65,component_min_affinity=.72,
              ambiguity_margin=.025,distance_slack_m=1.5,max_speed_m_s=9.,
              local_matching=False,local_window_frames=60,sparse_bridge=False,
              bridge_max_side_gap=30,bridge_max_span=45,bridge_max_residual_m=1.,bridge_margin_m=.3,
              reference_threshold=.85,reference_margin=.15,reference_vote=.75,
              appearance_floor=None,short_appearance_floor=None,long_appearance_floor=None,
              gallery_rewrite_affinity=True,gallery_veto=False,gallery_floor=None,
              absorb_affinity=.80,mutual_unique=False,
              unregistered_long_floor=None,unmatched_to_registered_long_floor=None,
              occupancy_veto=False,occupancy_max_gap=45,occupancy_residual_m=1.2,
              mixed_box_exclude=False,peel_max_gap=20,peel_max_m=3.5,
              jersey_agree=False,jersey_votes=None,jersey_votes_file=None,
              gta_connect=False,gta_merge_dist=.4,gta_additive=False,
              rule_geometry=False,rule_max_gap=60,rule_max_m=3.5,
              quarantine_bridge=False,quarantine_max_gap=120,
              rule_only=False,seed_mapping=None,seed_mapping_file=None,
              gap_scaled_floor=False,appearance_floor_near=.76,appearance_floor_far=.60,
              appearance_floor_span=180,unmatched_premium_near=.04,unmatched_premium_far=0.,
              absorb_near=.86,absorb_far=.66)


def occupancy_index(nodes,quarantine=True):
    """Frame occupancy. Quarantine points never become merge identities."""
    by_frame={}
    for n in nodes:
        if bool(n.get('quarantine'))!=quarantine:continue
        pos=n.get('positions') or {}
        for r in n.get('rows') or []:
            xy=pos.get(r[0])
            if xy is None:continue
            by_frame.setdefault(int(r[0]),[]).append(dict(oid=n.get('output_id',n.get('id')),xy=np.asarray(xy,float),kit=n.get('kit',-1)))
    return by_frame


def _iou(a,b):
    x1,y1,w1,h1=a[2:6];x2,y2,w2,h2=b[2:6]
    xa,ya=max(x1,x2),max(y1,y2);xb,yb=min(x1+w1,x2+w2),min(y1+h1,y2+h2)
    inter=max(0.,xb-xa)*max(0.,yb-ya)
    return float(inter/max(w1*h1+w2*h2-inter,1e-9))


def adjacent_rival(a,b,config):
    """Short-gap, same-kit, close in metres, but boxes do not overlap: two people, not a continuation."""
    earlier,later=(a,b) if a['end']<=b['start'] else (b,a)
    gap=later['start']-earlier['end']
    if not 1<=gap<=config.get('peel_max_gap',20):return False
    if earlier.get('kit',-1)>=0 and later.get('kit',-1)>=0 and earlier['kit']!=later['kit']:return False
    host=max(earlier['rows'],key=lambda r:r[0]);guest=min(later['rows'],key=lambda r:r[0])
    if _iou(host,guest)>=config.get('peel_min_iou',.15):return False
    pa,pb=earlier['positions'].get(host[0]),later['positions'].get(guest[0])
    if pa is None or pb is None:return False
    return float(np.linalg.norm(pa-pb))<=config.get('peel_max_m',3.5)


def path_hits(a,b,geo,normal,quarantine,config):
    """Kalman-linear occupants on the A-B path. Quarantine is evidence; other normals are rivals."""
    if not geo:return [],[]
    ids={a['id'],b['id']}
    rad=config.get('occupancy_residual_m',1.2)
    n_hits=[];q_hits=[]
    for fa,fb in geo['anchors']:
        pa,pb=a['positions'].get(fa),b['positions'].get(fb)
        if pa is None or pb is None:continue
        lo,hi=int(min(fa,fb)),int(max(fa,fb));span=max(hi-lo,1)
        for f in range(lo+1,hi):
            pred=pa+((f-lo)/span)*(pb-pa)
            for hit in normal.get(f,()):
                if hit['oid'] in ids:continue
                if float(np.linalg.norm(hit['xy']-pred))<=rad:n_hits.append(dict(hit,frame=f))
            for hit in quarantine.get(f,()):
                if float(np.linalg.norm(hit['xy']-pred))<=rad:q_hits.append(dict(hit,frame=f))
    return n_hits,q_hits


def occupancy_blocks(a,b,geo,index,config):
    if not geo or not index:return False
    if geo['gap']>config.get('occupancy_max_gap',45):return False
    ids={a['id'],b['id']}
    for fa,fb in geo['anchors']:
        pa,pb=a['positions'].get(fa),b['positions'].get(fb)
        if pa is None or pb is None:continue
        lo,hi=int(min(fa,fb)),int(max(fa,fb))
        span=max(hi-lo,1)
        for f in range(lo+1,hi):
            t=(f-lo)/span
            pred=pa+t*(pb-pa)
            for hit in index.get(f,()):
                if hit['oid'] in ids:continue
                if float(np.linalg.norm(hit['xy']-pred))<=config.get('occupancy_residual_m',1.2):
                    return True
    return False


def describe(track,config):
    samples=track['samples']
    if len(samples)<config['min_samples']:return None
    indices=np.unique(np.linspace(0,len(samples)-1,min(len(samples),config['max_samples'])).astype(int))
    features=np.array([samples[i][1] for i in indices],dtype=np.float32)
    if features.ndim!=2 or not np.isfinite(features).all():return None
    norms=np.linalg.norm(features,axis=1,keepdims=True)
    if np.any(norms<1e-9):return None
    features/=norms
    centroid=features.mean(0);centroid/=max(np.linalg.norm(centroid),1e-9)
    colour=np.mean([samples[i][2] for i in indices],axis=0)
    return dict(features=features,centroid=centroid,colour=colour)


def geometry(a,b,config):
    """Check actual observed transitions, including interleaved time envelopes."""
    if set(a['positions'])&set(b['positions']):return None,'same_frame'
    if len(a['kits']|b['kits'])>1:return None,'kit_conflict'
    observations=sorted([(r[0],0,a['positions'].get(r[0])) for r in a['rows']]+
                        [(r[0],1,b['positions'].get(r[0])) for r in b['rows']],key=lambda x:x[0])
    transitions=[];anchors=[]
    for l,r in zip(observations,observations[1:]):
        if l[1]==r[1]:continue
        gap=r[0]-l[0]
        if gap<=0:return None,'same_frame'
        if l[2] is None or r[2] is None:return None,'missing_geometry'
        dist=float(np.linalg.norm(l[2]-r[2]))
        if not math.isfinite(dist):return None,'missing_geometry'
        if dist>config['distance_slack_m']+config['max_speed_m_s']*gap/config['fps']:
            return None,'unreachable'
        transitions.append((gap,dist))
        anchors.append((l[0] if l[1]==0 else r[0],r[0] if r[1]==1 else l[0]))
    if not transitions or min(t[0] for t in transitions)>config['max_gap']:return None,'time_gap'
    gap,dist=min(transitions)
    return dict(gap=gap,distance=dist,anchors=anchors,
                interleaved=a['start']<=b['end'] and b['start']<=a['end']),None


def appearance(x,y):
    if x is None or y is None:return None,None,None
    sim=x['features']@y['features'].T
    nearest=.5*(float(sim.max(1).mean())+float(sim.max(0).mean()))
    centroid=float(x['centroid']@y['centroid'])
    return .5*centroid+.5*nearest,centroid,cosine_sim(x['colour'],y['colour'])


def mean_pairwise_distance(x,y):
    """GTA-Link connector: 1 - mean cosine over all sample pairs."""
    if x is None or y is None:return None
    return float(1.-(x['features']@y['features'].T).mean())


def cluster_mean_distance(members_a,members_b,desc):
    fa=[desc[i]['features'] for i in members_a if desc.get(i) is not None]
    fb=[desc[j]['features'] for j in members_b if desc.get(j) is not None]
    if not fa or not fb:return 1.
    a=np.concatenate(fa,0);b=np.concatenate(fb,0)
    return float(1.-(a@b.T).mean())


def pair_appearance(a,b,x,y,geo,c):
    affinity,centroid,colour=appearance(x,y);source='global'
    if c['local_matching'] and geo and geo['gap']<=60:
        local=[]
        for fa,fb in geo['anchors']:
            if abs(fa-fb)>60:continue
            aa=dict(a,samples=[s for s in a['samples'] if abs(s[0]-fa)<=c['local_window_frames']])
            bb=dict(b,samples=[s for s in b['samples'] if abs(s[0]-fb)<=c['local_window_frames']])
            value=appearance(describe(aa,c),describe(bb,c))
            if value[0] is None:return affinity,centroid,colour,source
            local.append(value)
        if local:
            value=min(local,key=lambda v:v[0])
            if affinity is None or value[0]>affinity:affinity,centroid,colour=value;source='local'
    return affinity,centroid,colour,source


def bridge_sparse(tracks,owner,clusters,desc,c,reference_matches=None):
    """Attach only bracketed sparse fragments. Frozen anchors prevent bridge propagation."""
    if not c['sparse_bridge']:return []
    reference_matches=reference_matches or {}
    anchors={i:set(v) for i,v in clusters.items() if any(desc[n] is not None for n in v)}
    proposals=[]
    for i,t in tracks.items():
        if desc[i] is not None or len(clusters[owner[i]])!=1:continue
        if t['end']-t['start']+1>c['bridge_max_span']:continue
        scores=[]
        for anchor,members in anchors.items():
            if i in members:continue
            if any(geometry(t,tracks[n],c)[1] is not None for n in members):continue
            positions={f:xy for n in members for f,xy in tracks[n]['positions'].items()}
            before=[f for f in positions if f<t['start']];after=[f for f in positions if f>t['end']]
            if not before or not after:continue
            lo,hi=max(before),min(after)
            if t['start']-lo>c['bridge_max_side_gap'] or hi-t['end']>c['bridge_max_side_gap']:continue
            if any(r[0] not in t['positions'] for r in t['rows']):continue
            errors=[float(np.linalg.norm(t['positions'][r[0]]-(positions[lo]+(positions[hi]-positions[lo])*(r[0]-lo)/(hi-lo)))) for r in t['rows']]
            residual=max(errors)
            if not math.isfinite(residual):continue
            scores.append((residual,anchor,lo,hi))
        scores.sort()
        if not scores or scores[0][0]>c['bridge_max_residual_m']:continue
        if len(scores)>1 and scores[1][0]-scores[0][0]<c['bridge_margin_m']:continue
        residual,anchor,lo,hi=scores[0]
        proposals.append((residual,i,anchor,lo,hi))
    accepted=[]
    for residual,i,anchor,lo,hi in sorted(proposals):
        target=owner[next(iter(anchors[anchor]))]
        # Unregistered fragments cannot attach to a gallery-registered trunk.
        target_gallery={reference_matches[n][0] for n in clusters[target] if n in reference_matches}
        if not c.get('rule_geometry'):
            if c.get('gallery_veto') and target_gallery and i not in reference_matches:continue
            if c.get('gallery_veto') and i in reference_matches and target_gallery and reference_matches[i][0] not in target_gallery:continue
        # Include earlier attachments in collision checks, never as new interpolation evidence.
        if any(geometry(tracks[i],tracks[n],c)[1] is not None for n in clusters[target]):continue
        old=owner[i];rep=min(clusters[target])
        accepted.append(dict(from_oid=i,to_oid=rep,mode='sparse_bridge',accepted=True,
                             residual=residual,bracket=[lo,hi],requires_review=True,
                             left_members=[i],right_members=sorted(clusters[target])))
        clusters[target].add(i);owner[i]=target;del clusters[old]
    return accepted


def rule_geometry_merge(tracks,owner,clusters,c,nodes):
    """Appearance-free unique continuation. Quarantine may support a path; it never joins the identity."""
    if not c.get('rule_geometry') and not c.get('quarantine_bridge'):return []
    normal=occupancy_index(nodes,False);quarantine=occupancy_index(nodes,True)
    fps=c.get('fps',30);speed=c.get('max_speed_m_s',9.)
    accepted=[]
    while True:
        proposals=[]
        cids=list(clusters)
        for i,x in enumerate(cids):
            for y in cids[i+1:]:
                best=None;conflict=False
                for a in clusters[x]:
                    for b in clusters[y]:
                        g,reason=geometry(tracks[a],tracks[b],c)
                        if reason in ('same_frame','kit_conflict'):conflict=True;break
                        if reason:continue
                        if c.get('mixed_box_exclude') and adjacent_rival(tracks[a],tracks[b],c):continue
                        n_hits,q_hits=path_hits(tracks[a],tracks[b],g,normal,quarantine,c)
                        if n_hits:continue
                        short=g['gap']<=c.get('rule_max_gap',60) and g['distance']<=c.get('rule_max_m',3.5)
                        supported=bool(c.get('quarantine_bridge') and q_hits and g['gap']<=c.get('quarantine_max_gap',120)
                                       and g['distance']<=c.get('rule_max_m',3.5))
                        if not short and not supported:continue
                        score=g['gap']/fps+g['distance']/speed
                        item=(score,g['gap'],g['distance'],a,b,short,[h['oid'] for h in q_hits])
                        if best is None or item[0]<best[0]:best=item
                    if conflict:break
                if conflict or best is None:continue
                proposals.append((best[0],x,y,best))
        if not proposals:break
        best_of={};second={}
        for score,x,y,best in proposals:
            for src,dst in ((x,y),(y,x)):
                cur=best_of.get(src)
                if cur is None or score<cur[0]:
                    if cur is not None:second[src]=cur
                    best_of[src]=(score,dst,best,x,y)
                elif src not in second or score<second[src][0]:
                    second[src]=(score,dst,best,x,y)
        chosen=[];seen=set()
        for src,(score,dst,best,x,y) in best_of.items():
            other=best_of.get(dst)
            if other is None or other[1]!=src:continue
            alt=second.get(src)
            if alt is not None and alt[1]!=dst and alt[0]-score<0.2:continue
            key=tuple(sorted((src,dst)))
            if key in seen:continue
            seen.add(key);chosen.append((score,x,y,best))
        if not chosen:break
        score,x,y,best=min(chosen)
        a,b=best[3],best[4]
        frames=lambda cid:sum(len(tracks[i]['rows']) for i in clusters[cid])
        lo,hi=(x,y) if frames(x)>=frames(y) else (y,x)
        mode='quarantine_bridge' if (c.get('quarantine_bridge') and best[6] and not best[5]) else 'rule_geometry'
        accepted.append(dict(from_oid=a,to_oid=b,gap=best[1],distance=best[2],mode=mode,accepted=True,
                             quarantine_support=sorted(set(best[6])),requires_review=True,
                             left_members=sorted(clusters[x]),right_members=sorted(clusters[y])))
        clusters[lo]|=clusters[hi]
        for i in clusters[hi]:owner[i]=lo
        del clusters[hi]
    return accepted


def lerp_gap(gap,near,far,span):
    """near at gap=0, far at gap>=span. Shorter gap -> stricter if near>far."""
    if span<=0:return far
    t=min(max(float(gap),0.)/float(span),1.)
    return near+(far-near)*t


def appearance_floor(geo,agree,c,gallery_a=None,gallery_b=None):
    short=geo['gap']<=60
    if c.get('appearance_floor') is None:
        return c['min_affinity']
    if agree and c.get('gallery_floor') is not None:return c['gallery_floor']
    if c.get('gap_scaled_floor'):
        base=lerp_gap(geo['gap'],c.get('appearance_floor_near',.76),c.get('appearance_floor_far',.60),
                      c.get('appearance_floor_span',180))
        one=(gallery_a is None)^(gallery_b is None)
        if one:
            base+=lerp_gap(geo['gap'],c.get('unmatched_premium_near',.04),c.get('unmatched_premium_far',0.),
                           c.get('appearance_floor_span',180))
        return float(min(.95,base))
    if short:return c.get('short_appearance_floor') if c.get('short_appearance_floor') is not None else c['appearance_floor']
    none=gallery_a is None and gallery_b is None
    one=(gallery_a is None) ^ (gallery_b is None)
    if none and c.get('unregistered_long_floor') is not None:return c['unregistered_long_floor']
    if one and c.get('unmatched_to_registered_long_floor') is not None:return c['unmatched_to_registered_long_floor']
    long=c.get('long_appearance_floor')
    return long if long is not None else c['appearance_floor']


def centroid_floor(c):
    if c.get('appearance_floor') is None:return c['min_centroid']
    return min(c['min_centroid'],.55)


def unique_best_pairs(proposals,margin,compatible,cluster_gallery=None):
    """Mutual unique best. Same-gallery partners ignore unmatched seconds; long-gap unmatched pairs treat close seconds as rivals."""
    cluster_gallery=cluster_gallery or (lambda cid:None)
    best={};second={}
    for (a,b),e in proposals.items():
        score=e['combined']
        for src,dst in ((a,b),(b,a)):
            cur=best.get(src)
            if cur is None or score>cur[0]:
                if cur is not None:second[src]=cur
                best[src]=(score,dst,e)
            elif src not in second or score>second[src][0]:
                second[src]=(score,dst,e)
    def rival(src,dst,score,edge):
        alt=second.get(src)
        if alt is None or alt[1]==dst or score-alt[0]>=margin:return False
        gs,gd=cluster_gallery(src),cluster_gallery(dst)
        if gs is not None and gs==gd:return False
        if gs is not None and cluster_gallery(alt[1])==gs:return False
        if edge.get('gap',10**9)>60:return True
        return not compatible(dst,alt[1])
    chosen=[];seen=set()
    for src,(score,dst,e) in best.items():
        other=best.get(dst)
        if other is None or other[1]!=src:continue
        if rival(src,dst,score,e) or rival(dst,src,other[0],e):continue
        key=tuple(sorted((src,dst)))
        if key in seen:continue
        seen.add(key);chosen.append((src,dst,e))
    return chosen


def _jersey_number(votes,oid):
    if not votes:return None
    v=votes.get(oid,votes.get(str(oid)))
    if isinstance(v,dict):v=v.get('number')
    return int(v) if v is not None else None


def recover(nodes,config=None,references=None):
    c=dict(DEFAULTS,**(config or {}));tracks={t['id']:t for t in group_tracks(nodes)}
    occ=occupancy_index(nodes) if c.get('occupancy_veto') else {}
    mixed=set()
    desc={i:describe(t,c) for i,t in tracks.items()};pairs={};candidates=[];gta_candidates=[];rejected=Counter()
    reference_matches={}
    jersey_votes=c.get('jersey_votes') or {}
    if c.get('jersey_votes_file') and not jersey_votes:
        from jersey_vote import load_votes
        jersey_votes=load_votes(c['jersey_votes_file'])
    seed=c.get('seed_mapping') or {}
    if c.get('seed_mapping_file') and not seed:
        seed={int(k):int(v) for k,v in json.loads(Path(c['seed_mapping_file']).read_text(encoding='utf-8')).items()}
    seed={int(k):int(v) for k,v in seed.items()}
    if references is not None:
        for i,d in desc.items():
            if d is None:continue
            scores=d['centroid']@references.T;order=np.argsort(-scores)
            vote=float(np.mean(np.argmax(d['features']@references.T,axis=1)==order[0]))
            if scores[order[0]]>=c['reference_threshold'] and scores[order[0]]-scores[order[1]]>=c['reference_margin'] and vote>=c['reference_vote']:
                reference_matches[i]=(int(order[0]),float(scores[order[0]]))
    for a,b in combinations(sorted(tracks),2):
        geo,reason=geometry(tracks[a],tracks[b],c)
        x,y=desc[a],desc[b]
        affinity,centroid,colour,source=pair_appearance(tracks[a],tracks[b],x,y,geo,c)
        raw_affinity=affinity
        gallery_a=reference_matches[a][0] if a in reference_matches else None
        gallery_b=reference_matches[b][0] if b in reference_matches else None
        jersey_a=_jersey_number(jersey_votes,a);jersey_b=_jersey_number(jersey_votes,b)
        jersey_agree=bool(c.get('jersey_agree') and jersey_a is not None and jersey_b is not None and jersey_a==jersey_b)
        # Same number is gallery-like agreement. Different numbers never veto: OCR is noisy.
        reference_agreement=(gallery_a is not None and gallery_a==gallery_b) or jersey_agree
        gallery_conflict=c.get('gallery_veto') and gallery_a is not None and gallery_b is not None and gallery_a!=gallery_b
        rival=bool(c.get('mixed_box_exclude') and reason is None and adjacent_rival(tracks[a],tracks[b],c))
        if reference_agreement and c.get('gallery_rewrite_affinity',True):
            evidence=min(reference_matches[a][1],reference_matches[b][1])
            if affinity is not None and evidence>affinity:
                affinity=evidence;centroid=max(centroid,evidence);source='frozen_training_gallery'
        pairs[a,b]=dict(geometry_ok=reason is None and not gallery_conflict and not rival,affinity=affinity,centroid=centroid,
                        gallery_a=gallery_a,gallery_b=gallery_b)
        if reason:rejected[reason]+=1;continue
        if rival:rejected['adjacent_rival']+=1;continue
        if c.get('occupancy_veto') and occupancy_blocks(tracks[a],tracks[b],geo,occ,c):
            rejected['occupancy']+=1;continue
        if gallery_conflict:rejected['gallery_conflict']+=1;continue
        if affinity is None:rejected['insufficient_samples']+=1;continue
        if colour is None or colour<c['min_colour']:rejected['colour']+=1;continue
        gta_d=mean_pairwise_distance(x,y)
        additive=bool(c.get('gta_additive'))
        gta_only=bool(c.get('gta_connect')) and not additive
        motion=math.exp(-geo['distance']/(c['distance_slack_m']+c['max_speed_m_s']*geo['gap']/c['fps']))
        short=geo['gap']<=60
        a3_score=(.8*affinity+.1*colour+.1*motion) if short else (.9*affinity+.1*colour)
        edge=dict(from_oid=a,to_oid=b,**geo,affinity=affinity,centroid=centroid,
                  appearance_source=source,col_sim=colour,raw_affinity=raw_affinity,
                  reference_agreement=reference_agreement,jersey_agree=jersey_agree,
                  jersey_a=jersey_a,jersey_b=jersey_b,gta_distance=gta_d)
        if not gta_only:
            floor=appearance_floor(geo,reference_agreement,c,gallery_a,gallery_b)
            if affinity<floor or centroid<centroid_floor(c):rejected['appearance']+=1
            else:candidates.append(dict(edge,combined=a3_score,mode='short' if short else 'reentry'))
        if gta_only or additive:
            if gta_d is None or gta_d>=c.get('gta_merge_dist',.4):rejected['gta_distance']+=1
            else:gta_candidates.append(dict(edge,combined=1.-gta_d,mode='gta_connect'))
    clusters={i:{i} for i in tracks};owner={i:i for i in tracks};accepted=[]
    if seed:
        grouped={}
        def seeded_root(i):
            r=int(seed.get(i,i));seen=set()
            while True:
                nxt=int(seed.get(r,r))
                if nxt==r or nxt in seen:return r
                seen.add(r);r=nxt
        for i in tracks:
            grouped.setdefault(seeded_root(i),set()).add(i)
        clusters={min(v):set(v) for v in grouped.values()}
        owner={i:rep for rep,mem in clusters.items() for i in mem}
    def compatible(x,y,gta=False):
        for a in clusters[x]:
            for b in clusters[y]:
                p=pairs[min(a,b),max(a,b)]
                if not p['geometry_ok'] or p['affinity'] is None:return False
                if gta:continue
                if p['centroid']<c['component_min_centroid'] or p['affinity']<c['component_min_affinity']:return False
        return True
    def absorb_need(e):
        if c.get('gap_scaled_floor'):
            return lerp_gap(e.get('gap',180),c.get('absorb_near',.86),c.get('absorb_far',.66),
                            c.get('appearance_floor_span',180))
        return c.get('absorb_affinity',.80)
    def absorb_ok(x,y,e):
        if not c.get('mutual_unique'):return True
        if len(clusters[x])==1 and len(clusters[y])==1:return True
        return bool(e.get('reference_agreement') or e['affinity']>=absorb_need(e))
    additive=bool(c.get('gta_additive'));gta_only=bool(c.get('gta_connect')) and not additive
    rule_only=bool(c.get('rule_only'))
    while not gta_only and not rule_only:
        proposals={}
        for e in candidates:
            a,b=owner[e['from_oid']],owner[e['to_oid']]
            if a==b:continue
            key=tuple(sorted((a,b)))
            if key not in proposals or e['combined']>proposals[key]['combined']:proposals[key]=e
        proposals={k:e for k,e in proposals.items() if compatible(*k) and absorb_ok(*k,e)}
        if not proposals:break
        if c.get('mutual_unique'):
            def cluster_gallery(cid):
                tags={reference_matches[i][0] for i in clusters[cid] if i in reference_matches}
                return next(iter(tags)) if len(tags)==1 else None
            ranked=unique_best_pairs(proposals,c['ambiguity_margin'],compatible,cluster_gallery)
            if not ranked:break
            used=set();progress=False
            for a,b,e in ranked:
                a,b=owner[e['from_oid']],owner[e['to_oid']]
                if a==b or a in used or b in used or not compatible(a,b) or not absorb_ok(a,b,e):continue
                lo,hi=min(a,b),max(a,b)
                accepted.append(dict(e,accepted=True,left_members=sorted(clusters[a]),right_members=sorted(clusters[b])))
                clusters[lo]|=clusters[hi]
                for i in clusters[hi]:owner[i]=lo
                del clusters[hi];used.add(lo);progress=True
            if not progress:break
            continue
        chosen=None
        for (a,b),e in sorted(proposals.items(),key=lambda p:(-p[1]['combined'],p[0])):
            ambiguous=False
            for (x,y),other in proposals.items():
                if (x,y)==(a,b):continue
                for anchor,target in [(a,b),(b,a)]:
                    rival=y if x==anchor else x if y==anchor else None
                    if rival is not None and rival!=target and not compatible(target,rival):
                        if e['combined']-other['combined']<c['ambiguity_margin']:ambiguous=True
            if not ambiguous:chosen=(a,b,e);break
        if chosen is None:break
        a,b,e=chosen;lo,hi=min(a,b),max(a,b)
        accepted.append(dict(e,accepted=True,left_members=sorted(clusters[a]),right_members=sorted(clusters[b])))
        clusters[lo]|=clusters[hi]
        for i in clusters[hi]:owner[i]=lo
        del clusters[hi]
    if (gta_only or additive) and not rule_only:
        alpha=float(c.get('gta_merge_dist',.4))
        while True:
            best=None
            for e in gta_candidates:
                a,b=owner[e['from_oid']],owner[e['to_oid']]
                if a==b or not compatible(a,b,gta=True):continue
                d=cluster_mean_distance(clusters[a],clusters[b],desc)
                if d>=alpha:continue
                key=(d,min(a,b),max(a,b))
                if best is None or key<best[0]:best=(key,a,b,e,d)
            if best is None:break
            _,a,b,e,d=best;lo,hi=min(a,b),max(a,b)
            accepted.append(dict(e,accepted=True,gta_distance=d,mode='gta_connect',
                                 left_members=sorted(clusters[a]),right_members=sorted(clusters[b])))
            clusters[lo]|=clusters[hi]
            for i in clusters[hi]:owner[i]=lo
            del clusters[hi]
    accepted.extend(rule_geometry_merge(tracks,owner,clusters,c,nodes))
    accepted.extend(bridge_sparse(tracks,owner,clusters,desc,c,reference_matches))
    # Canonical output IDs depend on membership, not on the order of sparse attachments.
    owner={i:min(clusters[owner[i]]) for i in tracks}
    return owner,accepted,dict(candidates=candidates,rejections=dict(rejected),config=c,
                              normal_ids_before=len(tracks),normal_ids_after=len(clusters),
                              occupancy_used=bool(c.get('occupancy_veto')),mixed_box_exclude=bool(c.get('mixed_box_exclude')),
                              jersey_agree=bool(c.get('jersey_agree')),
                              jersey_voted_tracks=sorted(i for i in tracks if _jersey_number(jersey_votes,i) is not None),
                              jersey_agree_candidates=sum(1 for e in candidates if e.get('jersey_agree')),
                              gta_connect=bool(c.get('gta_connect')),gta_additive=bool(c.get('gta_additive')),
                              gta_merge_dist=c.get('gta_merge_dist',.4),
                              gta_merges=sum(1 for e in accepted if e.get('mode')=='gta_connect'),
                              rule_geometry=bool(c.get('rule_geometry')),
                              rule_only=bool(c.get('rule_only')),
                              seed_used=bool(seed),
                              rule_merges=sum(1 for e in accepted if e.get('mode') in ('rule_geometry','quarantine_bridge')),
                              labels_used_for_association=references is not None,query_identity_map_used=False,
                              training_gallery_used=references is not None,
                              reference_matched_tracks=sorted(reference_matches))


def run_second_stage(audit_path,output_dir,config=None):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'association_report.json').exists():raise ValueError('Second-stage output already exists')
    with Path(audit_path).open('rb') as f:audit=pickle.load(f)
    expected=int((config or {}).get('expected_dimension',3840))
    if audit.get('feature_metadata',{}).get('dimension')!=expected:raise ValueError('Real TransReID features are required')
    references=None
    if (config or {}).get('use_training_references') and not (config or {}).get('reference_file'):
        raise ValueError('Requested training references but no checkpoint-bound reference bank was provided')
    cfg=dict(config or {})
    if cfg.get('jersey_votes_file') and not cfg.get('jersey_votes'):
        from jersey_vote import load_votes
        cfg['jersey_votes']=load_votes(cfg['jersey_votes_file'])
    if (cfg or {}).get('reference_file'):
        with np.load(cfg['reference_file'],allow_pickle=False) as bank:
            if str(bank['checkpoint'])!=audit['feature_metadata']['fingerprint']['checkpoint']:raise ValueError('Reference bank checkpoint mismatch')
            references=bank['prototypes']
            if references.ndim!=2 or references.shape[1]!=expected or not np.isfinite(references).all():raise ValueError('Invalid reference bank')
            if not (config or {}).get('allow_new_match_references'):
                if int(bank['max_frame'])>1049:raise ValueError('Held-out reference bank')
                if references.shape!=(9,3840):raise ValueError('Invalid reference bank')
    mapping,accepted,report=recover(audit['nodes'],cfg,references)
    rows=write_result(audit,mapping,out)
    report.update(input_rows=len(audit['rows']),output_rows=len(rows),same_frame_conflicts=0,
                  accepted_edges=len(accepted),feature_metadata=audit['feature_metadata'],
                  automatic_training_labels=False)
    for name,obj in [('trajectory_mapping.json',mapping),('accepted_edges.json',accepted),('association_report.json',report)]:
        (out/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
    return dict(mapping=mapping,accepted_edges=accepted,report=report,output_tracking=str(out/'tracking_reid_trajectory.txt'))
