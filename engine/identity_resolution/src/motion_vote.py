"""Frozen, two-sided Kalman proposals; no identity labels and no quarantine anchors."""
import math
from itertools import combinations
import numpy as np
from second_stage import geometry,DEFAULTS,describe,appearance
from trajectory_reassociation import group_tracks


def predict(observations, times, sigma, acceleration):
    """CV Kalman filter. Times are seconds; backwards time is reflected by caller."""
    observations=sorted(observations,key=lambda p:p[0])
    if len(observations)<3:return None
    state=np.r_[observations[0][1],0.,0.]
    covariance=np.diag([sigma**2]*2+[(acceleration*2)**2]*2)
    h=np.c_[np.eye(2),np.zeros((2,2))];r=np.eye(2)*sigma**2
    def step(x,p,dt):
        f=np.eye(4);f[:2,2:]=np.eye(2)*dt
        g=np.r_[np.eye(2)*dt**2/2,np.eye(2)*dt]
        return f@x,f@p@f.T+g@g.T*acceleration**2
    previous=observations[0][0]
    for t,z in observations[1:]:
        state,covariance=step(state,covariance,t-previous)
        s=h@covariance@h.T+r;k=np.linalg.solve(s,h@covariance).T
        state+=k@(z-h@state)
        a=np.eye(4)-k@h;covariance=a@covariance@a.T+k@r@k.T
        previous=t
    result=[]
    for t in times:
        if t<previous:raise ValueError('Prediction must follow observed history')
        x,p=step(state,covariance,t-previous)
        result.append((x[:2],h@p@h.T+r))
    return result


def points(track,space):
    if space=='metric':return track['positions']
    return {r[0]:np.array([r[2]+r[4]/2,r[3]+r[5]],float) for r in track['rows']}


def evidence(query,anchor,space,max_side_gap=60):
    qp,ap=points(query,space),points(anchor,space)
    before=sorted(f for f in ap if f<query['start'])
    after=sorted(f for f in ap if f>query['end'])
    if len(before)<3 or len(after)<3:return None
    if query['start']-before[-1]>max_side_gap or after[0]-query['end']>max_side_gap:return None
    before=[f for f in before if f>=before[-1]-30]
    after=[f for f in after if f<=after[0]+30]
    frames=sorted(qp)
    if not frames:return None
    frames=[frames[i] for i in np.unique(np.linspace(0,len(frames)-1,min(5,len(frames))).astype(int))]
    sigma,acc=(.35,3.) if space=='metric' else (8.,80.)
    forward=predict([(f/30,ap[f]) for f in before],[f/30 for f in frames],sigma,acc)
    backward=predict([(-f/30,ap[f]) for f in after],[-f/30 for f in frames],sigma,acc)
    if forward is None or backward is None:return None
    distances=[];residuals=[]
    for f,left,right in zip(frames,forward,backward):
        ds=[];rs=[]
        for mean,cov in (left,right):
            error=qp[f]-mean
            ds.append(float(error@np.linalg.solve(cov,error)));rs.append(float(np.linalg.norm(error)))
        distances.append(max(ds));residuals.append(max(rs))
    return dict(space=space,frames=frames,nis=distances,max_residual=max(residuals),
                score=float(np.mean(np.exp(-np.array(distances)/2))),
                bracket=[before[-1],after[0]])


def recover_motion(nodes,mapping,mode='both'):
    if mode not in ('metric','image','both'):raise ValueError(mode)
    tracks=group_tracks([dict(n,output_id=mapping.get(n['output_id'],n['output_id'])) for n in nodes])
    anchors=[t for t in tracks if len(t['rows'])>=150 and len(t['samples'])>=3]
    diagnostics=[];proposals=[]
    for q in tracks:
        if len(q['rows'])>300:continue
        candidates=[]
        for a in anchors:
            if a['id']==q['id'] or len(a['rows'])<2*len(q['rows']):continue
            geo,reason=geometry(q,a,DEFAULTS)
            if reason:continue
            metric=evidence(q,a,'metric');pixel=evidence(q,a,'image')
            if metric is None or pixel is None:continue
            candidates.append(dict(target=a['id'],metric=metric,image=pixel))
        decisions={}
        for space in ('metric','image'):
            ranked=sorted(candidates,key=lambda c:-c[space]['score'])
            chosen=None
            if ranked:
                best=ranked[0][space]
                # Votes describe temporal consistency, not independent probability.
                vote=float(np.mean(np.array(best['nis'])<=4.605))
                margin=best['score']-(ranked[1][space]['score'] if len(ranked)>1 else 0.)
                if len(best['frames'])>=3 and vote>=.8 and max(best['nis'])<=9.21 and best['score']>=.35 and margin>=.15:
                    chosen=ranked[0]['target']
            decisions[space]=chosen
        chosen=decisions[mode] if mode!='both' else (decisions['metric'] if decisions['metric']==decisions['image'] else None)
        diagnostics.append(dict(query=q['id'],candidates=candidates,decisions=decisions,chosen=chosen))
        if chosen is not None:proposals.append((q['id'],chosen))
    # Frozen candidates; accepted fragments cannot manufacture additional anchors.
    result=dict(mapping);accepted=[];by_id={t['id']:t for t in tracks};members={t['id']:{t['id']} for t in tracks}
    owner={t['id']:t['id'] for t in tracks}
    for q,target in proposals:
        left,right=owner[q],owner[target]
        if left==right:continue
        if any(geometry(by_id[a],by_id[b],DEFAULTS)[1] for a in members[left] for b in members[right]):continue
        merged=members[left]|members[right];root=min(merged)
        for i in merged:owner[i]=root
        del members[left];del members[right];members[root]=merged
        accepted.append(dict(from_oid=q,to_oid=target,mode='kalman_'+mode,requires_review=True,automatic_training_label=False))
    result={i:owner[v] for i,v in result.items()}
    return result,accepted,diagnostics


def recover_split_brackets(nodes,mapping,mode='metric'):
    """Jointly propose clean left/right trunks around a short fragment.

    Both trunks must independently be compatible; they need not already share an ID.
    No proposal can create evidence for another proposal in this pass.
    """
    tracks=group_tracks([dict(n,output_id=mapping.get(n['output_id'],n['output_id'])) for n in nodes])
    anchors=[t for t in tracks if len(t['rows'])>=150 and len(t['samples'])>=3]
    proposals=[];diagnostics=[]
    for q in tracks:
        if q['end']-q['start']>45 or len(q['samples'])>=2:continue
        candidates=[]
        for a,b in combinations(anchors,2):
            if q['id'] in (a['id'],b['id']):continue
            if any(geometry(x,y,DEFAULTS)[1] for x,y in ((a,b),(q,a),(q,b))):continue
            aff,cent,colour=appearance(describe(a,DEFAULTS),describe(b,DEFAULTS))
            # This is a motion-only review alternative, not permission to train.
            # Keep appearance conflicts visible instead of silently forcing agreement.
            if colour is None or colour<.35:continue
            merged=dict(a,rows=sorted(a['rows']+b['rows']),positions=a['positions']|b['positions'],
                        start=min(a['start'],b['start']),end=max(a['end'],b['end']))
            values={s:evidence(q,merged,s) for s in ('metric','image')}
            if any(v is None for v in values.values()):continue
            spaces=('metric','image') if mode=='both' else (mode,)
            score=min(values[s]['score'] for s in spaces)
            ok=all(len(values[s]['frames'])>=3 and max(values[s]['nis'])<=9.21 and
                   np.mean(np.array(values[s]['nis'])<=4.605)>=.8 for s in spaces)
            candidates.append(dict(targets=[a['id'],b['id']],score=score,ok=bool(ok),appearance=aff,
                                   appearance_conflict=aff is None or aff<.72 or cent<.65,**values))
        candidates.sort(key=lambda c:-c['score'])
        accepted=bool(candidates and candidates[0]['ok'] and candidates[0]['score']>=.35 and
                      (len(candidates)==1 or candidates[0]['score']-candidates[1]['score']>=.15))
        diagnostics.append(dict(query=q['id'],candidates=candidates,accepted=accepted))
        if accepted:proposals.append((q['id'],candidates[0]['targets']))
    owner={t['id']:t['id'] for t in tracks};members={i:{i} for i in owner};lookup={t['id']:t for t in tracks};edges=[]
    for q,targets in proposals:
        roots={owner[i] for i in [q]+targets};all_ids=set().union(*(members[r] for r in roots))
        if any(geometry(lookup[a],lookup[b],DEFAULTS)[1] for a,b in combinations(all_ids,2)):continue
        root=min(all_ids)
        for r in roots:del members[r]
        members[root]=all_ids
        for i in all_ids:owner[i]=root
        edges.append(dict(from_oid=q,to_oid=targets[0],other_trunk=targets[1],mode='split_bracket_'+mode,
                          requires_review=True,automatic_training_label=False))
    return {i:owner[v] for i,v in mapping.items()},edges,diagnostics


def recover_one_sided_review(nodes,mapping,max_gap=150):
    """Experimental metre-coordinate review proposals, never automatic labels.

    Long-horizon NIS can look excellent merely because covariance grows. Rank
    absolute prediction error instead, disclose covariance and require a margin.
    """
    tracks=group_tracks([dict(n,output_id=mapping.get(n['output_id'],n['output_id'])) for n in nodes])
    proposals=[];diagnostics=[]
    for q in tracks:
        candidates=[]
        for a in tracks:
            if a['id']==q['id'] or len(a['rows'])<150 or len(a['rows'])<=len(q['rows']):continue
            if geometry(q,a,DEFAULTS)[1]:continue
            sides=[]
            for sign in (1,-1):
                qp={sign*f:p for f,p in q['positions'].items()};ap={sign*f:p for f,p in a['positions'].items()}
                frames=sorted(qp);history=sorted(f for f in ap if f<frames[0])
                if not history or frames[0]-history[-1]>max_gap:continue
                history=[f for f in history if f>=history[-1]-30]
                chosen=frames[:5]
                prediction=predict([(f/30,ap[f]) for f in history],[f/30 for f in chosen],.35,3.)
                if prediction is None:continue
                errors=[float(np.linalg.norm(qp[f]-mu)) for f,(mu,cov) in zip(chosen,prediction)]
                sigma=max(float(np.sqrt(np.linalg.eigvalsh(cov).max())) for mu,cov in prediction)
                sides.append(dict(direction='forward' if sign==1 else 'backward',gap=frames[0]-history[-1],
                                  errors_m=errors,prediction_sigma_m=sigma))
            if not sides:continue
            error=max(max(s['errors_m']) for s in sides)
            candidates.append(dict(target=a['id'],error_m=error,sides=sides))
        candidates.sort(key=lambda c:c['error_m'])
        accepted=bool(candidates and len(q['rows'])>=3 and candidates[0]['error_m']<=1.0 and
                      (len(candidates)==1 or candidates[1]['error_m']-candidates[0]['error_m']>=.5))
        diagnostics.append(dict(query=q['id'],candidates=candidates,accepted=accepted))
        if accepted:proposals.append((q['id'],candidates[0]['target']))
    owner={t['id']:t['id'] for t in tracks};members={i:{i} for i in owner};lookup={t['id']:t for t in tracks};edges=[]
    for q,target in proposals:
        roots={owner[q],owner[target]}
        if len(roots)<2:continue
        all_ids=set().union(*(members[r] for r in roots))
        if any(geometry(lookup[a],lookup[b],DEFAULTS)[1] for a,b in combinations(all_ids,2)):continue
        root=min(all_ids)
        for r in roots:del members[r]
        members[root]=all_ids
        for i in all_ids:owner[i]=root
        edges.append(dict(from_oid=q,to_oid=target,mode='one_sided_motion_review',requires_review=True,
                          automatic_training_label=False,uncalibrated_uncertainty=True))
    return {i:owner[v] for i,v in mapping.items()},edges,diagnostics
