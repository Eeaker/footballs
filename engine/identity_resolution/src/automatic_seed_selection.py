"""Annotation-free seed selection by metric position, simultaneity and coverage."""
from collections import Counter,defaultdict
import math
import numpy as np


def select_seeds(audit,max_outfield=8,min_frames=150,goal_radius=7.,min_keeper_exclusive=60):
    records=defaultdict(list);live=defaultdict(dict);occupancy=defaultdict(set)
    for n in audit['nodes']:
        for r in n['rows']:
            xy=np.asarray(n['positions'][r[0]])
            distances=[float(np.linalg.norm(xy-g)) for g in [np.array([0,12.5]),np.array([45,12.5])]]
            for side,d in enumerate(distances):
                if d<goal_radius:occupancy[r[0],side].add(n['output_id'])
            if not n['quarantine']:
                records[n['output_id']].append((r,n,xy,distances))
                live[r[0]][n['output_id']]=(r,distances)
    stats={};keepers=[]
    for oid,rr in records.items():
        xy=np.asarray([x[2] for x in rr]);extent=np.quantile(xy,.95,axis=0)-np.quantile(xy,.05,axis=0)
        coverage=float(np.linalg.norm(extent));fs=sorted(x[0][0] for x in rr)
        stats[oid]=dict(observations=len(rr),coverage_m=coverage,first=fs[0],last=fs[-1],
            score=math.log1p(len(rr)/30)+math.log1p(coverage))
        side=int(np.argmin(np.median([x[3] for x in rr],axis=0)))
        exclusive=sorted(x[0][0] for x in rr if occupancy[x[0][0],side]=={oid})
        longest=current=0;prev=-2
        for f in exclusive:current=current+1 if f==prev+1 else 1;longest=max(longest,current);prev=f
        if len(rr)>=min_frames and all(x[3][side]<goal_radius for x in rr) and longest>=min_keeper_exclusive:
            keepers.append(dict(id=oid,side=side,sole_frames=exclusive,longest_exclusive=longest))
    keeper=max(keepers,key=lambda k:(len(k['sole_frames']),k['longest_exclusive'],-k['id'])) if keepers else None
    states=[]
    for f,people in sorted(live.items()):
        ids=[i for i,(r,distances) in people.items() if stats[i]['observations']>=min_frames and min(distances)>=goal_radius and r[6]>=.5]
        ids=tuple(sorted(sorted(ids,key=lambda i:(-stats[i]['score'],i))[:max_outfield]))
        if len(ids)<2:continue
        if states and states[-1]['ids']==ids and states[-1]['end']==f-1:states[-1]['end']=f
        else:states.append(dict(ids=ids,start=f,end=f,score=sum(stats[i]['score'] for i in ids)))
    sustained=[s for s in states if s['end']-s['start']+1>=15]
    if not sustained:raise ValueError('No sustained simultaneous identities')
    best=max(sustained,key=lambda s:(len(s['ids']),s['score'],s['end']-s['start'],-s['start']))
    return dict(outfield_ids=list(best['ids']),keeper=keeper,anchor_interval=[best['start'],best['end']],
                candidates=stats,criterion='max simultaneous >=15 frames outside 7m goal regions, then log duration + log metric coverage',
                labels_used=False,full_tracks=True,maximum_visible_goalkeepers=1)
