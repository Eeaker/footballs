"""Single-camera known-player retrieval and held-out track association metrics."""
from collections import defaultdict
import numpy as np


def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def retrieval(q, gallery, qr, gr, cross_algorithm=False):
    scores=unit(q)@unit(gallery).T
    records=[]
    for i,row in enumerate(qr):
        keep=np.array([(row['pid']!=r['pid'] or abs(row['frame']-r['frame'])>60) and row['image']!=r['image']
                       and (not cross_algorithm or row['algorithm_id']!=r['algorithm_id']) for r in gr])
        positions=np.where(keep)[0]
        positions=positions[np.argsort(-scores[i,positions],kind='stable')]
        relevant=np.array([gr[j]['pid']==row['pid'] for j in positions])
        if not relevant.any():continue
        precision=np.cumsum(relevant)/np.arange(1,len(relevant)+1)
        records.append((row['pid'],float(relevant[0]),float(precision[relevant].mean())))
    bypid=defaultdict(list)
    for pid,r1,ap in records:bypid[pid].append((r1,ap))
    return dict(valid_queries=len(records),total_queries=len(qr),covered_identities=len(bypid),
                rank1_macro=float(np.mean([np.mean(v,axis=0)[0] for v in bypid.values()])) if bypid else None,
                mAP_macro=float(np.mean([np.mean(v,axis=0)[1] for v in bypid.values()])) if bypid else None)


def track_scores(q,gallery,qr,gr):
    groups=defaultdict(list)
    for i,r in enumerate(qr):groups[(r['pid'],r['algorithm_id'],r['source_local_id'])].append(i)
    # Separate query tracklets; one balanced enrolment prototype per known identity.
    pids=sorted({r['pid'] for r in gr})
    prototypes=unit(np.array([unit(gallery[[i for i,r in enumerate(gr) if r['pid']==p]]).mean(0) for p in pids]))
    positives=[];negatives=[];all_neg=[];out=[]
    kits={r['pid']:r['kit'] for r in gr}
    for key,idx in groups.items():
        pid,algorithm_id,source=key
        feature=unit(unit(q[idx]).mean(0,keepdims=True))
        scores=(feature@prototypes.T)[0]
        positive=float(scores[pids.index(pid)])
        positives.append(positive)
        negatives.extend(float(scores[j]) for j,p in enumerate(pids) if p!=pid and kits[p]==kits[pid] and kits[pid]>=0)
        all_neg.extend(float(scores[j]) for j,p in enumerate(pids) if p!=pid)
        out.append(dict(pid=pid,algorithm_id=algorithm_id,source_local_id=source,crops=len(idx),
                        predicted_pid=pids[int(np.argmax(scores))],positive_score=positive,
                        cross_algorithm=not any(r['algorithm_id']==algorithm_id for r in gr if r['pid']==pid)))
    return np.asarray(positives),np.asarray(negatives),np.asarray(all_neg),out


def calibrate_threshold(negatives,target=.01):
    if not len(negatives):return None
    values=np.sort(np.asarray(negatives,dtype=np.float64))
    index=max(0,min(len(values)-1,int(np.ceil((1-target)*len(values)))-1))
    return float(np.nextafter(values[index],np.inf))


def evaluate(q,gallery,qr,gr,threshold=None,calibrate=False):
    pos,neg,allneg,tracks=track_scores(q,gallery,qr,gr)
    if calibrate:threshold=calibrate_threshold(neg)
    grouped=defaultdict(list)
    for r in tracks:grouped[r['pid']].append(float(r['pid']==r['predicted_pid']))
    cross=[r for r in tracks if r['cross_algorithm']]
    strict=[i for i,r in enumerate(qr) if int(r.get('nearest_training_gap',61))>60]
    return dict(image_retrieval=retrieval(q,gallery,qr,gr),
                strict_time_separated_retrieval=retrieval(q[strict],gallery,[qr[i] for i in strict],gr,True) if strict else None,
                cross_algorithm_retrieval=retrieval(q,gallery,qr,gr,True),
                track_rank1_macro=float(np.mean([np.mean(v) for v in grouped.values()])),
                track_count=len(tracks),cross_algorithm_track_count=len(cross),
                cross_algorithm_track_rank1=float(np.mean([r['pid']==r['predicted_pid'] for r in cross])) if cross else None,
                same_kit_negative_pairs=len(neg),all_negative_pairs=len(allneg),positive_pairs=len(pos),
                threshold_from_validation=threshold,
                TAR=float(np.mean(pos>=threshold)) if threshold is not None else None,
                same_kit_FAR=float(np.mean(neg>=threshold)) if threshold is not None and len(neg) else None,
                all_FAR=float(np.mean(allneg>=threshold)) if threshold is not None else None,
                tracks=tracks,
                caveat='Small correlated single-clip sample; empirical FAR is not a population guarantee.')
