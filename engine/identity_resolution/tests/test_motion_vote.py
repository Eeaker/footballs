import numpy as np
from motion_vote import predict,evidence,recover_one_sided_review


def track(i,frames,q=False):
    return dict(id=i,output_id=i,quarantine=q,kit=0,kits={0},start=min(frames),end=max(frames),
                rows=[(f,i,500,400,40,80,.9) for f in frames],positions={f:np.array([20.,10.]) for f in frames},
                samples=[(f,np.array([1.,0.]),np.array([1.,0.])) for f in frames])


def test_kalman_constant_velocity_prediction():
    obs=[(t,np.array([t,2*t])) for t in np.arange(0,2,.05)]
    mean,cov=predict(obs,[2.1],.1,1.)[0]
    assert np.linalg.norm(mean-np.array([2.1,4.2]))<.1
    assert np.linalg.eigvalsh(cov).min()>0


def test_two_sided_evidence_requires_both_anchors():
    assert evidence(track(1,[40,41,42]),track(2,list(range(30))),'metric') is None


def test_one_sided_review_rejects_equal_rivals_and_quarantine():
    ns=[track(1,[160,161,162]),track(2,list(range(150))),track(3,list(range(150)))]
    assert recover_one_sided_review(ns,{1:1,2:2,3:3})[1]==[]
    ns[1]['quarantine']=True;ns[2]['quarantine']=True
    assert recover_one_sided_review(ns,{1:1})[1]==[]


def test_single_observation_cannot_win_motion_vote():
    assert recover_one_sided_review([track(1,[160]),track(2,list(range(150)))],{1:1,2:2})[1]==[]
