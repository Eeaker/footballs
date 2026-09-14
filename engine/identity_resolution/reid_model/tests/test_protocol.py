import csv
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from evaluation import evaluate,calibrate_threshold,retrieval
from train_adapt import BalancedTemporalSampler,load_rows


def test_real_split_disjoint_and_embargoed():
    rows=load_rows()
    keys=[(r['frame'],r['source_local_id']) for r in rows]
    assert len(set(keys))==len(keys)
    assert len({r['sha256'] for r in rows})==len(rows)
    fs={s:[r['frame'] for r in rows if r['split']==s] for s in ['train','val','test']}
    assert min(fs['val'])-max(fs['train'])>60
    assert min(fs['test'])-max(fs['val'])>60
    assert all(r['split']=='train' for r in rows if r['enrolment_gallery'])
    for s in fs:assert {r['pid'] for r in rows if r['split']==s}==set(range(9))


@pytest.mark.parametrize('k',[4,8])
def test_balanced_sampler_has_no_duplicate_images(k):
    rows=[r for r in load_rows() if r['split']=='train']
    for batch in BalancedTemporalSampler(rows,k,4,123):
        assert len(batch)==len(set(batch))==9*k
        assert set(Counter(rows[i]['pid'] for i in batch).values())=={k}


def test_single_camera_positives_are_kept_and_same_id_filter_is_explicit():
    q=np.array([[1.,0.],[0.,1.]])
    gr=[dict(pid=0,algorithm_id=1,frame=0,image='a',kit=1),dict(pid=1,algorithm_id=2,frame=0,image='b',kit=1)]
    qr=[dict(pid=0,algorithm_id=3,source_local_id=30,frame=100,image='c',kit=1),dict(pid=1,algorithm_id=4,source_local_id=40,frame=100,image='d',kit=1)]
    m=evaluate(q,q,qr,gr,calibrate=True)
    assert m['image_retrieval']['rank1_macro']==1
    assert m['cross_algorithm_retrieval']['mAP_macro']==1
    assert m['TAR']==1 and m['same_kit_FAR']==0
    qr[0]['algorithm_id']=1
    assert retrieval(q,q,qr,gr,True)['valid_queries']==1


def test_threshold_never_exceeds_target_empirical_false_acceptance():
    negatives=np.array([.1]*30+[.5]*40+[.9]*30)
    threshold=calibrate_threshold(negatives,.01)
    assert np.mean(negatives>=threshold)<=.01
