import numpy as np
import pytest
from features import apply_observation_policy,select_observations
from selftraining import SeedBalancedBatches


def test_reviewed_invalid_box_is_excluded_before_event_or_feature_use():
    n=dict(source_local=180,rows=[(353,180,0,0,30,50,.9)],quarantine=False,positions={353:[1,2]},samples=[1])
    a=apply_observation_policy({'nodes':[n]},{'invalid_detections':[dict(source_local_id=180,frames=[353])]})
    assert a['nodes'][0]['quarantine'] and a['nodes'][0]['invalid_detection']
    assert a['nodes'][0]['positions']=={} and select_observations(n,{}, {})==[]


def test_partial_node_cannot_silently_quarantine_other_observations():
    a={'nodes':[dict(source_local=180,rows=[(353,),(354,)])]}
    with pytest.raises(ValueError,match='Partial invalid node'):
        apply_observation_policy(a,{'invalid_detections':[dict(source_local_id=180,frames=[353])]})


def test_human_supplement_is_capped_like_pseudo_without_losing_provenance():
    rows=[dict(pid=p,origin='seed',frame=i*60) for p in range(9) for i in range(4)]
    rows += [dict(pid=0,origin='human_reviewed_hard',frame=i) for i in range(10)]
    batch=next(iter(SeedBalancedBatches(rows,2,1,42)))
    assert sum(i>=36 for i in batch)==1
    assert all(sum(rows[i]['pid']==p for i in batch)==2 for p in range(9))
