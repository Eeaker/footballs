"""Regression checks for label leakage, collisions and held-out self-training."""
import json
import pickle
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trajectory_reassociation import (expand_to_full_mapping,greedy_associate,
    build_trajectory_graph,write_result,run_reassociation)
from features import restrict_audit,overlap,select_observations
from selftraining import select_pseudo,SeedBalancedBatches,promotion_allowed,validate_windows,validate_dataset
from render_video import load_tracking_data,render_video_func
from second_stage import recover,geometry,DEFAULTS,run_second_stage


def node(oid,frames,kit=0):
    return dict(id=oid,output_id=oid,quarantine=False,kit=kit,start=min(frames),end=max(frames),
        rows=[(f,oid,10.,10.,30.,60.,.9) for f in frames],positions={f:np.zeros(2) for f in frames},
        samples=[(f,np.array([1.,0.]),np.array([1.,0.])) for f in frames])


def edge(a,b,score):return dict(from_oid=a,to_oid=b,combined=score)


def test_zero_edges_never_uses_human_identity_map():
    ns=[node(129,[0,1]),node(5,[4,5])]
    assert expand_to_full_mapping([],ns)=={5:5,129:129}
    assert expand_to_full_mapping([edge(129,5,.99)],ns)=={5:5,129:5}


def test_mutual_matching_never_branches():
    ns=[node(1,[0,1]),node(2,[4,5]),node(3,[4,5])]
    accepted,succ,pred=greedy_associate([edge(1,2,.99),edge(1,3,.98)],ns,margin=0)
    assert len(accepted)==1 and succ=={1:2} and pred=={2:1}
    assert greedy_associate([edge(1,2,.99),edge(1,3,.98)],ns,margin=.05)[0]==[]


def test_component_collision_rejected():
    ns=[node(1,[0,1]),node(2,[3,4]),node(3,[1,2])]
    es=[edge(1,2,.99),edge(2,3,.98)]
    assert len(greedy_associate(es,ns,margin=0)[0])==1
    with pytest.raises(ValueError):expand_to_full_mapping(es,ns)


def test_missing_geometry_and_invalid_features_fail_closed():
    ns=[node(1,[0,1]),node(2,[3,4])]
    assert len(build_trajectory_graph(ns,{}))==1
    ns[0]['positions']={}
    assert build_trajectory_graph(ns,{})==[]
    ns=[node(1,[0,1]),node(2,[3,4])];ns[0]['samples']=[(0,np.zeros(2),np.ones(2))]
    assert build_trajectory_graph(ns,{})==[]


def test_truth_filter_and_legacy_encoder_rejected(tmp_path):
    with pytest.raises(ValueError):greedy_associate([],[],use_gt_filter=True)
    p=tmp_path/'legacy.pkl';p.write_bytes(pickle.dumps({'nodes':[]}))
    with pytest.raises(ValueError,match='legacy cache'):run_reassociation(p,tmp_path/'out')


def test_mot_is_one_based_preserves_unknown_and_quarantine(tmp_path):
    ns=[node(129,[0,1]),node(500,[0,1])];ns[1]['quarantine']=True
    audit={'rows':ns[0]['rows']+ns[1]['rows']}
    rows=write_result(audit,{129:5},tmp_path)
    assert len(rows)==4 and {r[1] for r in rows}=={5,500}
    text=(tmp_path/'tracking_reid_trajectory.txt').read_text()
    assert text.startswith('1,5,')
    load_tracking_data(tmp_path/'tracking_reid_trajectory.txt')
    with pytest.raises(ValueError):write_result(audit,{129:500},tmp_path)


def test_split_applied_before_graph_features():
    n=node(1,[0,1,1050,1500]);a=dict(nodes=[n],rows=n['rows'],selected=[(0,4)])
    cut=restrict_audit(a,[[0,1049]])
    assert len(a['rows'])==4 and len(cut['rows'])==2
    assert set(cut['nodes'][0]['positions'])=={0,1}
    assert [s[0] for s in cut['nodes'][0]['samples']]==[0,1] and cut['selected']==[]


def test_overlap_filter_is_per_person():
    a=(0,1,0,0,20,40,.9);b=(0,2,0,0,20,40,.9)
    assert overlap(a,[a,b])==1 and overlap(a,[a])==0


def test_pseudo_requires_anchor_teacher_and_no_test_frame():
    seeds=[dict(frame=0,source_local_id=1,algorithm_id=129,pid=0)]
    r=dict(frame=20,source_local_id=1,algorithm_id=5)
    settings=dict(min_similarity=.9,teacher_margin=.1,max_pseudo_per_seed=1)
    proto=np.eye(2);feat=[np.array([1.,0.])]
    assert len(select_pseudo([r],feat,{129:5,5:5},seeds,proto,[[0,1049]],settings))==1
    assert select_pseudo([r],feat,{129:129,5:5},seeds,proto,[[0,1049]],settings)==[]
    assert select_pseudo([r],[np.array([0.,1.])],{129:5,5:5},seeds,proto,[[0,1049]],settings)==[]
    with pytest.raises(ValueError):select_pseudo([dict(r,frame=1200)],feat,{129:5,5:5},seeds,proto,[[0,1049]],settings)


def test_balanced_batches_keep_original_supervision():
    rows=[dict(pid=p,frame=f*60,origin=origin) for p in range(9) for origin in ['seed','pseudo'] for f in range(5)]
    for batch in SeedBalancedBatches(rows,4,3,42):
        assert len(batch)==len(set(batch))==36
        for p in range(9):
            items=[rows[i] for i in batch if rows[i]['pid']==p]
            assert len(items)==4 and sum(r['origin']=='seed' for r in items)>=2


def test_promotion_requires_quality_gain_without_safety_regression():
    b=dict(cross_algorithm_retrieval=dict(valid_queries=10,covered_identities=9,mAP_macro=.8,rank1_macro=.8),same_kit_FAR=0,TAR=.8,track_rank1_macro=.8)
    c=json.loads(json.dumps(b));c['cross_algorithm_retrieval']['mAP_macro']=.81
    assert promotion_allowed(b,c)
    c['same_kit_FAR']=.01
    assert not promotion_allowed(b,c)
    assert not promotion_allowed(b,b)


def test_training_windows_cannot_include_validation(tmp_path):
    (tmp_path/'dataset_report.json').write_text(json.dumps({'windows':{'train':[0,1049]}}))
    validate_windows([[0,1049]],tmp_path/'manifest.csv')
    with pytest.raises(ValueError):validate_windows([[0,1799]],tmp_path/'manifest.csv')


def test_dataset_namespace_mismatch_fails_before_training(tmp_path):
    audit=tmp_path/'audit.pkl';audit.write_bytes(b'changed')
    manifest=tmp_path/'manifest.csv';manifest.write_text('changed')
    (tmp_path/'dataset_report.json').write_text(json.dumps({'audit_sha256':'old','manifest_sha256':'old'}))
    with pytest.raises(ValueError,match='different audit'):
        validate_dataset({'audit_data':audit,'manifest':manifest},[])


def test_render_subsampling_keeps_duration(tmp_path):
    import cv2
    video=tmp_path/'source.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(96,64))
    assert writer.isOpened()
    for _ in range(30):writer.write(np.zeros((64,96,3),dtype=np.uint8))
    writer.release();mot=tmp_path/'mot.txt';mot.write_text('1,129,10,10,10,20,.9,-1,-1,-1\n')
    report=render_video_func(video,mot,tmp_path/'render.mp4',sample_rate=3,num_samples=0)
    assert report['rendered_frames']==10 and report['output_fps']==10


def test_second_stage_recovers_long_gap_without_truth():
    ns=[node(129,[0,1]),node(5,[400,401])]
    assert build_trajectory_graph(ns,{})==[]
    mapping,accepted,report=recover(ns)
    assert mapping[129]==mapping[5] and len(accepted)==1
    assert not report['labels_used_for_association']


def test_second_stage_interleaved_envelopes_not_same_frame():
    a=node(1,[0,1,8,9]);b=node(2,[4,5])
    mapping,accepted,_=recover([a,b])
    assert mapping[1]==mapping[2] and accepted[0]['interleaved']
    b=node(2,[1,4]);mapping,accepted,_=recover([a,b])
    assert mapping[1]!=mapping[2] and not accepted


def test_second_stage_rejects_ambiguous_mutually_exclusive_people():
    ns=[node(1,[0,1]),node(2,[4,5]),node(3,[4,5])]
    mapping,accepted,_=recover(ns)
    assert len(set(mapping.values()))==3 and not accepted


def test_second_stage_never_bridges_incompatible_components():
    a=node(1,[0,1]);b=node(2,[4,5]);c=node(3,[8,9],kit=1)
    mapping,accepted,_=recover([a,b,c])
    assert mapping[1]==mapping[2] and mapping[3]!=mapping[1]
    a['samples']=[]
    assert recover([a,b])[1]==[]


def test_second_stage_does_not_force_nine_clusters():
    ns=[node(i,[0,1]) for i in range(12)]
    assert len(set(recover(ns)[0].values()))==12


def test_short_inference_sampling_keeps_training_guards():
    n=node(1,list(range(5)));by={r[0]:[r] for r in n['rows']}
    assert select_observations(n,by,{})==[]
    assert len(select_observations(n,by,dict(endpoint_guard_frames=0,short_track_frames=45,short_track_samples=4)))==4
    for f in by:by[f].append((f,2,10.,10.,30.,60.,.9))
    assert select_observations(n,by,dict(endpoint_guard_frames=0,short_track_frames=45,short_track_samples=4))==[]


def test_local_appearance_recovers_endpoint_pose():
    a=node(1,[0,1]);b=node(2,[10,11,200,201])
    b['samples']=[(f,np.array([1.,0.]) if f<100 else np.array([0.,1.]),np.array([1.,0.])) for f in [10,11,200,201]]
    assert recover([a,b])[1]==[]
    assert len(recover([a,b],{'local_matching':True})[1])==1


def test_sparse_bridge_requires_two_sides_and_unique_anchor():
    sparse=node(1,[3,4]);sparse['samples']=[]
    anchor=node(2,[0,1,6,7])
    mapping,edges,_=recover([sparse,anchor],{'sparse_bridge':True})
    assert mapping[1]==mapping[2] and edges[0]['mode']=='sparse_bridge' and edges[0]['requires_review']
    assert recover([sparse,node(2,[0,1])],{'sparse_bridge':True})[1]==[]
    assert recover([sparse,anchor,node(3,[0,1,6,7])],{'sparse_bridge':True})[1]==[]


def test_gap_scaled_floor_is_stricter_nearby_and_looser_far():
    a=node(1,[0,1]);near=node(2,[20,21]);far=node(3,[400,401])
    vec=np.array([.70,.71414284]);vec=vec/np.linalg.norm(vec)
    a['samples']=[(f,np.array([1.,0.]),np.array([1.,0.])) for f in [0,1]]
    near['samples']=[(f,vec,np.array([1.,0.])) for f in [20,21]]
    far['samples']=[(f,vec,np.array([1.,0.])) for f in [400,401]]
    policy=dict(appearance_floor=.65,short_appearance_floor=.65,long_appearance_floor=.7,
                unregistered_long_floor=.8,unmatched_to_registered_long_floor=.8,
                gap_scaled_floor=True,appearance_floor_near=.80,appearance_floor_far=.60,
                appearance_floor_span=180,unmatched_premium_near=0,unmatched_premium_far=0,
                min_centroid=.55,component_min_affinity=.62,component_min_centroid=.55,
                mutual_unique=True,gallery_rewrite_affinity=False)
    assert recover([a,near],policy)[1]==[]
    mapping,accepted,_=recover([a,far],policy)
    assert mapping[1]==mapping[3] and accepted


def test_unregistered_long_gap_rejects_mid_score_pairs():
    a=node(1,[0,1]);b=node(2,[400,401])
    b['samples']=[(f,np.array([.79,.6131]),np.array([1.,0.])) for f in [400,401]]
    policy=dict(appearance_floor=.65,short_appearance_floor=.65,long_appearance_floor=.7,
                unregistered_long_floor=.8,unmatched_to_registered_long_floor=.8,
                min_centroid=.55,component_min_affinity=.62,component_min_centroid=.55,
                mutual_unique=True,gallery_rewrite_affinity=False,gallery_veto=True)
    assert recover([a,b],policy)[1]==[]


def test_mutual_unique_does_not_attach_a_third_track_to_a_stronger_pair():
    a=node(1,[0,1]);b=node(2,[10,11]);c=node(3,[20,21])
    a['samples']=[(f,np.array([1.,0.]),np.array([1.,0.])) for f in [0,1]]
    b['samples']=[(f,np.array([1.,0.]),np.array([1.,0.])) for f in [10,11]]
    c['samples']=[(f,np.array([.75,.6614]),np.array([1.,0.])) for f in [20,21]]
    policy=dict(appearance_floor=.65,min_centroid=.55,component_min_affinity=.62,component_min_centroid=.55,
                ambiguity_margin=.05,mutual_unique=True,gallery_rewrite_affinity=False,absorb_affinity=.80)
    mapping,accepted,_=recover([a,b,c],policy)
    assert mapping[1]==mapping[2] and mapping[3]!=mapping[1]


def test_adjacent_nonoverlap_is_not_a_continuation_but_overlap_still_merges():
    a=node(7,list(range(10)))
    a['positions']={f:np.array([0.,0.]) for f in a['positions']}
    b=node(31,list(range(16,20)))
    b['positions']={f:np.array([1.,0.]) for f in b['positions']}
    b['rows']=[(f,31,200.,10.,30.,60.,.9) for f in range(16,20)]
    mapping,accepted,_=recover([a,b],dict(mixed_box_exclude=True,min_affinity=.5,min_centroid=.5))
    assert mapping[7]!=mapping[31] and not accepted
    c=node(31,list(range(16,20)))
    c['positions']={f:np.array([1.,0.]) for f in c['positions']}
    mapping,accepted,_=recover([a,c],dict(mixed_box_exclude=True,min_affinity=.5,min_centroid=.5))
    assert mapping[7]==mapping[31] and accepted


def test_occupancy_vetoes_short_path_and_never_merges_quarantine():
    a=node(1,[0,1]);b=node(2,[20,21])
    a['positions']={0:np.array([0.,0.]),1:np.array([0.,0.])}
    b['positions']={20:np.array([3.,0.]),21:np.array([3.,0.])}
    q=node(90,[10]);q['quarantine']=True;q['positions']={10:np.array([1.42,0.])}
    mapping,accepted,_=recover([a,b,q],dict(occupancy_veto=True,occupancy_max_gap=90,occupancy_residual_m=1.2))
    assert not accepted and mapping[1]!=mapping[2] and 90 not in mapping
    mapping,accepted,_=recover([a,b,q])
    assert accepted and mapping[1]==mapping[2] and 90 not in mapping


def test_training_references_use_appearance_not_query_ids():
    angle=np.deg2rad(25)
    a=node(666,[0,1]);b=node(777,[400,401])
    for n,sign in [(a,1),(b,-1)]:
        n['samples']=[(f,np.array([np.cos(angle),sign*np.sin(angle)]),np.array([1.,0.])) for f in [n['start'],n['end']]]
    assert recover([a,b])[1]==[]
    mapping,edges,report=recover([a,b],references=np.eye(2))
    assert mapping[666]==mapping[777] and report['training_gallery_used']
    assert report['labels_used_for_association'] and not report['query_identity_map_used']


def test_gta_connect_uses_mean_distance_and_keeps_adjacent_rival():
    a=node(150,[0,1]);b=node(295,[400,401])
    vec=np.array([.68,.73321211]);vec=vec/np.linalg.norm(vec)
    a['samples']=[(f,np.array([1.,0.]),np.array([1.,0.])) for f in [0,1]]
    b['samples']=[(f,vec,np.array([1.,0.])) for f in [400,401]]
    policy=dict(appearance_floor=.65,unregistered_long_floor=.8,unmatched_to_registered_long_floor=.8,
                min_centroid=.55,mutual_unique=True,gallery_rewrite_affinity=False)
    assert recover([a,b],policy)[1]==[]
    mapping,accepted,report=recover([a,b],dict(policy,gta_connect=True,gta_merge_dist=.4))
    assert mapping[150]==mapping[295] and accepted[0]['mode']=='gta_connect' and report['gta_connect']
    far=node(295,[400,401])
    far['samples']=[(f,np.array([0.,1.]),np.array([1.,0.])) for f in [400,401]]
    assert recover([a,far],dict(policy,gta_connect=True,gta_merge_dist=.4))[1]==[]
    r7=node(7,list(range(10)));r7['positions']={f:np.array([0.,0.]) for f in r7['positions']}
    r31=node(31,list(range(16,20)))
    r31['positions']={f:np.array([1.,0.]) for f in r31['positions']}
    r31['rows']=[(f,31,200.,10.,30.,60.,.9) for f in range(16,20)]
    mapping,accepted,_=recover([r7,r31],dict(gta_connect=True,mixed_box_exclude=True,min_affinity=.5,min_centroid=.5))
    assert mapping[7]!=mapping[31] and not accepted


def test_rule_geometry_unique_short_gap_ignores_appearance_and_skips_rivals():
    a=node(129,[0,1]);b=node(244,[16,17])
    a['positions']={0:np.array([0.,0.]),1:np.array([0.,0.])}
    b['positions']={16:np.array([1.5,0.]),17:np.array([1.5,0.])}
    b['samples']=[]
    policy=dict(appearance_floor=.65,unmatched_to_registered_long_floor=.8,mutual_unique=True,
                mixed_box_exclude=True,rule_geometry=True,rule_max_gap=60,rule_max_m=3.5)
    mapping,accepted,_=recover([a,b],policy)
    assert mapping[129]==mapping[244] and accepted[0]['mode']=='rule_geometry'
    rival=node(31,list(range(16,20)))
    rival['positions']={f:np.array([1.,0.]) for f in rival['positions']}
    rival['rows']=[(f,31,200.,10.,30.,60.,.9) for f in range(16,20)]
    host=node(7,list(range(10)))
    host['positions']={f:np.array([0.,0.]) for f in host['positions']}
    mapping,accepted,_=recover([host,rival],dict(policy,min_affinity=.5,min_centroid=.5))
    assert mapping[7]!=mapping[31]


def test_quarantine_supports_path_but_never_becomes_identity():
    a=node(1,[0,1]);b=node(2,[80,81])
    a['positions']={0:np.array([0.,0.]),1:np.array([0.,0.])}
    b['positions']={80:np.array([3.,0.]),81:np.array([3.,0.])}
    b['samples']=[(f,np.array([0.,1.]),np.array([1.,0.])) for f in [80,81]]
    q=node(90,[40]);q['quarantine']=True;q['positions']={40:np.array([1.5,0.])}
    policy=dict(appearance_floor=.65,unregistered_long_floor=.8,rule_geometry=True,
                quarantine_bridge=True,quarantine_max_gap=120,occupancy_residual_m=1.2)
    mapping,accepted,_=recover([a,b,q],policy)
    assert mapping[1]==mapping[2] and 90 not in mapping
    assert accepted[0]['mode']=='quarantine_bridge' and 90 in accepted[0]['quarantine_support']
    live=node(3,list(range(0,82)))
    live['positions']={f:np.array([1.5,0.]) for f in live['positions']}
    mapping,accepted,_=recover([a,b,q,live],policy)
    assert mapping[1]!=mapping[2]
    far=node(4,[80,81]);far['positions']={80:np.array([20.,0.]),81:np.array([20.,0.])}
    far['samples']=[(f,np.array([0.,1.]),np.array([1.,0.])) for f in [80,81]]
    mapping,accepted,_=recover([a,far,q],policy)
    assert mapping[1]!=mapping[4]


def test_third_stage_keeps_seed_and_only_adds_rule_merges():
    a=node(1,[0,1]);b=node(2,[10,11]);c=node(3,[26,27])
    a['positions']={0:np.array([0.,0.]),1:np.array([0.,0.])}
    b['positions']={10:np.array([0.2,0.]),11:np.array([0.2,0.])}
    c['positions']={26:np.array([1.6,0.]),27:np.array([1.6,0.])}
    c['samples']=[]
    policy=dict(appearance_floor=.65,unregistered_long_floor=.8,mutual_unique=True,
                mixed_box_exclude=True,rule_only=True,rule_geometry=True,rule_max_gap=60,rule_max_m=3.5,
                seed_mapping={1:1,2:1,3:3})
    mapping,accepted,report=recover([a,b,c],policy)
    assert mapping[1]==mapping[2] and mapping[3]==mapping[1] and report['rule_only'] and report['seed_used']
    assert all(e['mode'] in ('rule_geometry','quarantine_bridge','sparse_bridge') for e in accepted)
    assert recover([a,b,c],dict(policy,seed_mapping={1:1,2:1,3:3},rule_geometry=False))[0][3]==3


def test_gta_additive_keeps_a3_merges_and_adds_gta_only_pairs():
    a=node(1,[0,1]);b=node(2,[10,11]);c=node(3,[400,401])
    vec=np.array([.68,.73321211]);vec=vec/np.linalg.norm(vec)
    a['samples']=[(f,np.array([1.,0.]),np.array([1.,0.])) for f in [0,1]]
    b['samples']=[(f,np.array([1.,0.]),np.array([1.,0.])) for f in [10,11]]
    c['samples']=[(f,vec,np.array([1.,0.])) for f in [400,401]]
    policy=dict(appearance_floor=.65,unregistered_long_floor=.8,unmatched_to_registered_long_floor=.8,
                min_centroid=.55,component_min_affinity=.62,component_min_centroid=.55,
                mutual_unique=True,gallery_rewrite_affinity=False)
    mapping,accepted,_=recover([a,b,c],policy)
    assert mapping[1]==mapping[2] and mapping[3]!=mapping[1]
    mapping,accepted,report=recover([a,b,c],dict(policy,gta_additive=True,gta_merge_dist=.4))
    assert mapping[1]==mapping[2]==mapping[3] and report['gta_additive']
    assert any(e['mode']!='gta_connect' for e in accepted) and any(e['mode']=='gta_connect' for e in accepted)


def test_jersey_agree_lowers_floor_but_never_vetoes():
    from jersey_vote import vote_track, parse_number
    assert parse_number('17')==17 and parse_number('07')==7 and parse_number('0') is None
    n,info=vote_track([dict(number=7,conf=.9),dict(number=7,conf=.91),dict(number=1,conf=.86)])
    assert n==7 and info['reason']=='ok'
    assert vote_track([dict(number=7,conf=.9)])[0] is None
    a=node(150,[0,1]);b=node(295,[400,401])
    vec=np.array([.68,.73321211]);vec=vec/np.linalg.norm(vec)
    a['samples']=[(f,np.array([1.,0.]),np.array([1.,0.])) for f in [0,1]]
    b['samples']=[(f,vec,np.array([1.,0.])) for f in [400,401]]
    policy=dict(appearance_floor=.65,short_appearance_floor=.65,long_appearance_floor=.7,
                unregistered_long_floor=.8,unmatched_to_registered_long_floor=.8,
                min_centroid=.55,component_min_affinity=.62,component_min_centroid=.55,
                gallery_floor=.65,gallery_rewrite_affinity=False,mutual_unique=True)
    assert recover([a,b],policy)[1]==[]
    mapping,accepted,_=recover([a,b],dict(policy,jersey_agree=True,jersey_votes={150:25,295:25}))
    assert mapping[150]==mapping[295] and accepted[0]['jersey_agree']
    mapping,accepted,_=recover([a,b],dict(policy,jersey_agree=True,jersey_votes={150:25,295:3}))
    assert mapping[150]!=mapping[295] and not accepted
    strong=node(295,[400,401])
    strong['samples']=[(f,np.array([1.,0.]),np.array([1.,0.])) for f in [400,401]]
    mapping,accepted,_=recover([a,strong],dict(policy,jersey_agree=True,jersey_votes={150:25,295:3}))
    assert mapping[150]==mapping[295] and accepted


def test_requested_gallery_cannot_silently_be_skipped(tmp_path):
    p=tmp_path/'audit.pkl';p.write_bytes(pickle.dumps(dict(nodes=[],rows=[],feature_metadata={'dimension':3840})))
    with pytest.raises(ValueError,match='no checkpoint-bound reference bank'):
        run_second_stage(p,tmp_path/'out',dict(use_training_references=True))


@pytest.mark.parametrize('entry',['run_pipeline.py','scripts/train_selftrain.py'])
def test_entrypoints_work_from_any_directory(entry,tmp_path):
    result=subprocess.run([sys.executable,'-B',str(ROOT/entry),'--help'],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
