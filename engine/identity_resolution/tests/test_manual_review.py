import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.apply_manual_review import apply_review
import pytest


def test_invalid_mixed_box_cannot_keep_motion_or_appearance():
    audit={'nodes':[dict(output_id=114,quarantine=False,positions={1:[2,3]},samples=[1])],'rows':[]}
    result,mapping=apply_review(audit,{114:114},{'confirmed_pairs':[],
                            'invalid_detections':[dict(source_id=114,reason='mixed_people')]})
    assert result['nodes'][0]['quarantine'] and not result['nodes'][0]['positions'] and not result['nodes'][0]['samples']
    assert mapping=={} and not audit['nodes'][0]['quarantine']
    from features import select_observations
    assert select_observations(result['nodes'][0],{}, {})==[]


def test_invalid_id_already_merged_fails_closed():
    audit={'nodes':[dict(output_id=114,quarantine=False),dict(output_id=5,quarantine=False)]}
    with pytest.raises(ValueError,match='already merged'):
        apply_review(audit,{114:5,5:5},{'confirmed_pairs':[],
                     'invalid_detections':[dict(source_id=114,reason='mixed_people')]})


def test_degradation_preserves_shape_and_zero_probability_is_identity():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reid_model'))
    from augmentations import RandomNativeDegradation
    from PIL import Image
    import numpy as np
    im=Image.fromarray(np.random.default_rng(0).integers(0,255,(80,40,3),dtype=np.uint8))
    assert np.array_equal(np.asarray(im),np.asarray(RandomNativeDegradation(0,0)(im)))
    for transform in [RandomNativeDegradation(1,0),RandomNativeDegradation(0,1)]:
        output=transform(im)
        assert output.size==im.size and output.mode=='RGB'


def test_fragment_views_keep_original_and_do_not_read_identity_groups():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reid_model'))
    import augmentations
    from augmentations import apply_fragment_view,expand_train_views
    import inspect
    import numpy as np
    from PIL import Image
    source=inspect.getsource(augmentations)
    assert 'identity_groups' not in source
    rng=np.random.default_rng(0)
    im=Image.fromarray(rng.integers(0,255,(90,40,3),dtype=np.uint8))
    class R:
        def __init__(self,seed):self._r=__import__('random').Random(seed)
        def random(self):return self._r.random()
        def uniform(self,a,b):return self._r.uniform(a,b)
        def randint(self,a,b):return self._r.randint(a,b)
        def choice(self,seq):return self._r.choice(seq)
    assert np.array_equal(np.asarray(im),np.asarray(apply_fragment_view(im,'original',R(1))))
    other=Image.fromarray(rng.integers(0,255,(90,40,3),dtype=np.uint8))
    light=apply_fragment_view(im,'light',R(2),other=other);heavy=apply_fragment_view(im,'heavy',R(3),other=other)
    assert light.size==im.size==heavy.size and light.mode==heavy.mode=='RGB'
    assert not np.array_equal(np.asarray(im),np.asarray(light))
    assert not np.array_equal(np.asarray(im),np.asarray(heavy))
    rows=[dict(split='train',pid=0,label_source='algorithm_seed_only',algorithm_id=31),
          dict(split='val',pid=0,label_source='held_out_fragment',algorithm_id=5)]
    expanded=expand_train_views(rows,dict(fragment_appearance=dict(views=['original','light','heavy'])))
    assert len(expanded)==1 and expanded[0]['algorithm_id']==31
    with pytest.raises(ValueError,match='Human correspondence'):
        expand_train_views([dict(split='train',label_source='human_group',algorithm_id=5)],
                           dict(fragment_appearance=dict(views=['original','heavy'])))


def test_view_schedule_keeps_original_majority():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reid_model'))
    from augmentations import view_schedule
    from train_adapt import ViewBalancedSampler
    assert view_schedule(4).count('original')==2 and 'light' in view_schedule(4) and 'heavy' in view_schedule(4)
    assert view_schedule(2,0)==['original','light'] and view_schedule(2,1)==['original','heavy']
    rows=[]
    for pid in range(3):
        for t in range(8):
            rows.append(dict(pid=pid,frame=t*60,label_source='algorithm_seed_only'))
    for batch in ViewBalancedSampler(rows,4,3,0):
        assert len(batch)==12
        views=[v for _,v in batch]
        assert views.count('original')>=6
        bypid={}
        for (i,view) in batch:
            bypid.setdefault(rows[i]['pid'],[]).append(view)
        for pid,vs in bypid.items():
            assert vs.count('original')>=2
