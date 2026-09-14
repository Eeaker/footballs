import numpy as np
from identity.common import UnionFind
from identity.one_to_one import merge_cluster_swaps
from identity.multi_assign import merge_hole_fills


def node(i, frames, q=False):
    return dict(id=i, output_id=i, source_local=1, quarantine=q, kit=0,
                start=min(frames), end=max(frames),
                rows=[(f,i,500,400,40,80,.9) for f in frames],
                positions={f:np.array([20.,10.]) for f in frames},
                samples=[(f,np.array([1.,0.]),np.array([1.,0.])) for f in frames])


def test_quarantine_is_not_a_birth_death_companion_or_candidate():
    clean=[node(1,[0,1]),node(2,[4,5])]
    dirty=[node(90,[1,2],True),node(91,[3,4],True)]
    def run(ns):
        _,edges,_=merge_cluster_swaps([r for n in ns for r in n['rows']],ns,set())
        return [(e['from_'],e['to'],e['companions']) for e in edges]
    assert run(clean)==run(clean+dirty)==[(1,2,[])]


def test_hole_inside_a_node_uses_actual_boundaries():
    ns=[node(1,list(range(20))+list(range(25,45))),node(2,[21,22,23])]
    _,edges,_=merge_hole_fills([r for n in ns for r in n['rows']],ns,set(),UnionFind())
    assert len(edges)==1 and edges[0]['to']==2


def test_competing_hole_fill_is_not_transitively_merged():
    ns=[node(1,list(range(20))+list(range(25,45))),
        node(2,[21,22,23]),node(3,[21,22,23])]
    _,edges,_=merge_hole_fills([r for n in ns for r in n['rows']],ns,set(),UnionFind())
    assert edges==[]
