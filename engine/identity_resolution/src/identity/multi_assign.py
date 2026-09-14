"""Same-kit multi-death assignment by combined score, plus hole fill."""
from __future__ import annotations

from collections import defaultdict

from identity.common import (
    box_on_image_edge, cluster_reachability, nodes_by_output, remap_rows,
    spans_from_rows, union_prefer_selected,
)

NM_GAP = 60
MIN_TRACK = 8
SCORE_MARGIN = 0.04
SCORE_MIN = 0.63
HOLE_MAX_FRAMES = 20


def pick_best(edges, side, margin=SCORE_MARGIN):
    ranked = sorted(edges, key=lambda edge: (-edge['score'], edge.get('residual_m') or 9, edge['from_'], edge['to']))
    if not ranked:
        return None
    if len(ranked) == 1:
        return ranked[0]
    first, second = ranked[0], ranked[1]
    if first['score'] - second['score'] >= margin:
        return first
    # Length is sample quantity, not evidence that two fragments are one person.
    return None


def assign_mutual_score(items, margin=SCORE_MARGIN):
    passing = [item for item in items if item.get('ok') and item.get('score') is not None and item['score'] >= SCORE_MIN]
    outgoing, incoming = defaultdict(list), defaultdict(list)
    for edge in passing:
        outgoing[edge['from_']].append(edge)
        incoming[edge['to']].append(edge)
    selected = []
    for edge in passing:
        best_out = pick_best(outgoing[edge['from_']], 'out', margin)
        best_in = pick_best(incoming[edge['to']], 'in', margin)
        if edge is not best_out or edge is not best_in:
            edge['accepted'] = False
            edge['reject'] = 'not_mutual_best_score'
            continue
        rivals_out = [item for item in outgoing[edge['from_']] if item is not edge]
        rivals_in = [item for item in incoming[edge['to']] if item is not edge]
        edge['out_margin'] = edge['score'] - max((r['score'] for r in rivals_out), default=0)
        edge['in_margin'] = edge['score'] - max((r['score'] for r in rivals_in), default=0)
        edge['accepted'] = True
        selected.append(edge)
    return selected


def nm_candidates(spans, grouped, max_gap=NM_GAP, min_track=MIN_TRACK):
    identities = [i for i in spans if i in grouped and spans[i]['n'] >= min_track]
    found = []
    for left_id in identities:
        for right_id in identities:
            if left_id == right_id:
                continue
            gap = spans[right_id]['first'] - spans[left_id]['last']
            if not 1 <= gap <= max_gap:
                continue
            if spans[left_id]['frames'] & spans[right_id]['frames']:
                continue
            left_node, right_node = grouped[left_id][-1], grouped[right_id][0]
            if left_node['kit'] < 0 or right_node['kit'] < 0 or left_node['kit'] != right_node['kit']:
                continue
            if box_on_image_edge(left_node['rows'][-1]) or box_on_image_edge(right_node['rows'][0]):
                continue
            gate = cluster_reachability(left_node, right_node)
            item = dict(from_=left_id, to=right_id, kit=left_node['kit'],
                        left_frames=spans[left_id]['n'], right_frames=spans[right_id]['n'],
                        left_last=spans[left_id]['last'], right_first=spans[right_id]['first'],
                        left_node=left_node['id'], right_node=right_node['id'],
                        left_kit=left_node['kit'], right_kit=right_node['kit'], **gate)
            item['gap'] = gap
            if item.get('ok') and item.get('score') is not None:
                item['reason'] = 'cluster_nm_pass'
            found.append(item)
    return found


def merge_cluster_nm(rows, nodes, skip, uf, selected=()):
    accepted, inspected, seen = [], [], set()
    for round_id in range(20):
        current_rows = remap_rows(rows, uf)
        current_nodes = [dict(node, output_id=uf.find(node['output_id']) if not node['quarantine'] else node['output_id'])
                         for node in nodes]
        current_skip = {node['output_id'] for node in current_nodes if node['quarantine']} | skip
        spans = spans_from_rows(current_rows, current_skip)
        grouped = nodes_by_output(current_nodes, current_skip)
        found = nm_candidates(spans, grouped)
        for item in found:
            key = (item['from_'], item['to'], item.get('left_last'), item.get('right_first'))
            if key in seen:
                continue
            seen.add(key)
            inspected.append(item)
        chosen = assign_mutual_score(found)
        if not chosen:
            break
        progressed = False
        for edge in chosen:
            if union_prefer_selected(uf, edge['from_'], edge['to'], selected):
                edge['round'] = round_id + 1
                accepted.append(edge)
                progressed = True
        if not progressed:
            break
    return uf, accepted, inspected


def hole_fill_candidates(spans, grouped, hole_max=HOLE_MAX_FRAMES):
    found = []
    longs = [i for i, span in spans.items() if span['n'] >= 30 and i in grouped]
    shorts = [i for i, span in spans.items() if 2 <= span['n'] <= hole_max and i in grouped]
    for long_id in longs:
        long_frames = spans[long_id]['frames']
        ordered = sorted(long_frames)
        holes = []
        for prev, nxt in zip(ordered, ordered[1:]):
            if nxt - prev > 2:
                holes.append((prev, nxt))
        if not holes:
            continue
        long_kit = grouped[long_id][-1]['kit']
        if long_kit < 0:
            continue
        for short_id in shorts:
            if grouped[short_id][0]['kit'] != long_kit:
                continue
            if spans[short_id]['frames'] & long_frames:
                continue
            sf, sl = spans[short_id]['first'], spans[short_id]['last']
            hole = next((h for h in holes if h[0] < sf and sl < h[1]), None)
            if hole is None:
                continue
            left_node = next(n for n in reversed(grouped[long_id]) if n['end'] == hole[0] or hole[0] in {r[0] for r in n['rows']})
            right_node = next(n for n in grouped[long_id] if n['start'] == hole[1] or hole[1] in {r[0] for r in n['rows']})
            short_left, short_right = grouped[short_id][0], grouped[short_id][-1]
            # A hole may be inside one node: use the observed hole boundaries,
            # not that node's overall start/end (which can give negative gaps).
            def portion(node, predicate):
                rows = [r for r in node['rows'] if predicate(r[0])]
                frames = {r[0] for r in rows}
                return dict(node, rows=rows, start=rows[0][0], end=rows[-1][0],
                            positions={f:p for f,p in node['positions'].items() if f in frames},
                            samples=[s for s in node['samples'] if s[0] in frames])
            into = cluster_reachability(portion(left_node, lambda f:f <= hole[0]), short_left)
            outof = cluster_reachability(short_right, portion(right_node, lambda f:f >= hole[1]))
            ok = bool(into.get('ok') and outof.get('ok'))
            item = dict(from_=long_id, to=short_id, kit=long_kit, gap=sf - hole[0],
                        left_frames=spans[long_id]['n'], right_frames=spans[short_id]['n'],
                        left_last=hole[0], right_first=sf, left_node=left_node['id'],
                        right_node=short_left['id'], left_kit=long_kit, right_kit=short_left['kit'],
                        ok=ok, reason='hole_fill_pass' if ok else f'into:{into.get("reason")}/out:{outof.get("reason")}',
                        distance_m=into.get('distance_m'), residual_m=into.get('residual_m'),
                        appearance=into.get('appearance'), colour=into.get('colour'),
                        score=None if not ok else ((into.get('score') or 0) + (outof.get('score') or 0)) / 2)
            found.append(item)
    return found


def merge_hole_fills(rows, nodes, skip, uf, selected=()):
    accepted, inspected = [], []
    current_rows = remap_rows(rows, uf)
    current_nodes = [dict(node, output_id=uf.find(node['output_id']) if not node['quarantine'] else node['output_id'])
                     for node in nodes]
    current_skip = {node['output_id'] for node in current_nodes if node['quarantine']} | skip
    spans = spans_from_rows(current_rows, current_skip)
    grouped = nodes_by_output(current_nodes, current_skip)
    found = hole_fill_candidates(spans, grouped)
    inspected.extend(found)
    # Reject competing holes/identities instead of merging every feasible edge.
    for edge in assign_mutual_score(found):
        if union_prefer_selected(uf, edge['from_'], edge['to'], selected):
            edge['accepted'] = True
            edge['round'] = 1
            accepted.append(edge)
    return uf, accepted, inspected
