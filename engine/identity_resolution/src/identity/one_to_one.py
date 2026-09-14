"""Unique 1-to-1 ID replacement inside a colour cluster."""
from __future__ import annotations

from collections import defaultdict

from identity.common import (
    MAX_GAP, UnionFind, box_on_image_edge, cluster_reachability, kit_at,
    live_by_frame, nodes_by_output, remap_rows, spans_from_rows, union_prefer_selected,
)


def cluster_swap_candidates(spans, live, grouped, max_gap=MAX_GAP):
    identities = sorted(spans)
    candidates = []
    for left_id in identities:
        for right_id in identities:
            if left_id == right_id:
                continue
            gap = spans[right_id]['first'] - spans[left_id]['last']
            if not 1 <= gap <= max_gap:
                continue
            if spans[left_id]['frames'] & spans[right_id]['frames']:
                continue
            if left_id not in grouped or right_id not in grouped:
                continue
            left_node, right_node = grouped[left_id][-1], grouped[right_id][0]
            kit = left_node['kit']
            if kit < 0 or right_node['kit'] < 0 or right_node['kit'] != kit:
                continue
            left_last, right_first = spans[left_id]['last'], spans[right_id]['first']
            before = {i for i in live[left_last] if kit_at(grouped, i, left_last) == kit}
            after = {i for i in live[right_first] if kit_at(grouped, i, right_first) == kit}
            if before - {left_id} != after - {right_id}:
                continue
            if before - after != {left_id} or after - before != {right_id}:
                continue
            deaths, births = set(), set()
            for identity, span in spans.items():
                if left_last <= span['last'] < right_first and kit_at(grouped, identity, span['last']) == kit:
                    deaths.add(identity)
                if left_last < span['first'] <= right_first and kit_at(grouped, identity, span['first']) == kit:
                    births.add(identity)
            if deaths != {left_id} or births != {right_id}:
                continue
            if box_on_image_edge(left_node['rows'][-1]) or box_on_image_edge(right_node['rows'][0]):
                continue
            gate = cluster_reachability(left_node, right_node)
            item = dict(from_=left_id, to=right_id, kit=kit,
                        companions=sorted(before - {left_id}), companion_count=len(before) - 1,
                        left_frames=spans[left_id]['n'], right_frames=spans[right_id]['n'],
                        left_last=left_last, right_first=right_first,
                        left_node=left_node['id'], right_node=right_node['id'],
                        left_kit=left_node['kit'], right_kit=right_node['kit'], **gate)
            item['gap'] = gap
            candidates.append(item)
    return candidates


def select_unique_pairs(candidates):
    passing = [item for item in candidates if item.get('ok')]
    outgoing, incoming = defaultdict(list), defaultdict(list)
    for edge in passing:
        outgoing[edge['from_']].append(edge)
        incoming[edge['to']].append(edge)
    selected = []
    for edge in passing:
        if len(outgoing[edge['from_']]) != 1 or len(incoming[edge['to']]) != 1:
            edge['accepted'] = False
            continue
        edge['accepted'] = True
        selected.append(edge)
    return selected


def merge_cluster_swaps(rows, nodes, skip, selected=()):
    uf = UnionFind()
    accepted, inspected, seen = [], [], set()
    for round_id in range(20):
        current_rows = remap_rows(rows, uf)
        current_nodes = []
        for node in nodes:
            item = dict(node)
            if not item['quarantine']:
                item['output_id'] = uf.find(item['output_id'])
            current_nodes.append(item)
        current_skip = {node['output_id'] for node in current_nodes if node['quarantine']} | skip
        spans = spans_from_rows(current_rows, current_skip)
        live = live_by_frame(current_rows, current_skip)
        grouped = nodes_by_output(current_nodes, current_skip)
        found = cluster_swap_candidates(spans, live, grouped)
        for item in found:
            key = (item['from_'], item['to'], item['left_last'], item['right_first'])
            if key in seen:
                continue
            seen.add(key)
            inspected.append(item)
        chosen = select_unique_pairs(found)
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
