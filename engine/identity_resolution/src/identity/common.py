"""Shared geometry, IO and identity bookkeeping."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict

import numpy as np

IMAGE = (1920, 1080)
EDGE_PX = 12
PITCH_EDGE_M = 2.0
MAX_GAP = 60


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left, right):
        root_l, root_r = self.find(left), self.find(right)
        if root_l == root_r:
            return False
        self.parent[root_r] = root_l
        return True


def union_prefer_selected(uf, left, right, selected=()):
    root_l, root_r = uf.find(left), uf.find(right)
    if root_l == root_r:
        return False
    selected = set(selected)
    if root_r in selected and root_l not in selected:
        uf.parent[root_l] = root_r
    else:
        uf.parent[root_r] = root_l
    return True


def unit(x):
    x = np.asarray(x, dtype=float)
    return x / max(np.linalg.norm(x), 1e-9)


def runs(frames):
    out = []
    for frame in sorted(set(frames)):
        if not out or frame != out[-1][-1] + 1:
            out.append([])
        out[-1].append(frame)
    return out


def statistics(rows):
    tracks = defaultdict(list)
    for row in rows:
        tracks[int(row[1])].append(int(row[0]))
    lengths = [len(v) for v in tracks.values()]
    continuous = [max(map(len, runs(v))) for v in tracks.values()]
    return dict(
        rows=len(rows), ids=len(tracks),
        median_observed_s=float(np.median(lengths)) / 30 if lengths else 0,
        maximum_observed_s=max(lengths, default=0) / 30,
        ids_at_least_5s=sum(n >= 150 for n in lengths),
        ids_at_least_10s=sum(n >= 300 for n in lengths),
        maximum_contiguous_s=max(continuous, default=0) / 30,
        top10_mean_observed_s=float(np.mean(sorted(lengths, reverse=True)[:10])) / 30 if lengths else 0,
        max_simultaneous=max(Counter(int(r[0]) for r in rows).values(), default=0),
    )


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')


def write_mot(path, rows):
    with path.open('w', encoding='utf8') as handle:
        for frame, identity, x, y, w, h, conf in sorted(rows):
            handle.write(f'{frame + 1},{identity},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{conf:.5f},-1,-1,-1\n')


def csv_write(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def endpoint_velocity(node, side):
    rows = node['rows'][-20:] if side == 'end' else node['rows'][:20]
    t = np.array([x[0] for x in rows], float)
    t -= t.mean()
    pts = np.array([node['positions'][x[0]] for x in rows])
    return (t[:, None] * pts).sum(0) / max(float(t @ t), 1e-9)


def live_by_frame(rows, skip):
    live = defaultdict(set)
    for frame, identity, *_ in rows:
        if identity not in skip:
            live[frame].add(identity)
    return live


def spans_from_rows(rows, skip):
    by_id = defaultdict(list)
    for frame, identity, *_ in rows:
        if identity not in skip:
            by_id[identity].append(frame)
    spans = {}
    for identity, frames in by_id.items():
        frames = sorted(frames)
        spans[identity] = dict(first=frames[0], last=frames[-1], n=len(frames), frames=set(frames))
    return spans


def nodes_by_output(nodes, skip):
    grouped = defaultdict(list)
    for node in nodes:
        if node['quarantine'] or node['output_id'] in skip:
            continue
        grouped[node['output_id']].append(node)
    for identity in grouped:
        grouped[identity].sort(key=lambda node: (node['start'], node['end']))
    return grouped


def remap_rows(rows, uf):
    return [(frame, uf.find(identity), *rest) for frame, identity, *rest in rows]


def remap_nodes(nodes, uf):
    for node in nodes:
        if not node['quarantine']:
            node['output_id'] = uf.find(node['output_id'])
    return nodes


def box_on_image_edge(row):
    x, y, w, h = (float(v) for v in row[2:6])
    return x <= EDGE_PX or y <= EDGE_PX or x + w >= IMAGE[0] - EDGE_PX


def near_pitch_edge(pos):
    x, y = float(pos[0]), float(pos[1])
    return x < PITCH_EDGE_M or x > 45 - PITCH_EDGE_M or y < PITCH_EDGE_M or y > 25 - PITCH_EDGE_M


def kit_at(grouped, identity, frame):
    for node in grouped.get(identity, []):
        if any(row[0] == frame for row in node['rows']):
            return node['kit']
    return -1


def reachability(left, right, max_gap=MAX_GAP):
    gap = right['start'] - left['end']
    report = dict(ok=False, reason='gap', gap=gap, distance_m=None, residual_m=None,
                  appearance=None, colour=None, score=None)
    if not 1 <= gap <= max_gap:
        return report
    if left['quarantine'] or right['quarantine']:
        report['reason'] = 'quarantine'
        return report
    if left['kit'] >= 0 and right['kit'] >= 0 and left['kit'] != right['kit']:
        report['reason'] = 'kit'
        return report
    distance = float(np.linalg.norm(right['positions'][right['start']] - left['positions'][left['end']]))
    report['distance_m'] = distance
    if distance > 1. + 9. * gap / 30:
        report['reason'] = 'distance'
        return report
    left_v, right_v = endpoint_velocity(left, 'end'), endpoint_velocity(right, 'start')
    residual = max(
        float(np.linalg.norm(left['positions'][left['end']] + left_v * gap - right['positions'][right['start']])),
        float(np.linalg.norm(right['positions'][right['start']] - right_v * gap - left['positions'][left['end']])),
    )
    report['residual_m'] = residual
    if residual > .7 + 3. * gap / 30:
        report['reason'] = 'residual'
        return report
    left_s, right_s = left['samples'][-3:], right['samples'][:3]
    if left_s and right_s:
        appearance = float(unit(np.mean([s[1] for s in left_s], 0)) @ unit(np.mean([s[1] for s in right_s], 0)))
        colour = float(unit(np.mean([s[2] for s in left_s], 0)) @ unit(np.mean([s[2] for s in right_s], 0)))
        report['appearance'] = appearance
        report['colour'] = colour
        if appearance < .80 or colour < .35:
            report['reason'] = 'appearance'
            return report
    else:
        if gap > 10 or left['source_local'] != right['source_local'] or distance > .8:
            report['reason'] = 'no_evidence'
            return report
        appearance = colour = None
    motion = math.exp(-residual / (.6 + 2. * gap / 30))
    report['score'] = .5 * motion + .35 * (appearance if appearance is not None else .85) + .15 * (colour if colour is not None else .7)
    report['ok'] = True
    report['reason'] = 'pass'
    return report


def cluster_reachability(left, right, max_gap=MAX_GAP):
    gate = reachability(left, right, max_gap=max_gap)
    if gate['ok'] or gate['reason'] != 'residual':
        return gate
    if not (near_pitch_edge(left['positions'][left['end']]) or near_pitch_edge(right['positions'][right['start']])):
        return gate
    gap = right['start'] - left['end']
    left_s, right_s = left['samples'][-3:], right['samples'][:3]
    if not (left_s and right_s):
        return gate
    appearance = float(unit(np.mean([s[1] for s in left_s], 0)) @ unit(np.mean([s[1] for s in right_s], 0)))
    colour = float(unit(np.mean([s[2] for s in left_s], 0)) @ unit(np.mean([s[2] for s in right_s], 0)))
    if appearance < .80 or colour < .35:
        gate['appearance'] = appearance
        gate['colour'] = colour
        gate['reason'] = 'appearance'
        return gate
    motion = np.exp(-gate['residual_m'] / (.6 + 2. * gap / 30))
    gate.update(ok=True, reason='pass_edge_residual', appearance=appearance, colour=colour,
                score=.5 * float(motion) + .35 * appearance + .15 * colour)
    return gate


def public_edge(edge):
    keep = ('from_', 'to', 'gap', 'companion_count', 'companions', 'left_frames', 'right_frames',
            'left_last', 'right_first', 'distance_m', 'residual_m', 'appearance', 'colour',
            'score', 'reason', 'accepted', 'round', 'left_kit', 'right_kit', 'kit',
            'out_margin', 'in_margin')
    out = {key: edge[key] for key in keep if key in edge}
    if 'from_' in out:
        out['from'] = out.pop('from_')
    return out


def audit_errors(before_rows, after_rows, nodes, skip, accepted, selected=()):
    overlapping = []
    by_id = defaultdict(list)
    for frame, identity, *_ in after_rows:
        if identity not in skip:
            by_id[identity].append(frame)
    for identity, frames in by_id.items():
        if len(frames) != len(set(frames)):
            overlapping.append(identity)
    selected = set(selected)
    selected_hits = [{'from': e.get('from_', e.get('from')), 'to': e['to']} for e in accepted
                     if e.get('from_', e.get('from')) in selected and e['to'] in selected]
    conflicts = len(after_rows) - len({(r[0], r[1]) for r in after_rows})
    kits = defaultdict(set)
    for node in nodes:
        if node['quarantine'] or node['output_id'] in skip or node['kit'] < 0:
            continue
        kits[node['output_id']].add(node['kit'])
    kit_conflicts = {i: sorted(v) for i, v in kits.items() if len(v) > 1}
    before_ids = {r[1] for r in before_rows if r[1] not in skip}
    after_ids = {r[1] for r in after_rows if r[1] not in skip}
    return dict(
        same_frame_conflicts=conflicts, overlapping_frame_ids=overlapping, kit_conflicts=kit_conflicts,
        selected_pair_collisions=selected_hits, row_count_unchanged=len(before_rows) == len(after_rows),
        normal_ids_before=len(before_ids), normal_ids_after=len(after_ids),
        ids_reduced=len(before_ids) - len(after_ids), accepted_merges=len(accepted),
        error_association_found=bool(conflicts or overlapping or kit_conflicts or selected_hits
                                     or len(before_rows) != len(after_rows)),
    )


def track_table(rows, nodes, skip, source_offset=3150):
    grouped = nodes_by_output(nodes, skip)
    by_id = defaultdict(list)
    for row in rows:
        if row[1] not in skip:
            by_id[row[1]].append(row)
    table = []
    for identity, items in by_id.items():
        items = sorted(items)
        frames = [r[0] for r in items]
        table.append(dict(
            id=identity, first_clip=frames[0], last_clip=frames[-1],
            first_source=frames[0] + source_offset, last_source=frames[-1] + source_offset,
            observed_frames=len(items), observed_seconds=len(items) / 30,
            span_seconds=(frames[-1] - frames[0] + 1) / 30,
            longest_contiguous_seconds=max(map(len, runs(frames))) / 30,
            nodes=len(grouped.get(identity, [])),
        ))
    table.sort(key=lambda row: -row['observed_frames'])
    return table
