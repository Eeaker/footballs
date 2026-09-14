"""Quarantine isolated / high-overlap heads and tails so they are not endpoints."""
from __future__ import annotations

import copy

OVERLAP_TRIM = 0.35


def trim_dirty_segments(nodes, rows, evidence, overlap_trim=OVERLAP_TRIM):
    nodes = copy.deepcopy(nodes)
    next_node = max(node['id'] for node in nodes) + 1
    next_out = max(node['output_id'] for node in nodes) + 1
    rebuilt = []
    moved = {}
    for node in nodes:
        if node['quarantine'] or len(node['rows']) < 2:
            rebuilt.append(node)
            continue
        local = node['source_local']
        ordered = sorted(node['rows'], key=lambda row: row[0])
        pieces = []
        for row in ordered:
            if not pieces or row[0] != pieces[-1][-1][0] + 1:
                pieces.append([])
            pieces[-1].append(row)
        body = max(pieces, key=len)
        keep, drop = [], []
        for piece in pieces:
            if piece is not body and len(piece) <= 15:
                drop.extend(piece)
            else:
                keep.extend(piece)
        keep.sort(key=lambda row: row[0])

        def overlap(row):
            return float(evidence.get((row[0], local), {}).get('overlap', 0))

        while keep and overlap(keep[0]) >= overlap_trim:
            drop.append(keep.pop(0))
        while keep and overlap(keep[-1]) >= overlap_trim:
            drop.append(keep.pop())
        if not keep:
            node['quarantine'] = True
            rebuilt.append(node)
            continue
        if drop:
            quarantined = dict(node)
            quarantined['id'] = next_node
            next_node += 1
            quarantined['output_id'] = next_out
            next_out += 1
            quarantined['quarantine'] = True
            quarantined['rows'] = sorted(drop, key=lambda row: row[0])
            quarantined['start'] = quarantined['rows'][0][0]
            quarantined['end'] = quarantined['rows'][-1][0]
            kept_frames = {row[0] for row in keep}
            quarantined['samples'] = [s for s in node['samples'] if s[0] not in kept_frames]
            drop_frames = {r[0] for r in drop}
            quarantined['positions'] = {f:p for f,p in node['positions'].items() if f in drop_frames}
            rebuilt.append(quarantined)
            for row in quarantined['rows']:
                moved[(node['output_id'], row[0])] = quarantined['output_id']
        node['rows'] = keep
        node['start'] = keep[0][0]
        node['end'] = keep[-1][0]
        kept_frames = {row[0] for row in keep}
        node['samples'] = [s for s in node['samples'] if s[0] in kept_frames]
        node['positions'] = {f:p for f,p in node['positions'].items() if f in kept_frames}
        rebuilt.append(node)
    new_rows = []
    for row in rows:
        key = (row[1], row[0])
        new_rows.append((row[0], moved[key], *row[2:]) if key in moved else row)
    return rebuilt, new_rows, len(moved)
