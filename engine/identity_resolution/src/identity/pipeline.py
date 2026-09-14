"""Run dirty-trim + kit-cluster 1-to-1 + same-kit multi assignment."""
from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path

from identity.common import (
    audit_errors, public_edge, remap_nodes, remap_rows, statistics, track_table,
    write_json, write_mot, csv_write,
)
from identity.multi_assign import merge_cluster_nm, merge_hole_fills
from identity.one_to_one import merge_cluster_swaps
from identity.trim import trim_dirty_segments


def load_selected(selection_path: Path | None):
    if selection_path and selection_path.exists():
        data = json.loads(selection_path.read_text(encoding='utf8'))
        return tuple(data.get('candidate_ids') or data.get('selected_ids') or ())
    return ()


def refine(audit, evidence, selected=()):
    nodes, rows, trimmed = trim_dirty_segments(copy.deepcopy(audit['nodes']), copy.deepcopy(audit['rows']), evidence)
    skip = {node['output_id'] for node in nodes if node['quarantine']}
    uf, cluster, cluster_inspected = merge_cluster_swaps(rows, nodes, skip, selected)
    uf, nm, nm_inspected = merge_cluster_nm(rows, nodes, skip, uf, selected)
    uf, holes, hole_inspected = merge_hole_fills(rows, nodes, skip, uf, selected)
    after_rows = remap_rows(rows, uf)
    after_nodes = remap_nodes(nodes, uf)
    after_skip = {node['output_id'] for node in after_nodes if node['quarantine']}
    return dict(
        rows=after_rows, nodes=after_nodes, skip=after_skip, trimmed=trimmed,
        cluster=cluster, nm=nm, holes=holes,
        cluster_inspected=cluster_inspected, nm_inspected=nm_inspected, hole_inspected=hole_inspected,
        uf=uf, selected_after={uf.find(i) for i in selected} if selected else set(),
    )


def export_result(result, out: Path, source_audit_rows, selected=(), source_offset=3150):
    out.mkdir(parents=True, exist_ok=True)
    after_rows, after_nodes, after_skip = result['rows'], result['nodes'], result['skip']
    accepted = result['cluster'] + result['nm'] + result['holes']
    audit = audit_errors(source_audit_rows, after_rows, after_nodes, after_skip, accepted, selected)
    metadata = track_table(after_rows, after_nodes, after_skip, source_offset=source_offset)
    write_mot(out / 'tracking_long.txt', after_rows)
    if result['selected_after']:
        write_mot(out / 'selected_identities.txt', [r for r in after_rows if r[1] in result['selected_after']])
    csv_write(out / 'track_lengths.csv', metadata)
    write_json(out / 'merges.json', dict(
        cluster=[public_edge(e) for e in result['cluster']],
        cluster_nm=[public_edge(e) for e in result['nm']],
        hole_fill=[public_edge(e) for e in result['holes']],
    ))
    write_json(out / 'audit.json', audit)
    write_json(out / 'summary.json', dict(
        dirty_frames_quarantined=result['trimmed'],
        cluster_one_to_one=len(result['cluster']),
        cluster_nm=len(result['nm']),
        hole_fill=len(result['holes']),
        normal=statistics([r for r in after_rows if r[1] not in after_skip]),
        all=statistics(after_rows),
        selected_ids_after=sorted(result['selected_after']),
        audit=audit,
        training_run=False,
    ))
    with (out / 'audit_data.pkl').open('wb') as handle:
        pickle.dump(dict(nodes=after_nodes, rows=after_rows, selected=accepted), handle)
    return audit, metadata


