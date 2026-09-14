"""A3 merges frozen, then GTA only adds remaining connections. Also dump unmerged diagnostics."""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from eval_identity_merge import evaluate
from second_stage import (
    appearance, appearance_floor, adjacent_rival, describe, geometry, group_tracks,
    mean_pairwise_distance, run_second_stage, DEFAULTS,
)
from trajectory_reassociation import cosine_sim

RUNS = ROOT.parent / 'reid_pipeline_runs'
A3 = RUNS / 'onemin_trackval_adjacent' / 'association'
FEATURES = RUNS / 'onemin_trackval_20260910' / 'features'
OUT = RUNS / 'onemin_trackval_a3_gta'
GROUPS = ROOT / 'reid_model' / 'identity_groups.json'
AUDIT = ROOT / 'data' / 'audit_data.pkl'


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def mapping_of(path):
    return {int(k): int(v) for k, v in json.loads(Path(path).read_text(encoding='utf-8')).items()}


def components(mapping, ids):
    buckets = defaultdict(list)
    for i in ids:
        buckets[mapping.get(i, i)].append(i)
    return [sorted(v) for v in buckets.values()]


def diagnose(audit, mapping, groups, config):
    c = dict(DEFAULTS, **config)
    tracks = {t['id']: t for t in group_tracks(audit['nodes'])}
    desc = {i: describe(t, c) for i, t in tracks.items()}
    rows = []
    for name, ids in groups.items():
        live = [i for i in ids if i in tracks]
        for a, b in combinations(sorted(live), 2):
            if mapping.get(a, a) == mapping.get(b, b):
                continue
            ta, tb = tracks[a], tracks[b]
            geo, reason = geometry(ta, tb, c)
            aff, centroid, colour = appearance(desc[a], desc[b])
            gta_d = mean_pairwise_distance(desc[a], desc[b])
            rival = bool(c.get('mixed_box_exclude') and reason is None and adjacent_rival(ta, tb, c))
            samples = (0 if desc[a] is None else len(desc[a]['features']),
                       0 if desc[b] is None else len(desc[b]['features']))
            floor = None if geo is None else appearance_floor(geo, False, c, None, None)
            cause = []
            if desc[a] is None or desc[b] is None or min(samples) < c['min_samples']:
                cause.append('too_few_samples')
            if reason:
                cause.append(reason)
            if rival:
                cause.append('adjacent_rival')
            if aff is not None and floor is not None and aff < floor:
                cause.append(f'below_floor_{floor:.2f}')
            if gta_d is not None and gta_d >= c.get('gta_merge_dist', .4):
                cause.append(f'gta_dist_{gta_d:.3f}')
            if cause == ['below_floor_0.80'] and gta_d is not None and gta_d < .4:
                cause.append('would_gta_but_already_checked')
            rows.append(dict(
                person=name, a=a, b=b, frames=(len(ta['rows']), len(tb['rows'])),
                samples=samples, gap=None if geo is None else geo['gap'],
                metres=None if geo is None else geo['distance'],
                affinity=aff, centroid=centroid, colour=colour, gta_distance=gta_d,
                floor=floor, reason=reason, adjacent_rival=rival, cause=cause,
            ))
    return rows


def main():
    pipe = json.loads((ROOT / 'configs' / 'pipeline_config.json').read_text(encoding='utf-8'))
    second = dict(pipe['second_stage'])
    second.update(
        reference_file=str((FEATURES / 'reference_bank.npz').resolve()),
        allow_new_match_references=True, use_training_references=True,
        occupancy_veto=False, mixed_box_exclude=True, jersey_agree=False,
        gta_additive=True, gta_merge_dist=0.4,
    )
    assoc = OUT / 'association'
    if not (assoc / 'association_report.json').exists():
        result = run_second_stage(FEATURES / 'features.pkl', assoc, second)
    else:
        result = dict(report=json.loads((assoc / 'association_report.json').read_text(encoding='utf-8')))
    report = evaluate(AUDIT, assoc / 'trajectory_mapping.json', GROUPS)
    dump(assoc / 'identity_merge_eval.json', report)
    a3 = mapping_of(A3 / 'trajectory_mapping.json')
    now = mapping_of(assoc / 'trajectory_mapping.json')
    groups = json.loads(GROUPS.read_text(encoding='utf-8'))['groups']
    kept, added, lost = [], [], []
    for name, ids in groups.items():
        live = [i for i in ids if i in now]
        a3c = {frozenset(x) for x in components(a3, live)}
        nowc = {frozenset(x) for x in components(now, live)}
        if a3c - nowc:
            lost.append(dict(person=name, a3=[sorted(x) for x in a3c], now=[sorted(x) for x in nowc]))
        extra = nowc - a3c
        if extra and not (a3c - nowc):
            added.append(dict(person=name, extra=[sorted(x) for x in extra], a3=[sorted(x) for x in a3c]))
        if a3c <= nowc:
            kept.append(name)
    audit = pickle.loads((FEATURES / 'features.pkl').read_bytes())
    leftover = diagnose(audit, now, groups, second)
    summary = dict(
        experiment='a3_then_gta_additive',
        frozen='A3 floors/gallery/mixed_box; GTA only adds after A3; occupancy off; jersey off',
        ids=(result['report']['normal_ids_before'], result['report']['normal_ids_after']),
        edges=result['report']['accepted_edges'],
        gta_merges=result['report'].get('gta_merges'),
        complete_people=report['complete_people'],
        split_people=report['split_people'],
        contaminated_people=report['contaminated_people'],
        a3_groups_preserved=kept,
        a3_merges_lost=lost,
        new_components=added,
        people=[{'name': p['name'], 'complete': p['complete'], 'contaminated': p['contaminated'],
                 'components': [[c['id'], c['members'], c['frames']] for c in p['components']]}
                for p in report['people']],
    )
    dump(OUT / 'summary.json', summary)
    dump(OUT / 'unmerged_pairs.json', leftover)
    print(json.dumps({k: summary[k] for k in summary if k != 'people'}, ensure_ascii=False, indent=2))
    for p in report['people']:
        print(p['name'], 'complete', p['complete'], 'contaminated', p['contaminated'],
              [[c['id'], c['members'], c['frames']] for c in p['components']])
    print('unmerged_pairs', len(leftover))
    for row in leftover:
        print(row['person'], row['a'], row['b'], 'aff', row['affinity'], 'gta', row['gta_distance'],
              'gap', row['gap'], 'samples', row['samples'], 'cause', row['cause'])


if __name__ == '__main__':
    main()
