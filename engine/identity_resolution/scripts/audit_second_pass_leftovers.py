"""After third merge: remaining disappear-reappear pairs vs second-stage thresholds."""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
from second_stage import (
    DEFAULTS, adjacent_rival, appearance, appearance_floor, centroid_floor,
    describe, geometry, group_tracks, mean_pairwise_distance, occupancy_index, path_hits,
)

RUNS = ROOT.parent / 'reid_pipeline_runs'
AUDIT = RUNS / 'onemin_trackval_20260910' / 'features' / 'features.pkl'
MAP = RUNS / 'onemin_trackval_third' / 'third_stage' / 'trajectory_mapping.json'
GROUPS = ROOT / 'reid_model' / 'identity_groups.json'
BANK = RUNS / 'onemin_trackval_20260910' / 'features' / 'reference_bank.npz'


def main():
    audit = pickle.loads(AUDIT.read_bytes())
    mapping = {int(k): int(v) for k, v in json.loads(MAP.read_text(encoding='utf-8')).items()}
    groups = json.loads(GROUPS.read_text(encoding='utf-8'))['groups']
    pipe = json.loads((ROOT / 'configs' / 'pipeline_config.json').read_text(encoding='utf-8'))
    c = dict(DEFAULTS, **pipe['second_stage'])
    c.update(mixed_box_exclude=True, occupancy_veto=False)
    tracks = {t['id']: t for t in group_tracks(audit['nodes'])}
    desc = {i: describe(t, c) for i, t in tracks.items()}
    refs = np.load(BANK)['prototypes']
    ref_match = {}
    for i, d in desc.items():
        if d is None:
            continue
        scores = d['centroid'] @ refs.T
        order = np.argsort(-scores)
        vote = float(np.mean(np.argmax(d['features'] @ refs.T, axis=1) == order[0]))
        if scores[order[0]] >= c['reference_threshold'] and scores[order[0]] - scores[order[1]] >= c['reference_margin'] and vote >= c['reference_vote']:
            ref_match[i] = (int(order[0]), float(scores[order[0]]))
    truth = {int(i): name for name, ids in groups.items() for i in ids}
    comp = defaultdict(list)
    for i in tracks:
        comp[mapping.get(i, i)].append(i)

    def virt(mem):
        rows, pos, kits = [], {}, set()
        for i in mem:
            t = tracks[i]
            rows += t['rows']
            pos.update(t['positions'])
            kits |= t['kits']
        rows = sorted(rows, key=lambda r: r[0])
        return dict(members=mem, rows=rows, positions=pos, kits=kits,
                    kit=next(iter(kits)) if len(kits) == 1 else -1,
                    start=rows[0][0], end=rows[-1][0])

    components = {cid: virt(mem) for cid, mem in comp.items()}
    normal_idx = occupancy_index(audit['nodes'], False)
    q_idx = occupancy_index(audit['nodes'], True)
    rows = []
    cids = sorted(components)
    for i, x in enumerate(cids):
        for y in cids[i + 1:]:
            va, vb = components[x], components[y]
            conflict = False
            best = None
            for a in va['members']:
                for b in vb['members']:
                    g, reason = geometry(tracks[a], tracks[b], c)
                    if reason in ('same_frame', 'kit_conflict'):
                        conflict = True
                        break
                    if reason:
                        continue
                    rival = bool(c.get('mixed_box_exclude') and adjacent_rival(tracks[a], tracks[b], c))
                    aff, cent, col = appearance(desc[a], desc[b])
                    ga = ref_match[a][0] if a in ref_match else None
                    gb = ref_match[b][0] if b in ref_match else None
                    agree = ga is not None and ga == gb
                    floor = appearance_floor(g, agree, c, ga, gb) if aff is not None else None
                    gta = mean_pairwise_distance(desc[a], desc[b])
                    nh, qh = path_hits(tracks[a], tracks[b], g, normal_idx, q_idx, c)
                    item = dict(a=a, b=b, gap=g['gap'], dist=g['distance'], rival=rival, aff=aff,
                                cent=cent, col=col, floor=floor, agree=agree, gta=gta,
                                n_hits=len(nh), q_hits=len(qh))
                    if not rival and (best is None or g['gap'] < best['gap']):
                        best = item
                if conflict:
                    break
            if conflict or best is None:
                continue
            names = sorted({truth[i] for i in va['members'] + vb['members'] if i in truth})
            why = []
            app_ok = False
            if best['aff'] is None:
                why.append('no_appearance')
            elif best['floor'] is not None and best['aff'] < best['floor']:
                why.append('below_floor')
            elif best['cent'] is not None and best['cent'] < centroid_floor(c):
                why.append('centroid')
            elif best['col'] is not None and best['col'] < c['min_colour']:
                why.append('colour')
            else:
                app_ok = True
            absorb = len(va['members']) > 1 or len(vb['members']) > 1
            absorb_ok = (not absorb) or best['agree'] or (best['aff'] is not None and best['aff'] >= c['absorb_affinity'])
            if app_ok and not absorb_ok:
                why.append('absorb_0.80')
            second_would = app_ok and absorb_ok
            rows.append(dict(
                A=sorted(va['members']), B=sorted(vb['members']), names=names,
                same_person=len(names) == 1, mixed=len(names) > 1,
                gap=best['gap'], dist=round(best['dist'], 2),
                aff=None if best['aff'] is None else round(float(best['aff']), 3),
                floor=best['floor'], gta=None if best['gta'] is None else round(float(best['gta']), 3),
                agree=best['agree'], n_hits=best['n_hits'], q_hits=best['q_hits'],
                second_would=second_would, why=why,
                frames=(len(va['rows']), len(vb['rows'])), pair=(best['a'], best['b']),
            ))

    print('same-person leftover (geometry OK, not adjacent_rival)')
    leftover = [r for r in rows if r['same_person']]
    leftover.sort(key=lambda r: r['gap'])
    for r in leftover:
        print(r['names'][0], r['A'], '<->', r['B'], 'gap', r['gap'], 'm', r['dist'],
              'aff', r['aff'], 'floor', r['floor'], 'gta', r['gta'],
              'second_would', r['second_would'], 'why', r['why'],
              'pathN', r['n_hits'], 'Q', r['q_hits'], 'frames', r['frames'])
    would = [r for r in leftover if r['second_would']]
    print('same-person that still pass second-stage appearance+absorb:', len(would))
    print('unlabeled/mixed that would pass second-stage:')
    n = 0
    for r in rows:
        if r['second_would'] and not r['same_person']:
            n += 1
            tag = 'MIXED' if r['mixed'] else 'UNLAB'
            print(tag, r['names'], r['A'], r['B'], 'gap', r['gap'], 'm', r['dist'], 'aff', r['aff'], 'agree', r['agree'])
    print('count', n)
    print('same-person gap<=60 still split:')
    for r in leftover:
        if r['gap'] <= 60:
            print(r['names'][0], r['A'], r['B'], 'gap', r['gap'], 'm', r['dist'], 'aff', r['aff'], 'why', r['why'], 'pathN', r['n_hits'])


if __name__ == '__main__':
    main()
