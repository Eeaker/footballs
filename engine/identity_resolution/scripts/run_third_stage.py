"""Third merge: freeze A3+GTA mapping, then rule/quarantine geometry only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from eval_identity_merge import evaluate
from second_stage import run_second_stage

RUNS = ROOT.parent / 'reid_pipeline_runs'
FEATURES = RUNS / 'onemin_trackval_20260910' / 'features'
SECOND = RUNS / 'onemin_trackval_a3_gta' / 'association'
OUT = RUNS / 'onemin_trackval_third'
GROUPS = ROOT / 'reid_model' / 'identity_groups.json'
AUDIT = ROOT / 'data' / 'audit_data.pkl'


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def people_brief(report):
    return [{'name': p['name'], 'complete': p['complete'], 'contaminated': p['contaminated'],
             'components': [[c['id'], c['members'], c['frames']] for c in p['components']]}
            for p in report['people']]


def main():
    pipe = json.loads((ROOT / 'configs' / 'pipeline_config.json').read_text(encoding='utf-8'))
    second = json.loads((SECOND / 'identity_merge_eval.json').read_text(encoding='utf-8'))
    seed = {int(k): int(v) for k, v in json.loads((SECOND / 'trajectory_mapping.json').read_text(encoding='utf-8')).items()}
    cfg = dict(pipe['second_stage'])
    cfg.update(pipe.get('third_stage') or {})
    cfg.update(
        rule_only=True, gta_additive=False, gta_connect=False,
        seed_mapping=seed, use_training_references=False,
        mixed_box_exclude=True, occupancy_veto=False, jersey_agree=False,
        reference_file=str((FEATURES / 'reference_bank.npz').resolve()),
        allow_new_match_references=True,
    )
    assoc = OUT / 'third_stage'
    result = run_second_stage(FEATURES / 'features.pkl', assoc, cfg)
    report = evaluate(AUDIT, assoc / 'trajectory_mapping.json', GROUPS)
    dump(assoc / 'identity_merge_eval.json', report)
    modes = {}
    for e in result['accepted_edges']:
        modes[e.get('mode')] = modes.get(e.get('mode'), 0) + 1
    now = {int(k): int(v) for k, v in json.loads((assoc / 'trajectory_mapping.json').read_text(encoding='utf-8')).items()}
    changed = sorted({i for i in seed if seed[i] != now.get(i, i) or now.get(i) != seed[i]})
    # ids that joined a different component than second stage
    def comp(m, i):
        r = m.get(i, i)
        return tuple(sorted(j for j, v in m.items() if v == r))
    newly = []
    for i in sorted(set(seed) & set(now)):
        if comp(seed, i) != comp(now, i) and i == min(comp(now, i)):
            newly.append(list(comp(now, i)))
    summary = dict(
        experiment='third_rule_on_frozen_a3_gta',
        second=dict(ids=35, complete=second['complete_people'], split=second['split_people'],
                    contaminated=second['contaminated_people']),
        third=dict(ids=(result['report']['normal_ids_before'], result['report']['normal_ids_after']),
                   edges=result['report']['accepted_edges'], modes=modes,
                   complete=report['complete_people'], split=report['split_people'],
                   contaminated=report['contaminated_people'],
                   rule_merges=result['report'].get('rule_merges')),
        new_components=newly,
        people=people_brief(report),
    )
    dump(OUT / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in summary if k != 'people'}, ensure_ascii=False, indent=2))
    for p in report['people']:
        print(p['name'], 'complete', p['complete'], 'contaminated', p['contaminated'],
              [[c['id'], c['members'], c['frames']] for c in p['components']])
    for e in result['accepted_edges']:
        print(e.get('mode'), e['from_oid'], '->', e['to_oid'], 'gap', e.get('gap'),
              'm', None if e.get('distance') is None else round(float(e['distance']), 2),
              'Q', e.get('quarantine_support'))


if __name__ == '__main__':
    main()
