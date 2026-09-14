"""Single-variable: gap-scaled appearance/absorb floors, then frozen third-stage rules."""
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
OUT = RUNS / 'onemin_trackval_gapfloor'
GROUPS = ROOT / 'reid_model' / 'identity_groups.json'
AUDIT = ROOT / 'data' / 'audit_data.pkl'


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def brief(report):
    return [{'name': p['name'], 'complete': p['complete'], 'contaminated': p['contaminated'],
             'components': [[c['id'], c['members'], c['frames']] for c in p['components']]}
            for p in report['people']]


def main():
    pipe = json.loads((ROOT / 'configs' / 'pipeline_config.json').read_text(encoding='utf-8'))
    second = dict(pipe['second_stage'])
    second.update(
        reference_file=str((FEATURES / 'reference_bank.npz').resolve()),
        allow_new_match_references=True, use_training_references=True,
        occupancy_veto=False, mixed_box_exclude=True, jersey_agree=False,
        gta_additive=True, gta_merge_dist=0.4,
        rule_geometry=False, quarantine_bridge=False, rule_only=False,
        gap_scaled_floor=True,
        appearance_floor_near=0.76, appearance_floor_far=0.60,
        appearance_floor_span=180, unmatched_premium_near=0.04, unmatched_premium_far=0.0,
        absorb_near=0.86, absorb_far=0.66,
    )
    assoc2 = OUT / 'second'
    r2 = run_second_stage(FEATURES / 'features.pkl', assoc2, second)
    e2 = evaluate(AUDIT, assoc2 / 'trajectory_mapping.json', GROUPS)
    dump(assoc2 / 'identity_merge_eval.json', e2)
    third = dict(second)
    third.update(pipe.get('third_stage') or {})
    third.update(
        rule_only=True, gta_additive=False, gta_connect=False,
        seed_mapping={int(k): int(v) for k, v in r2['mapping'].items()},
        use_training_references=True,
    )
    assoc3 = OUT / 'third'
    r3 = run_second_stage(FEATURES / 'features.pkl', assoc3, third)
    e3 = evaluate(AUDIT, assoc3 / 'trajectory_mapping.json', GROUPS)
    dump(assoc3 / 'identity_merge_eval.json', e3)
    summary = dict(
        experiment='gap_scaled_appearance_then_third_rules',
        curve='near 0.76 at gap=0 -> far 0.60 at 180 frames; unmatched premium 0.04->0; absorb 0.86->0.66',
        second=dict(ids=(r2['report']['normal_ids_before'], r2['report']['normal_ids_after']),
                    edges=r2['report']['accepted_edges'], complete=e2['complete_people'],
                    split=e2['split_people'], contaminated=e2['contaminated_people'],
                    people=brief(e2)),
        third=dict(ids=(r3['report']['normal_ids_before'], r3['report']['normal_ids_after']),
                   edges=r3['report']['accepted_edges'], complete=e3['complete_people'],
                   split=e3['split_people'], contaminated=e3['contaminated_people'],
                   rule_merges=r3['report'].get('rule_merges'), people=brief(e3)),
    )
    dump(OUT / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in summary if k != 'second' and k != 'third'}, ensure_ascii=False, indent=2))
    print('SECOND complete', e2['complete_people'], 'contam', e2['contaminated_people'],
          'ids', r2['report']['normal_ids_after'])
    for p in e2['people']:
        print(' S', p['name'], p['complete'], p['contaminated'],
              [[c['id'], c['members'], c['frames']] for c in p['components']])
    print('THIRD complete', e3['complete_people'], 'contam', e3['contaminated_people'],
          'ids', r3['report']['normal_ids_after'])
    for p in e3['people']:
        print(' T', p['name'], p['complete'], p['contaminated'],
              [[c['id'], c['members'], c['frames']] for c in p['components']])


if __name__ == '__main__':
    main()
