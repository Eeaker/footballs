"""A3 + additive GTA + appearance-free unique geometry / quarantine-bridge."""
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
OUT = RUNS / 'onemin_trackval_rulegeo'
GROUPS = ROOT / 'reid_model' / 'identity_groups.json'
AUDIT = ROOT / 'data' / 'audit_data.pkl'


def main():
    pipe = json.loads((ROOT / 'configs' / 'pipeline_config.json').read_text(encoding='utf-8'))
    second = dict(pipe['second_stage'])
    second.update(
        reference_file=str((FEATURES / 'reference_bank.npz').resolve()),
        allow_new_match_references=True, use_training_references=True,
        occupancy_veto=False, mixed_box_exclude=True, jersey_agree=False,
        gta_additive=True, gta_merge_dist=0.4,
        rule_geometry=True, rule_max_gap=60, rule_max_m=3.5,
        quarantine_bridge=True, quarantine_max_gap=120,
    )
    assoc = OUT / 'association'
    result = run_second_stage(FEATURES / 'features.pkl', assoc, second)
    report = evaluate(AUDIT, assoc / 'trajectory_mapping.json', GROUPS)
    (assoc / 'identity_merge_eval.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    modes = {}
    for e in result['accepted_edges']:
        modes[e.get('mode')] = modes.get(e.get('mode'), 0) + 1
    summary = dict(
        experiment='a3_gta_rule_geometry',
        ids=(result['report']['normal_ids_before'], result['report']['normal_ids_after']),
        edges=result['report']['accepted_edges'],
        modes=modes,
        gta_merges=result['report'].get('gta_merges'),
        rule_merges=result['report'].get('rule_merges'),
        complete_people=report['complete_people'],
        split_people=report['split_people'],
        contaminated_people=report['contaminated_people'],
        people=[{'name': p['name'], 'complete': p['complete'], 'contaminated': p['contaminated'],
                 'components': [[c['id'], c['members'], c['frames']] for c in p['components']]}
                for p in report['people']],
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in summary if k != 'people'}, ensure_ascii=False, indent=2))
    for p in report['people']:
        print(p['name'], 'complete', p['complete'], 'contaminated', p['contaminated'],
              [[c['id'], c['members'], c['frames']] for c in p['components']])
    for e in result['accepted_edges']:
        if e.get('mode') in ('rule_geometry', 'quarantine_bridge', 'gta_connect'):
            print(e.get('mode'), e['from_oid'], e['to_oid'], 'gap', e.get('gap'),
                  'm', e.get('distance'), 'Q', e.get('quarantine_support'))


if __name__ == '__main__':
    main()
