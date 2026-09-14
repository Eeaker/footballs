"""Single-variable experiment: A3 association plus positive-only jersey agreement."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from jersey_vote import ocr_crops
from second_stage import run_second_stage
from eval_identity_merge import evaluate

RUNS = ROOT.parent / 'reid_pipeline_runs'
A3_FEATURES = RUNS / 'onemin_trackval_20260910' / 'features'
OUT = RUNS / 'onemin_trackval_jersey'


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    votes_path = OUT / 'jersey_votes.json'
    if votes_path.exists():
        votes = json.loads(votes_path.read_text(encoding='utf-8'))
    else:
        votes = ocr_crops(A3_FEATURES / 'crops.csv')
        dump(votes_path, {k: v for k, v in votes.items() if k != 'hits'})
        dump(OUT / 'jersey_ocr_hits.json', votes.get('hits', {}))
    pipe = json.loads((ROOT / 'configs' / 'pipeline_config.json').read_text(encoding='utf-8'))
    second = dict(pipe['second_stage'])
    second.update(
        reference_file=str((A3_FEATURES / 'reference_bank.npz').resolve()),
        allow_new_match_references=True,
        use_training_references=True,
        occupancy_veto=False,
        mixed_box_exclude=True,
        jersey_agree=True,
        jersey_votes_file=str(votes_path.resolve()),
    )
    assoc = OUT / 'association'
    result = run_second_stage(A3_FEATURES / 'features.pkl', assoc, second)
    report = evaluate(ROOT / 'data' / 'audit_data.pkl', assoc / 'trajectory_mapping.json',
                      ROOT / 'reid_model' / 'identity_groups.json')
    dump(assoc / 'identity_merge_eval.json', report)
    summary = dict(
        experiment='jersey_agree_only',
        frozen='A3 trackval + mixed_box_exclude; floors unchanged; occupancy off; same weights',
        voted=votes.get('voted', {}),
        rejections=result['report'].get('rejections'),
        jersey_agree_candidates=result['report'].get('jersey_agree_candidates'),
        jersey_voted_tracks=result['report'].get('jersey_voted_tracks'),
        ids=(result['report']['normal_ids_before'], result['report']['normal_ids_after']),
        edges=result['report']['accepted_edges'],
        complete_people=report['complete_people'],
        split_people=report['split_people'],
        contaminated_people=report['contaminated_people'],
        people=[{k: p[k] for k in ['name', 'complete', 'contaminated', 'split']} |
                {'components': [[c['id'], c['members'], c['frames']] for c in p['components']]}
                for p in report['people']],
    )
    dump(OUT / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in summary if k != 'people'}, ensure_ascii=False, indent=2))
    for person in report['people']:
        print(person['name'], 'complete', person['complete'], 'contaminated', person['contaminated'],
              'components', [[c['id'], c['members'], c['frames']] for c in person['components']])


if __name__ == '__main__':
    main()
