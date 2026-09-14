"""Run remaining single-variable experiments: GTA Connector, then frozen part ReID."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from eval_identity_merge import evaluate
from features import sha
from part_features import SPORTS_WEIGHT, gallery_prototypes, replace_audit_features
from second_stage import run_second_stage

RUNS = ROOT.parent / 'reid_pipeline_runs'
A3_FEATURES = RUNS / 'onemin_trackval_20260910' / 'features'
GROUPS = ROOT / 'reid_model' / 'identity_groups.json'
AUDIT = ROOT / 'data' / 'audit_data.pkl'
MANIFEST = RUNS / 'selftrain_trackval_20260910' / 'data' / 'manifest.csv'


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def a3_second(**extra):
    pipe = json.loads((ROOT / 'configs' / 'pipeline_config.json').read_text(encoding='utf-8'))
    second = dict(pipe['second_stage'])
    second.update(
        occupancy_veto=False, mixed_box_exclude=True, jersey_agree=False,
        allow_new_match_references=True, use_training_references=True,
        **extra,
    )
    return second


def score(out, features):
    report = evaluate(AUDIT, out / 'trajectory_mapping.json', GROUPS)
    dump(out / 'identity_merge_eval.json', report)
    return report


def summarize(name, result, report, extra=None):
    people = []
    for p in report['people']:
        people.append(dict(name=p['name'], complete=p['complete'], contaminated=p['contaminated'],
                           split=p['split'],
                           components=[[c['id'], c['members'], c['frames']] for c in p['components']]))
    data = dict(
        experiment=name,
        ids=(result['report']['normal_ids_before'], result['report']['normal_ids_after']),
        edges=result['report']['accepted_edges'],
        complete_people=report['complete_people'],
        split_people=report['split_people'],
        contaminated_people=report['contaminated_people'],
        rejections=result['report'].get('rejections'),
        people=people,
    )
    if extra:
        data.update(extra)
    return data


def run_gta():
    out = RUNS / 'onemin_trackval_gta'
    assoc = out / 'association'
    if (assoc / 'association_report.json').exists():
        report = json.loads((assoc / 'identity_merge_eval.json').read_text(encoding='utf-8'))
        result = dict(report=json.loads((assoc / 'association_report.json').read_text(encoding='utf-8')))
        return summarize('gta_connect_only', result, report, extra=dict(
            frozen='A3 trackval features + mixed_box_exclude; occupancy off; jersey off; official GTA alpha=0.4',
            gta_merges=result['report'].get('gta_merges'),
        ))
    second = a3_second(
        reference_file=str((A3_FEATURES / 'reference_bank.npz').resolve()),
        gta_connect=True, gta_merge_dist=0.4,
        sparse_bridge=False, local_matching=False,
    )
    result = run_second_stage(A3_FEATURES / 'features.pkl', assoc, second)
    report = score(assoc, A3_FEATURES / 'features.pkl')
    summary = summarize('gta_connect_only', result, report, extra=dict(
        frozen='A3 trackval features + mixed_box_exclude; occupancy off; jersey off; official GTA alpha=0.4',
        gta_merges=result['report'].get('gta_merges'),
        gta_merge_dist=0.4,
    ))
    dump(out / 'summary.json', summary)
    return summary


def run_part(device='cuda'):
    out = RUNS / 'onemin_trackval_part'
    feats = out / 'features'
    assoc = out / 'association'
    pkl = feats / 'features.pkl'
    if not pkl.exists():
        print('encoding frozen part features', flush=True)
        replace_audit_features(A3_FEATURES / 'features.pkl', A3_FEATURES / 'crops.csv', pkl, device=device)
        proto = gallery_prototypes(MANIFEST, device=device)
        np.savez(feats / 'reference_bank.npz', prototypes=proto, checkpoint=sha(SPORTS_WEIGHT),
                 manifest=sha(MANIFEST), gallery_count=int(proto.shape[0]), max_frame=1049)
        dump(feats / 'feature_metadata.json', pickle_meta(pkl))
    if (assoc / 'association_report.json').exists():
        report = json.loads((assoc / 'identity_merge_eval.json').read_text(encoding='utf-8'))
        result = dict(report=json.loads((assoc / 'association_report.json').read_text(encoding='utf-8')))
        return summarize('frozen_part_reid_only', result, report)
    meta = pickle_meta(pkl)
    second = a3_second(
        reference_file=str((feats / 'reference_bank.npz').resolve()),
        expected_dimension=int(meta['dimension']),
    )
    result = run_second_stage(pkl, assoc, second)
    report = score(assoc, pkl)
    summary = summarize('frozen_part_reid_only', result, report, extra=dict(
        frozen='A3 association rules + mixed_box_exclude; occupancy off; jersey off; GTA off',
        encoder=meta.get('encoder'), dimension=meta.get('dimension'),
        weight=str(SPORTS_WEIGHT),
    ))
    dump(out / 'summary.json', summary)
    return summary


def pickle_meta(pkl):
    import pickle
    audit = pickle.loads(Path(pkl).read_bytes())
    return audit.get('feature_metadata') or {}


def main():
    gta = run_gta()
    dump(RUNS / 'onemin_trackval_gta' / 'summary.json', gta)
    print(json.dumps({k: gta[k] for k in gta if k != 'people'}, ensure_ascii=False, indent=2), flush=True)
    for p in gta['people']:
        print('GTA', p['name'], 'complete', p['complete'], 'contaminated', p['contaminated'], p['components'], flush=True)
    part = run_part()
    dump(RUNS / 'onemin_trackval_part' / 'summary.json', part)
    print(json.dumps({k: part[k] for k in part if k != 'people'}, ensure_ascii=False, indent=2), flush=True)
    for p in part['people']:
        print('PART', p['name'], 'complete', p['complete'], 'contaminated', p['contaminated'], p['components'], flush=True)
    a3 = json.loads((RUNS / 'onemin_trackval_adjacent' / 'association' / 'identity_merge_eval.json').read_text(encoding='utf-8'))
    jersey = json.loads((RUNS / 'onemin_trackval_jersey' / 'summary.json').read_text(encoding='utf-8'))
    comparison = dict(
        a3=dict(complete=a3['complete_people'], split=a3['split_people'], contaminated=a3['contaminated_people']),
        jersey=dict(complete=jersey['complete_people'], split=jersey['split_people'], contaminated=jersey['contaminated_people'],
                    mapping_identical_to_a3=jersey.get('mapping_identical_to_a3')),
        gta=dict(complete=gta['complete_people'], split=gta['split_people'], contaminated=gta['contaminated_people'],
                 ids=gta['ids'], edges=gta['edges']),
        part=dict(complete=part['complete_people'], split=part['split_people'], contaminated=part['contaminated_people'],
                  ids=part['ids'], edges=part['edges']),
    )
    dump(RUNS / 'onemin_trackval_experiments_20260911' / 'comparison.json', comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
