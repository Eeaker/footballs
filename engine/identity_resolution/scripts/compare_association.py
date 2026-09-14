"""Frozen-protocol development comparison; identity labels are read only after inference."""
import argparse
from collections import defaultdict
from itertools import combinations
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trajectory_reassociation import run_reassociation
from second_stage import run_second_stage,DEFAULTS


def evaluate(mapping,edges,groups):
    truth={int(i):name for name,ids in groups.items() for i in ids}
    ids=sorted(set(mapping)&set(truth));tp=fp=fn=0
    for a,b in combinations(ids,2):
        same=truth[a]==truth[b];joined=mapping[a]==mapping[b]
        tp+=same and joined;fp+=not same and joined;fn+=same and not joined
    components=defaultdict(list)
    for i,label in mapping.items():components[label].append(i)
    mixed=[v for v in components.values() if len({truth[i] for i in v if i in truth})>1]
    unknown_edges=[e for e in edges if e['from_oid'] not in truth or e['to_oid'] not in truth]
    coverage={name:len({mapping[i] for i in ids if truth[i]==name}) for name in groups}
    return dict(known_ids=len(ids),known_pair_tp=int(tp),known_pair_fp=int(fp),known_pair_fn=int(fn),
        known_pair_precision=tp/(tp+fp) if tp+fp else None,known_pair_recall=tp/(tp+fn) if tp+fn else None,
        mixed_known_components=mixed,unreviewed_edges=len(unknown_edges),
        known_components=len({mapping[i] for i in ids}),components_per_identity=coverage,
        components={str(k):sorted(v) for k,v in components.items()},
        scope='Existing reviewed development clip; not independent unseen-time evaluation')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--enhanced-features',type=Path,help='Run sampling/local-matching/sparse-bridge ablations')
    p.add_argument('--reference-bank',type=Path,help='Add the explicitly supervised frozen-gallery variant')
    a=p.parse_args(argv);a.output.mkdir(parents=True,exist_ok=True)
    protocol=a.output/'protocol.json'
    if protocol.exists():raise ValueError('Comparison already exists; use a new directory')
    settings=json.loads((ROOT/'configs/pipeline_config.json').read_text())
    base=settings['reassociation']
    relaxed=json.loads(json.dumps(base));relaxed.update(max_gap=1800,threshold=.80,ambiguity_margin=.03)
    relaxed['hard_constraints'].update(distance_slack_m=1.5,residual_slack_m=2.,residual_growth_m_s=6.,min_app_sim=.80)
    relaxed['scoring_weights']=dict(motion=.15,appearance=.75,colour=.10)
    appearance_config=dict(settings.get('second_stage',DEFAULTS),use_training_references=False)
    appearance_config.pop('reference_file',None)
    protocols=dict(strict=base,relaxed=relaxed,split_reentry=appearance_config)
    inputs={name:a.features for name in protocols}
    if a.enhanced_features:
        common=dict(appearance_config,local_matching=False,sparse_bridge=False)
        protocols=dict(baseline=common,sampling_only=dict(common),matching_only=dict(common,local_matching=True),
                       combined=dict(common,local_matching=True),with_bridge=dict(common,local_matching=True,sparse_bridge=True))
        inputs={name:a.features if name in ['baseline','matching_only'] else a.enhanced_features for name in protocols}
    if a.reference_bank:
        protocols['gallery_assisted']=dict(appearance_config,local_matching=True,sparse_bridge=True,
            use_training_references=True,reference_file=str(a.reference_bank.resolve()))
        inputs['gallery_assisted']=a.enhanced_features or a.features
    protocol.write_text(json.dumps(dict(features=str(a.features.resolve()),
        inputs={k:str(v.resolve()) for k,v in inputs.items()},methods=protocols),indent=2),encoding='utf-8')
    results={}
    # All algorithms finish before reading the evaluation identity map.
    for name,config in protocols.items():
        fn=run_second_stage if name in ['split_reentry','gallery_assisted'] or a.enhanced_features else run_reassociation
        results[name]=fn(inputs[name],a.output/name,config)
    groups=json.loads((ROOT/'reid_model/identity_groups.json').read_text())['groups']
    scores={}
    for name,result in results.items():
        scores[name]=evaluate(result['mapping'],result['accepted_edges'],groups)
        scores[name]['normal_components']=len(set(result['mapping'].values()))
        scores[name]['accepted_edges']=len(result['accepted_edges'])
        (a.output/name/'development_evaluation.json').write_text(json.dumps(scores[name],indent=2),encoding='utf-8')
    (a.output/'comparison.json').write_text(json.dumps(scores,indent=2),encoding='utf-8')
    print(json.dumps({k:{n:v for n,v in s.items() if n not in ['components','components_per_identity','scope']} for k,s in scores.items()},indent=2))


if __name__=='__main__':main()
