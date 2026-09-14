"""Replay corrected initial IDs using unchanged, observation-verified crop features."""
import argparse,copy,json,pickle,sys
from pathlib import Path
from collections import defaultdict
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from second_stage import run_second_stage,geometry,DEFAULTS
from motion_vote import recover_motion,recover_split_brackets,recover_one_sided_review
from trajectory_reassociation import write_result,group_tracks
from features import sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('audit','features','reference','output'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise ValueError('Use a fresh output')
    audit=pickle.loads(args.audit.read_bytes());old=pickle.loads(args.features.read_bytes())
    oldnodes={n['id']:n for n in old['nodes']}
    for n in audit['nodes']:
        source=oldnodes[n['id']]
        if n['source_local']!=source['source_local'] or n['rows']!=source['rows']:
            raise ValueError('Observation changed; extract fresh features instead')
        n['samples']=copy.deepcopy(source['samples'])
    audit['feature_metadata']=copy.deepcopy(old['feature_metadata'])
    audit['feature_metadata']['reused_verified_observations_from']=str(args.features.resolve())
    audit['feature_metadata']['refined_audit_sha256']=sha(args.audit)
    cached=out/'features.pkl';cached.write_bytes(pickle.dumps(audit))
    config=json.loads((ROOT/'configs/pipeline_config.json').read_text())['second_stage']
    config['reference_file']=str(args.reference.resolve());config['use_training_references']=True
    baseline=run_second_stage(cached,out/'second_stage',config)
    outputs={'baseline':(baseline['mapping'],[])}
    for mode in ('metric','image','both','split_metric','split_image','split_both'):
        fn=recover_split_brackets if mode.startswith('split_') else recover_motion
        mapping,edges,diagnostics=fn(audit['nodes'],baseline['mapping'],mode.replace('split_',''))
        folder=out/mode;folder.mkdir(exist_ok=True);write_result(audit,mapping,folder)
        for name,data in [('mapping',mapping),('edges',edges),('diagnostics',diagnostics)]:
            (folder/(name+'.json')).write_text(json.dumps(data,indent=2),encoding='utf-8')
        outputs[mode]=(mapping,edges)
    mapping,edges,diagnostics=recover_one_sided_review(audit['nodes'],outputs['split_metric'][0])
    folder=out/'one_sided_review';folder.mkdir(exist_ok=True);write_result(audit,mapping,folder)
    for name,data in [('mapping',mapping),('edges',edges),('diagnostics',diagnostics)]:
        (folder/(name+'.json')).write_text(json.dumps(data,indent=2),encoding='utf-8')
    outputs['one_sided_review']=(mapping,edges)
    # Historical labels belong to historical IDs. Lift via unchanged node/row provenance.
    groups=json.loads((ROOT/'reid_model/identity_groups.json').read_text())['groups']
    truth={i:label for label,ids in groups.items() for i in ids}
    result={}
    for mode,(mapping,edges) in outputs.items():
        observed=defaultdict(set);labels=defaultdict(set)
        for n in audit['nodes']:
            if n['quarantine']:continue
            original=oldnodes[n['id']]['output_id']
            component=mapping[n['output_id']]
            observed[original].add(component)
            if original in truth:labels[component].add(truth[original])
        result[mode]=dict(normal_components=len(set(mapping.values())),known_components=len(labels),
            mixed_known_components={k:sorted(v) for k,v in labels.items() if len(v)>1},
            historical_ids_split={k:sorted(v) for k,v in observed.items() if len(v)>1},edges=edges,
            known_identity_components={label:sorted({c for i in ids for c in observed[i]}) for label,ids in groups.items()})
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
