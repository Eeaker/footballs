"""Feature extraction -> label-blind association -> optional gated self-training."""
import argparse
from datetime import datetime
import json
import pickle
import sys
from pathlib import Path

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def load_config(path):
    c=json.loads(Path(path).read_text(encoding='utf-8'))
    for key in ['audit_data','video','reid_config','reid_weights','output_dir','manifest']:
        value=Path(c['paths'][key]);c['paths'][key]=str((ROOT/value).resolve()) if not value.is_absolute() else str(value)
    if c['reassociation'].get('gt_filter'):raise ValueError('GT filtering is forbidden')
    weights=c['reassociation']['scoring_weights']
    if abs(sum(weights.values())-1)>1e-6 or min(weights.values())<0:raise ValueError('Invalid scoring weights')
    if c['reid']['feature_dim']!=3840:raise ValueError('TransReID JPM outputs 3840 dimensions')
    return c


def execute(config,out,stage=0,device='cpu',selftrain=False,rounds=1,tracking_path=None,second_stage=False):
    from features import extract_features
    from trajectory_reassociation import run_reassociation
    from render_video import render_video_func
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'run.json').exists():raise ValueError('Run exists; choose a new output directory')
    paths=config['paths'];model_config=json.loads(Path(paths['reid_config']).read_text())
    report=dict(status='running',stage=stage,training_requested=selftrain,second_stage=second_stage,config=config)
    marker=out/'run.json';marker.write_text(json.dumps(report,indent=2),encoding='utf-8')
    try:
        weight=paths['reid_weights']
        source_ids={};review_ids=set()
        if stage==3 or selftrain:
            from selftraining import run_self_training
            result=run_self_training(config,out/'selftrain',device,rounds=rounds)
            report['selftrain']=result
            weight=result['active_weight']
            if stage==3:
                report['status']='complete';return report
        if stage==5:
            if tracking_path is None:raise ValueError('Render-only stage requires --tracking')
            tracking=Path(tracking_path)
            with Path(paths['audit_data']).open('rb') as f:q={n['output_id'] for n in pickle.load(f)['nodes'] if n['quarantine']}
        else:
            cache=extract_features(paths['audit_data'],paths['video'],weight,model_config,out/'features',
                                   config.get('inference_quality',config['quality']),device,config['reid']['batch_size'])
            report['feature_cache']=str(cache)
            if stage==2:
                report['status']='complete';return report
            if second_stage:
                from second_stage import run_second_stage
                second_config=dict(config.get('second_stage',{}))
                if second_config.get('use_training_references'):
                    from features import build_reference_bank
                    reference=build_reference_bank(dict(paths,reid_weights=str(weight)),model_config,
                        out/'features/reference_bank.npz',device,config['reid']['batch_size'])
                    second_config['reference_file']=str(reference.resolve())
                result=run_second_stage(cache,out/'association',second_config)
                third=config.get('third_stage')
                if third:
                    third_config=dict(second_config)
                    third_config.update(third)
                    third_config['rule_only']=True
                    third_config['gta_additive']=False
                    third_config['gta_connect']=False
                    third_config['seed_mapping']={int(k):int(v) for k,v in result['mapping'].items()}
                    result=run_second_stage(cache,out/'third_stage',third_config)
                    report['third_stage']=result['report']
            else:
                result=run_reassociation(cache,out/'association',config['reassociation'])
            report['association']=result['report'];tracking=Path(result['output_tracking'])
            with cache.open('rb') as f:render_audit=pickle.load(f)
            q={n['output_id'] for n in render_audit['nodes'] if n['quarantine']}
            source_ids={(r[0],result['mapping'].get(r[1],r[1])):r[1] for r in render_audit['rows']}
            review_ids={e['from_oid'] for e in result['accepted_edges'] if e.get('requires_review')}
        if stage in [0,5]:
            r=config['render'];report['render']=render_video_func(paths['video'],tracking,out/'review.mp4',
                r['fps'],r['sample_rate'],quarantine_ids=q,show_confidence=r['show_confidence'],show_legend=r['show_legend'],
                source_ids=source_ids,review_ids=review_ids)
        report['active_weight']=str(weight);report['status']='complete';return report
    except Exception as exc:
        report['status']='failed';report['error']=repr(exc);raise
    finally:marker.write_text(json.dumps(report,indent=2),encoding='utf-8')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/pipeline_config.json')
    p.add_argument('--stage',type=int,choices=[0,1,2,3,4,5],default=0)
    p.add_argument('--output',type=Path);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--selftrain',action='store_true');p.add_argument('--rounds',type=int,default=1)
    p.add_argument('--tracking',type=Path,help='Explicit MOT result for render-only stage 5')
    association=p.add_mutually_exclusive_group()
    association.add_argument('--second-stage',action='store_true',help='Short-gap/re-entry association; not automatic training labels')
    association.add_argument('--strict-association',action='store_true',help='Use the previous strict association baseline')
    a=p.parse_args(argv);c=load_config(a.config)
    out=a.output or Path(c['paths']['output_dir'])/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    use_second=a.second_stage or (not a.strict_association and c['pipeline'].get('association_mode')=='second_stage')
    result=execute(c,out,a.stage,a.device,a.selftrain or c['selftrain']['enabled'],a.rounds,a.tracking,use_second)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
