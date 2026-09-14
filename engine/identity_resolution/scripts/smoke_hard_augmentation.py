"""Prepare augmentation comparison; optimizer work requires an explicit mode flag."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from scripts.run_pipeline import load_config
from selftraining import train_round,load_manifest,validate_dataset
from features import sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--supplement',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--run-comparison',action='store_true',help='Explicitly run bounded validation-gated training')
    modes.add_argument('--smoke',action='store_true',help='Explicitly perform one optimizer step per configuration')
    a=p.parse_args();c=load_config(ROOT/'configs/pipeline_config.json');paths=c['paths'];rows=load_manifest(paths['manifest']);validate_dataset(paths,rows)
    extra=json.loads(a.supplement.read_text(encoding='utf-8'));original=sha(paths['reid_weights'])
    for r in extra:
        if not r['training_allowed'] or r['split']!='train' or not 0<=r['frame']<=1049:raise ValueError('Unsafe supplemental record')
        if sha(r['image'])!=r['sha256']:raise ValueError('Crop changed')
    config=json.loads(Path(paths['reid_config']).read_text());config.update(workers=0,eval_batch=8)
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('Use a fresh output')
    a.output.mkdir(parents=True,exist_ok=True);result={}
    for name in ('old_augmentation','new_augmentation'):
        model_config=dict(config)
        if name=='new_augmentation':model_config.update(hard_augmentation=dict(motion_probability=.2,downsample_probability=.1),erasing_scale=[.02,.2])
        out=a.output/name;out.mkdir();(out/'config.json').write_text(json.dumps(model_config,indent=2))
        if not (a.smoke or a.run_comparison):
            result[name]=dict(training_steps=0,status='prepared_only',promoted=False)
            continue
        result[name]=train_round(paths['reid_weights'],rows,extra,model_config,c['selftrain'],out,'cuda',smoke=not a.run_comparison)
    assert sha(paths['reid_weights'])==original
    (a.output/'summary.json').write_text(json.dumps(dict(runs=result,smoke_only=a.smoke,prepare_only=not (a.smoke or a.run_comparison),production_weight_unchanged=True),indent=2))

if __name__=='__main__':main()
