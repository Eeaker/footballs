"""Train anchored pseudo samples without exposing test images to training."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_pipeline import load_config


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/pipeline_config.json')
    p.add_argument('--weights',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--rounds',type=int,default=1);p.add_argument('--epochs',type=int)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--smoke',action='store_true',help='One real optimizer step; never promote its checkpoint')
    a=p.parse_args(argv);c=load_config(a.config)
    if a.weights:c['paths']['reid_weights']=str(a.weights.resolve())
    if a.epochs is not None:
        if a.epochs<1:p.error('--epochs must be positive')
        c['selftrain']['epochs']=a.epochs
    out=a.output or Path(c['paths']['output_dir'])/('selftrain_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    from selftraining import run_self_training
    result=run_self_training(c,out,a.device,a.rounds,a.prepare_only,a.smoke)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
