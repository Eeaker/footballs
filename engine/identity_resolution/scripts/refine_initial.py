"""Conservative initial refinement, owned by reid_pipeline; external inputs/outputs."""
import argparse
import sys
import pickle
import json
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from identity.pipeline import refine,export_result
from features import apply_observation_policy,sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--dense',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('Use a new output directory')
    audit=pickle.loads(a.audit.read_bytes());dense=pickle.loads(a.dense.read_bytes())
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/pipeline_config.json').read_text(encoding='utf-8'))
    policy=config['quality'].get('observation_policy')
    if policy:
        if sha(a.audit) not in policy.get('compatible_audit_sha256',[policy['audit_sha256']]):
            raise ValueError('Reviewed observation exclusions belong to another audit; review provenance first')
        apply_observation_policy(audit,policy)
    result=refine(audit,dense)
    print(export_result(result,a.output,audit['rows'])[0])

if __name__=='__main__':main()
