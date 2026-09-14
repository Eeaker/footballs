"""Compare metre and image-plane Kalman recovery on frozen second-stage output."""
import argparse,json,pickle,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from motion_vote import recover_motion
from trajectory_reassociation import write_result
from scripts.compare_association import evaluate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('features','mapping','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'comparison.json').exists():raise ValueError('Use fresh output')
    audit=pickle.loads(a.features.read_bytes());mapping={int(k):v for k,v in json.loads(a.mapping.read_text()).items()}
    results={}
    for mode in ('metric','image','both'):
        m,edges,diagnostics=recover_motion(audit['nodes'],mapping,mode)
        out=a.output/mode;out.mkdir(exist_ok=True)
        write_result(audit,m,out)
        for name,data in [('mapping',m),('edges',edges),('diagnostics',diagnostics)]:
            (out/(name+'.json')).write_text(json.dumps(data,indent=2),encoding='utf-8')
        results[mode]=(m,edges)
    # Read human labels only after all inference has finished.
    groups=json.loads((ROOT/'reid_model/identity_groups.json').read_text())['groups']
    report={'baseline':evaluate(mapping,[],groups)}
    report.update({mode:evaluate(m,edges,groups) for mode,(m,edges) in results.items()})
    (a.output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({mode:{k:v for k,v in r.items() if k in ('known_components','known_pair_tp','known_pair_fp','known_pair_fn','unreviewed_edges')} for mode,r in report.items()},indent=2))

if __name__=='__main__':main()
