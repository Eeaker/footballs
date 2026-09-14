"""Explicit human-reviewed identity corrections; never reported as algorithm accuracy."""
import argparse,copy,json,pickle,sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trajectory_reassociation import write_result
from features import sha


def apply_review(audit,mapping,review):
    audit=copy.deepcopy(audit);mapping=dict(mapping)
    present={n['output_id'] for n in audit['nodes'] if not n['quarantine']}
    for entry in review['confirmed_pairs']:
        source,target=entry['source_id'],entry['identity_representative']
        if source not in present or target not in present:raise ValueError('Review IDs do not match input audit')
        before,after=mapping[source],mapping[target]
        mapping={k:after if v==before else v for k,v in mapping.items()}
    for entry in review['invalid_detections']:
        oid=entry['source_id']
        if oid not in present:raise ValueError('Invalid detection ID absent')
        if any(k!=oid and v==mapping[oid] for k,v in mapping.items()):
            raise ValueError('Invalid ID already merged; undo that association explicitly first')
        mapping.pop(oid)
        for n in audit['nodes']:
            if n['output_id']==oid:
                n.update(quarantine=True,exclusion_reason=entry['reason'],samples=[],positions={})
    return audit,mapping


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('audit','mapping','review','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('Use a fresh output directory')
    audit=pickle.loads(a.audit.read_bytes());mapping={int(k):v for k,v in json.loads(a.mapping.read_text()).items()}
    review=json.loads(a.review.read_text(encoding='utf-8'))
    audit,mapping=apply_review(audit,mapping,review)
    a.output.mkdir(parents=True,exist_ok=True);rows=write_result(audit,mapping,a.output)
    (a.output/'audit_data.pkl').write_bytes(pickle.dumps(audit))
    (a.output/'mapping.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    report=dict(source='algorithm output plus explicit human corrections',algorithm_only=False,
                input_audit_sha256=sha(a.audit),review_sha256=sha(a.review),
                normal_groups=len(set(mapping.values())),rows=len(rows),
                invalid_detection_rows=sum(len(n['rows']) for n in audit['nodes'] if n.get('exclusion_reason')),
                automatic_training_labels=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
