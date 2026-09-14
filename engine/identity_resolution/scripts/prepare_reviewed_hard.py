"""Version explicit, per-observation reviewed hard crops within the frozen train split."""
import argparse,json,pickle,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from scripts.run_pipeline import load_config
from selftraining import load_manifest,validate_dataset
from features import sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('review','crop_records','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();c=load_config(ROOT/'configs/pipeline_config.json');paths=c['paths']
    rows=load_manifest(paths['manifest']);validate_dataset(paths,rows)
    review=json.loads(a.review.read_text(encoding='utf-8'));records=json.loads(a.crop_records.read_text(encoding='utf-8'))
    if review['audit_sha256']!=sha(paths['audit_data']):raise ValueError('Review audit mismatch')
    if review['crop_records_sha256']!=sha(a.crop_records):raise ValueError('Review image list mismatch')
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('Use a fresh output')
    a.output.mkdir(parents=True,exist_ok=True)
    audit=pickle.loads(Path(paths['audit_data']).read_bytes());seeds=[r for r in rows if r['split']=='train']
    seedkeys={(r['source_local_id'],r['frame']) for r in seeds};supplement=[]
    for item in review['selected_observations']:
        oid,frame=item['algorithm_id'],item['frame']
        if not any(lo<=frame<=hi for lo,hi in c['selftrain']['train_windows']):raise ValueError('Held-out reviewed crop cannot enter training')
        matches=[n for n in audit['nodes'] if n['output_id']==oid and any(r[0]==frame for r in n['rows'])]
        if len(matches)!=1 or matches[0]['quarantine']:raise ValueError('Ambiguous or quarantined source')
        n=matches[0]
        if (n['source_local'],frame) in seedkeys:raise ValueError('Duplicate existing training observation')
        record=next(r for r in records if r['id']==oid and r['frame']==frame)
        image=a.crop_records.parent/record['image']
        prototype=next(r for r in seeds if r['identity']==str(item['identity_representative']))
        supplement.append(dict(pid=prototype['pid'],identity=prototype['identity'],kit=prototype['kit'],image=str(image.resolve()),sha256=sha(image),frame=frame,source_frame=frame+3150,
            algorithm_id=oid,source_local_id=n['source_local'],node=n['id'],enrolment_gallery=False,
            origin='human_reviewed_hard',training_allowed=True,label_source='user_identity_plus_per_frame_quality_review',
            quality_reason=item['reason'],split='train'))
    (a.output/'supplement.json').write_text(json.dumps(supplement,ensure_ascii=False,indent=2),encoding='utf-8')
    report=dict(seed_images=len(seeds),new_images=len(supplement),identities=sorted({r['identity'] for r in supplement}),
                sampling='equal identity count; supplemental images at most half of each identity batch',
                original_manifest_sha256=sha(paths['manifest']),review_sha256=sha(a.review),test_images_added=0)
    (a.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(report)

if __name__=='__main__':main()
