"""Freeze trunk-only training first; human mappings are read only for held-out scoring."""
import csv,json,hashlib,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.csv').exists():raise ValueError('Do not overwrite frozen dataset')
    cache=ROOT.parent/'实验结论/二阶段完整优化_20260909/最终完整运行/features/crops.csv'
    records=list(csv.DictReader(cache.open(encoding='utf-8')))
    # Explicit selected trunks; each is a class. No identity correspondence table.
    trunks=[129,31,120,158,124,150,161,142,307]
    def convert(r,pid,split):
        image=Path(r['image']);assert image.exists()
        return dict(pid=pid,identity=str(trunks[pid]),algorithm_id=int(r['algorithm_id']),
            original_v3_id=int(r['algorithm_id']),source_local_id=int(r['source_local_id']),node=int(r['node']),
            frame=int(r['frame']),source_frame=int(r['frame'])+3150,split=split,kit=int(r['kit']),
            image=str(image.resolve()),sha256=digest(image),enrolment_gallery=False)
    train=[convert(r,trunks.index(int(r['algorithm_id'])),'train') for r in records if int(r['algorithm_id']) in trunks]
    for pid in range(9):
        rr=sorted([r for r in train if r['pid']==pid],key=lambda r:r['frame']);assert len(rr)>=8
        for index in np.linspace(0,len(rr)-1,min(12,len(rr))).astype(int):rr[index]['enrolment_gallery']=True
    write_csv(out/'train_manifest.csv',train)
    frozen=digest(out/'train_manifest.csv')
    # Scoring metadata only, after training data was frozen.
    groups=json.loads((ROOT/'reid_model/identity_groups.json').read_text(encoding='utf-8'))['groups']
    queries=[];partitions={}
    trainkeys={(r['source_local_id'],r['frame']) for r in train};trainhash={r['sha256'] for r in train}
    for pid,trunk in enumerate(trunks):
        ids=next(v for v in groups.values() if trunk in v)
        candidates=defaultdict(list)
        for r in records:
            oid=int(r['algorithm_id'])
            if oid in ids and oid not in trunks:
                item=convert(r,pid,'test')
                if (item['source_local_id'],item['frame']) not in trainkeys and item['sha256'] not in trainhash:candidates[oid].append(item)
        ordered=sorted(candidates,key=lambda i:min(r['frame'] for r in candidates[i]))
        assert ordered,('No held-out fragment',trunk)
        # Whole IDs; last fragment reserved for test, remaining fragments for validation.
        test_id=ordered[-1];partitions[trunk]=dict(val=ordered[:-1],test=[test_id])
        for oid in ordered:
            for r in candidates[oid]:r['split']='test' if oid==test_id else 'val';queries.append(r)
    assert digest(out/'train_manifest.csv')==frozen
    allrows=train+queries;write_csv(out/'manifest.csv',allrows)
    sets={s:{r['algorithm_id'] for r in allrows if r['split']==s} for s in ('train','val','test')}
    assert not sets['train']&sets['val'] and not sets['train']&sets['test'] and not sets['val']&sets['test']
    report=dict(training_ids=trunks,training_labels='one class per selected algorithm trunk; no manual cross-ID merge',
        keeper_trunk=307,keeper_selection='single longest observed fragment among previously identified keeper candidates; counterparts excluded',
        counts={s:sum(r['split']==s for r in allrows) for s in sets},
        partitions=partitions,train_manifest_sha256=frozen,full_manifest_sha256=digest(out/'manifest.csv'),
        identity_coverage={s:len({r['pid'] for r in allrows if r['split']==s}) for s in sets},
        source_crop_manifest_sha256=digest(cache),scope='same-video cross-fragment development test; not unseen-match evaluation')
    (out/'dataset_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    config=json.loads((ROOT/'reid_model/config.json').read_text())
    config.update(manifest=str((out/'manifest.csv').resolve()),data_root=str(out.resolve()),run_dir=str((out.parent/'training').resolve()),
        num_class=9,workers=0,eval_batch=8,autotune_k=[2],epochs=30,patience=6,
        weight_file=str((ROOT/'reid_model/weights/jx_vit_base_p16_224-80ecf9dd.pth').resolve()),
        hard_augmentation=dict(motion_probability=.2,downsample_probability=.1))
    (out/'training_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
