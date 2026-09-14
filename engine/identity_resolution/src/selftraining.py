"""Anchored self-training: train-window-only pseudo labels, validation-gated promotion."""
import csv
import json
import math
import pickle
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
from features import ROOT,backend,extract_features,load_model,sha
from trajectory_reassociation import run_reassociation


def json_write(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def load_manifest(path):
    path=Path(path);rows=list(csv.DictReader(path.open(encoding='utf-8')))
    for r in rows:
        for k in ['pid','frame','source_frame','algorithm_id','source_local_id','original_v3_id','kit','node']:r[k]=int(r[k])
        r['enrolment_gallery']=r['enrolment_gallery']=='True'
        r['image']=str((path.parent/r['image']).resolve())
    if len({(r['frame'],r['source_local_id']) for r in rows})!=len(rows):raise ValueError('Duplicate split observation')
    return rows


def validate_windows(windows,manifest):
    report=Path(manifest).parent/'dataset_report.json'
    allowed=json.loads(report.read_text())['windows']['train']
    if not windows or any(lo>hi or lo<allowed[0] or hi>allowed[1] for lo,hi in windows):
        raise ValueError('Self-training windows must be inside the frozen original training partition')


def validate_dataset(paths,rows):
    report=json.loads((Path(paths['manifest']).parent/'dataset_report.json').read_text())
    if sha(paths['audit_data'])!=report['audit_sha256'] or sha(paths['manifest'])!=report['manifest_sha256']:
        raise ValueError('Training labels belong to a different audit/manifest; rebuild and review the dataset')
    if {r['pid'] for r in rows if r['split']=='train'}!=set(range(9)):
        raise ValueError('Checkpoint requires the original nine anchored identities')
    for r in rows:
        if r['split'] not in report['windows']:raise ValueError('Unknown dataset split')
        lo,hi=report['windows'][r['split']]
        if not lo<=r['frame']<=hi or r['source_frame']!=r['frame']+report['source_offset']:
            raise ValueError('Frozen split/frame convention changed')
        if r['enrolment_gallery'] and r['split']!='train':raise ValueError('Held-out gallery leakage')


def select_pseudo(records,embeddings,mapping,seeds,prototypes,windows,settings):
    anchors=defaultdict(set);by_pid=defaultdict(list);seed_keys={(r['frame'],r['source_local_id']) for r in seeds}
    for r in seeds:anchors[mapping.get(r['algorithm_id'],r['algorithm_id'])].add(r['pid'])
    for r,feature in zip(records,embeddings):
        if r.get('training_allowed') is False or r.get('exclusion_reason'):continue
        if not any(a<=r['frame']<=b for a,b in windows):raise ValueError('Held-out row reached pseudo-label selection')
        if (r['frame'],r['source_local_id']) in seed_keys:continue
        component=mapping.get(r['algorithm_id'])
        labels=anchors.get(component,set())
        if len(labels)!=1:continue  # Unknown/new identities and conflicting anchors remain unknown.
        pid=next(iter(labels));scores=np.asarray(feature,dtype=np.float64)@prototypes.T
        order=np.argsort(-scores);margin=float(scores[order[0]]-scores[order[1]])
        if int(order[0])!=pid or scores[pid]<settings['min_similarity'] or margin<settings['teacher_margin']:continue
        item=dict(r,pid=pid,origin='pseudo',teacher_score=float(scores[pid]),teacher_margin=margin,
                  label_source='training_anchor_component_and_teacher_agreement')
        by_pid[pid].append(item)
    counts=defaultdict(int)
    for r in seeds:counts[r['pid']]+=1
    selected=[]
    for pid,rr in by_pid.items():
        cap=max(0,int(counts[pid]*settings['max_pseudo_per_seed']))
        selected.extend(sorted(rr,key=lambda r:(-r['teacher_score'],r['frame']))[:cap])
    return selected


def promotion_allowed(baseline,candidate,min_gain=.002):
    b=baseline['cross_algorithm_retrieval'];c=candidate['cross_algorithm_retrieval']
    if c['valid_queries']!=b['valid_queries'] or c['covered_identities']!=b['covered_identities']:return False
    if b['mAP_macro'] is None or c['mAP_macro'] is None:return False
    # Unknown safety metrics cannot silently pass a promotion gate.
    keys=['same_kit_FAR','TAR','track_rank1_macro']
    if any(baseline[k] is None or candidate[k] is None for k in keys):return False
    return (c['mAP_macro']>=b['mAP_macro']+min_gain and c['rank1_macro']>=b['rank1_macro']
            and candidate['same_kit_FAR']<=baseline['same_kit_FAR']+1e-12
            and candidate['TAR']>=baseline['TAR'] and candidate['track_rank1_macro']>=baseline['track_rank1_macro'])


class SeedBalancedBatches:
    def __init__(self,rows,k,steps,seed):
        self.rows=rows;self.k=k;self.steps=steps;self.seed=seed;self.epoch=0
        self.seeds=defaultdict(list);self.pseudo=defaultdict(list)
        for i,r in enumerate(rows):(self.seeds if r['origin']=='seed' else self.pseudo)[r['pid']].append(i)
        if set(self.seeds)!=set(range(9)) or min(map(len,self.seeds.values()))<k:raise ValueError('All nine anchored classes need enough original samples')
    def __len__(self):return self.steps
    def __iter__(self):
        rng=random.Random(self.seed+self.epoch)
        for _ in range(self.steps):
            batch=[]
            for pid,seed_ids in self.seeds.items():
                n=min(self.k//2,len(self.pseudo[pid]));batch+=rng.sample(self.pseudo[pid],n)
                pool=list(seed_ids);rng.shuffle(pool);chosen=[];bins=set()
                for i in pool:
                    b=self.rows[i]['frame']//60
                    if b not in bins:chosen.append(i);bins.add(b)
                    if len(chosen)==self.k-n:break
                if len(chosen)<self.k-n:chosen+=rng.sample([i for i in pool if i not in chosen],self.k-n-len(chosen))
                batch+=chosen
            rng.shuffle(batch);yield batch


def validation_report(model,rows,config,device):
    b=backend()
    import evaluation
    gallery=[r for r in rows if r['enrolment_gallery'] and r['split']=='train'];val=[r for r in rows if r['split']=='val']
    if not gallery or not val:raise ValueError('Frozen gallery and validation are required')
    return evaluation.evaluate(b.encode(model,val,config,device),b.encode(model,gallery,config,device),val,gallery,calibrate=True)


def train_round(weight,rows,pseudo,config,settings,out,device,smoke=False):
    import torch
    from torch.utils.data import DataLoader
    b=backend();b.seed_all(config['seed']);model=load_model(config,weight,device)
    baseline=validation_report(model,rows,config,device);json_write(out/'teacher_validation.json',baseline)
    seeds=[dict(r,origin='seed') for r in rows if r['split']=='train'];training=seeds+pseudo
    sampler=SeedBalancedBatches(training,settings['instances_per_identity'],settings['steps_per_epoch'],config['seed'])
    loader=DataLoader(b.Crops(training,config,True),batch_sampler=sampler,num_workers=0)
    head=[];body=[]
    for name,p in model.named_parameters():
        if p.requires_grad:(head if 'classifier' in name or 'bottleneck' in name else body).append(p)
    optimizer=torch.optim.AdamW([{'params':body,'lr':settings['lr_backbone']},{'params':head,'lr':settings['lr_head']}],weight_decay=.01)
    dtype=torch.bfloat16 if device=='cuda' and torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda' and dtype==torch.float16)
    kits_by_pid={r['pid']:r['kit'] for r in seeds};kits=torch.tensor([kits_by_pid[i] for i in range(9)],device=device)
    initial=model.base.patch_embed.proj.weight.detach().cpu().clone();history=[];best=None;best_score=-1;stale=0
    for epoch in range(1 if smoke else settings['epochs']):
        model.train();b.freeze_bn_statistics(model);sampler.epoch=epoch;losses=[]
        for step,(x,y) in enumerate(loader):
            x,y=x.to(device),y.to(device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=dtype,enabled=device=='cuda'):
                scores,features=model(x);loss=b.objective(scores,features,y,kits)
            if not torch.isfinite(loss):raise ValueError('Non-finite self-training loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1);scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
            if smoke:break
        candidate=validation_report(model,rows,config,device)
        score=candidate['cross_algorithm_retrieval']['mAP_macro']
        eligible=promotion_allowed(baseline,candidate,settings['min_validation_gain']) and not smoke
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),validation=candidate,promotion_eligible=eligible))
        if eligible and score>best_score:
            best_score=score;best=out/'candidate.pth';torch.save(model.state_dict(),best);stale=0
        else:stale+=1
        json_write(out/'history.json',history)
        print(json.dumps({'selftrain_epoch':epoch+1,'loss':history[-1]['loss'],'eligible':eligible}),flush=True)
        if stale>=settings['patience']:break
    delta=float(torch.norm(model.base.patch_embed.proj.weight.detach().cpu()-initial))
    if delta<=0:raise ValueError('Self-training did not update backbone parameters')
    result=dict(training_steps=sum(1 if smoke else settings['steps_per_epoch'] for _ in history),
                backbone_weight_delta=delta,promoted=best is not None,smoke=smoke,
                selected_weight=str(best) if best else str(weight),test_evaluated=False)
    json_write(out/'round_result.json',result);del model
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return result


def run_self_training(config,out,device='cpu',rounds=1,prepare_only=False,smoke=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'selftrain_result.json').exists():raise ValueError('Self-training run already exists')
    paths=config['paths'];settings=dict(config['selftrain']);windows=settings['train_windows']
    validate_windows(windows,paths['manifest'])
    rows=load_manifest(paths['manifest']);seeds=[r for r in rows if r['split']=='train']
    validate_dataset(paths,rows)
    model_config=json.loads(Path(paths['reid_config']).read_text());model_config['workers']=0;model_config['eval_batch']=config['reid']['batch_size']
    active=paths['reid_weights'];initial=sha(active);results=[]
    if rounds<1 or rounds>settings['max_rounds']:raise ValueError('Invalid self-training round limit')
    if min(settings[k] for k in ['epochs','patience','steps_per_epoch'])<1 or settings['instances_per_identity']<2:
        raise ValueError('Invalid training schedule or positive-pair batch size')
    if not 0<=settings['max_pseudo_per_seed']<=1:raise ValueError('Pseudo data must not dominate original supervision')
    for number in range(rounds):
        folder=out/f'round_{number+1:02d}';folder.mkdir(exist_ok=True)
        cache=extract_features(paths['audit_data'],paths['video'],active,model_config,folder/'features',config['quality'],
                               device,config['reid']['batch_size'],windows=windows)
        associated=run_reassociation(cache,folder/'association',config['reassociation'])
        with cache.open('rb') as f:audit=pickle.load(f)
        records=list(csv.DictReader((cache.parent/'crops.csv').open(encoding='utf-8')))
        for r in records:
            for k in ['node','algorithm_id','source_local_id','frame','kit']:r[k]=int(r[k])
        lookup={(n['id'],s[0]):s[1] for n in audit['nodes'] for s in n['samples']}
        vectors=np.array([lookup[(r['node'],r['frame'])] for r in records])
        teacher=load_model(model_config,active,device);b=backend()
        gallery=[r for r in seeds if r['enrolment_gallery']];features=b.encode(teacher,gallery,model_config,device)
        prototypes=np.array([features[[i for i,r in enumerate(gallery) if r['pid']==pid]].mean(0) for pid in range(9)])
        prototypes/=np.maximum(np.linalg.norm(prototypes,axis=1,keepdims=True),1e-9)
        del teacher
        import torch
        if torch.cuda.is_available():torch.cuda.empty_cache()
        pseudo=select_pseudo(records,vectors,associated['mapping'],seeds,prototypes,windows,settings)
        json_write(folder/'pseudo_manifest.json',pseudo)
        if prepare_only or not pseudo:
            results.append(dict(pseudo_crops=len(pseudo),training_steps=0,promoted=False,
                                reason='prepare_only' if prepare_only else 'no_eligible_novel_pseudo_crops'));break
        trained=train_round(active,rows,pseudo,model_config,settings,folder,device,smoke)
        results.append(dict(trained,pseudo_crops=len(pseudo)))
        if not trained['promoted']:break
        active=trained['selected_weight']
    if sha(paths['reid_weights'])!=initial:raise ValueError('Original checkpoint changed')
    result=dict(active_weight=str(active),original_checkpoint_unchanged=True,rounds=results,
                test_used_for_self_training=False,training_windows=windows)
    json_write(out/'selftrain_result.json',result);return result
