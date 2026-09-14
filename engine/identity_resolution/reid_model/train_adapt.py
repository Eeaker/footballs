"""TransReID full-backbone adaptation; only --run starts GPU training.

Test is evaluated only after validation checkpoint selection. Gallery is an
explicit train enrolment subset for the known-player deployment scenario.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,Sampler
from torchvision import transforms as T

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor/TransReID'))
from config import cfg as base_cfg
from model.make_model import make_model
from evaluation import evaluate


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def num_class(config,rows=None):
    if rows is not None:
        return len({r['pid'] for r in rows})
    return int(config.get('num_class',config.get('batch_identities',9)))


def load_rows(manifest=None):
    path=Path(manifest) if manifest else HERE/'data/manifest.csv'
    rows=list(csv.DictReader(path.open(encoding='utf-8')))
    for r in rows:
        for k in ['pid','frame','source_frame','algorithm_id','source_local_id','original_v3_id','kit','node','nearest_training_gap']:
            if k in r and r[k]!='':r[k]=int(r[k])
        r['enrolment_gallery']=str(r.get('enrolment_gallery','')).lower() in {'true','1','yes'}
    return rows


def model_for(config,pretrained=True):
    c=base_cfg.clone();c.MODEL.NAME='transformer';c.MODEL.TRANSFORMER_TYPE='vit_base_patch16_224_TransReID'
    c.MODEL.JPM=True;c.MODEL.SIE_CAMERA=False;c.MODEL.SIE_VIEW=False;c.MODEL.RE_ARRANGE=True
    c.MODEL.PRETRAIN_CHOICE='imagenet' if pretrained else 'none'
    c.MODEL.PRETRAIN_PATH=str((HERE/config['weight_file']).resolve()) if not Path(config['weight_file']).is_absolute() else config['weight_file']
    c.MODEL.STRIDE_SIZE=config['stride'];c.INPUT.SIZE_TRAIN=config['image_size'];c.TEST.NECK_FEAT='before'
    return make_model(c,num_class=num_class(config),camera_num=0,view_num=0)


class Crops(Dataset):
    def __init__(self,rows,config,train=False):
        self.rows=rows
        self.data_root=Path(config.get('data_root',HERE/'data'))
        self.train=train
        self.fragment=config.get('fragment_appearance') if train else None
        ops=[]
        if train and config.get('hard_augmentation') and not self.fragment:
            from augmentations import RandomNativeDegradation
            ops.append(RandomNativeDegradation(**config['hard_augmentation']))
        ops += [T.Resize(config['image_size'],interpolation=T.InterpolationMode.BICUBIC)]
        if train and not self.fragment:
            ops += [T.RandomAffine(degrees=5,translate=(.04,.04),scale=(.95,1.05)),T.ColorJitter(brightness=.15,contrast=.15)]
            ops += [T.RandomApply([T.GaussianBlur(kernel_size=3,sigma=tuple(config['blur_sigma']))],p=config['blur_probability'])]
        ops += [T.ToTensor(),T.Normalize([.5]*3,[.5]*3)]
        erasing=T.RandomErasing(p=config.get('erasing_probability',.3),scale=tuple(config.get('erasing_scale',(.02,.15))),ratio=(.3,3.3),value=0)
        if train and not self.fragment:ops += [erasing]
        self.transform=T.Compose(ops)
        self.erasing=None
        self.light_mild=None
        self.others={}
        if self.fragment:
            bypid=defaultdict(list)
            for r in rows:
                p=Path(r['image']);p=p if p.is_absolute() else self.data_root/p
                bypid[r['pid']].append(p)
            self.others={pid:[p for q,ps in bypid.items() if q!=pid for p in ps] for pid in bypid}
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        view=None
        if isinstance(i,tuple):i,view=i
        r=self.rows[i]
        image=Path(r['image'])
        path=image if image.is_absolute() else self.data_root/image
        with Image.open(path) as raw:im=raw.convert('RGB')
        if self.fragment:
            from augmentations import apply_fragment_view
            view=view or r.get('appearance_view') or 'original'
            other=None
            if view in ('light','heavy') and self.others.get(r['pid']):
                with Image.open(random.choice(self.others[r['pid']])) as overlay:other=overlay.convert('RGB')
            im=apply_fragment_view(im,view,other=other)
            return self.transform(im),r['pid']
        return self.transform(im),r['pid']


class BalancedTemporalSampler(Sampler):
    """Each identity equally often; prefer different two-second bins, no duplicates in batch."""
    def __init__(self,rows,k,steps,seed):
        self.rows=rows;self.k=k;self.steps=steps;self.seed=seed;self.epoch=0
        self.pools=defaultdict(list)
        for i,r in enumerate(rows):self.pools[r['pid']].append(i)
        assert len(self.pools)>=2 and min(map(len,self.pools.values()))>=k
    def __len__(self):return self.steps
    def __iter__(self):
        rng=random.Random(self.seed+self.epoch)
        for _ in range(self.steps):
            batch=[]
            for pid in sorted(self.pools):
                bins=defaultdict(list)
                for i in self.pools[pid]:bins[self.rows[i]['frame']//60].append(i)
                keys=list(bins);rng.shuffle(keys)
                selected=[rng.choice(bins[key]) for key in keys[:self.k]]
                if len(selected)<self.k:selected+=rng.sample([i for i in self.pools[pid] if i not in selected],self.k-len(selected))
                batch+=selected
            rng.shuffle(batch);yield batch


class ViewBalancedSampler(Sampler):
    """Same temporal sampling, but each identity keeps a majority of unprocessed originals."""
    def __init__(self,rows,k,steps,seed):
        self.rows=rows;self.k=k;self.steps=steps;self.seed=seed;self.epoch=0
        self.pools=defaultdict(list)
        for i,r in enumerate(rows):self.pools[r['pid']].append(i)
        assert len(self.pools)>=2 and min(map(len,self.pools.values()))>=k
    def __len__(self):return self.steps
    def __iter__(self):
        from augmentations import view_schedule
        rng=random.Random(self.seed+self.epoch)
        for step in range(self.steps):
            batch=[]
            for pid in sorted(self.pools):
                bins=defaultdict(list)
                for i in self.pools[pid]:bins[self.rows[i]['frame']//60].append(i)
                keys=sorted(bins)
                if len(keys)>=self.k:
                    pick=np.unique(np.linspace(0,len(keys)-1,self.k).astype(int)).tolist()
                    while len(pick)<self.k:pick.append(rng.randrange(len(keys)))
                    keys=[keys[j] for j in pick]
                else:
                    keys=(keys*self.k)[:self.k]
                views=view_schedule(self.k,step)
                for key,view in zip(keys,views):
                    batch.append((rng.choice(bins[key]),view))
            rng.shuffle(batch);yield batch


def freeze_bn_statistics(model):
    for module in model.modules():
        if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()


def objective(scores,features,labels,kits=None):
    ce=sum(F.cross_entropy(x,labels,label_smoothing=.05) for x in scores)/len(scores)
    # Random positive/negative per anchor is less noise-sensitive than early batch-hard mining.
    f=F.normalize(features[0].float(),dim=1);distance=1-f@f.T
    losses=[]
    for i in range(len(labels)):
        pos=torch.where((labels==labels[i]) & (torch.arange(len(labels),device=labels.device)!=i))[0]
        neg=torch.where(labels!=labels[i])[0]
        if kits is not None and int(kits[labels[i]])>=0:
            same_kit=neg[kits[labels[neg]]==kits[labels[i]]]
            if len(same_kit):neg=same_kit
        if len(pos) and len(neg):
            p=pos[torch.randint(len(pos),(1,),device=labels.device)]
            n=neg[torch.randint(len(neg),(1,),device=labels.device)]
            losses.append(F.relu(distance[i,p]-distance[i,n]+.2).mean())
    metric=torch.stack(losses).mean() if losses else ce*0
    return ce+.5*metric


@torch.no_grad()
def encode(model,rows,config,device):
    model.eval();features=[]
    loader=DataLoader(Crops(rows,config),batch_size=config['eval_batch'],num_workers=config['workers'],pin_memory=device=='cuda')
    amp=device=='cuda' and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    for x,_ in loader:
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
            feat=model(x.to(device)).float()
        features.append(F.normalize(feat,dim=1).cpu().numpy())
    return np.concatenate(features)


def validation(model,rows,config,device,split='val',threshold=None,calibrate=True):
    gallery=[r for r in rows if r['enrolment_gallery']]
    query=[r for r in rows if r['split']==split]
    return evaluate(encode(model,query,config,device),encode(model,gallery,config,device),query,gallery,threshold,calibrate)


def autotune(model,config,dtype):
    """Forward/backward only, no optimizer update; preserve model weights and RNG externally."""
    decisions=[];chosen=None
    for k in config['autotune_k']:
        x=y=score=feat=loss=None
        try:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();model.train();freeze_bn_statistics(model)
            n=num_class(config)
            x=torch.zeros(n*k,3,*config['image_size'],device='cuda');y=torch.arange(n,device='cuda').repeat_interleave(k)
            with torch.autocast('cuda',dtype=dtype):score,feat=model(x);loss=objective(score,feat,y)
            loss.backward();torch.cuda.synchronize()
            peak=torch.cuda.max_memory_allocated();total=torch.cuda.get_device_properties(0).total_memory
            # AdamW moments + master fp32 parameters reserve; conservatively keep 10% free.
            optimizer_reserve=sum(p.numel()*12 for p in model.parameters() if p.requires_grad)
            acceptable=peak+optimizer_reserve < config['max_vram_fraction']*total
            decisions.append(dict(k=k,batch=n*k,peak_bytes=peak,optimizer_reserve=optimizer_reserve,accepted=acceptable))
            if acceptable:chosen=k
        except torch.cuda.OutOfMemoryError:
            decisions.append(dict(k=k,accepted=False,reason='out_of_memory'))
        finally:
            x=y=score=feat=loss=None
            model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    if chosen is None:raise RuntimeError('No configured batch fits safely; reduce P/K explicitly, do not fake duplicate images.')
    return chosen,decisions


def run(config):
    if not torch.cuda.is_available():raise RuntimeError('GPU is off. Preparation can run on CPU; training requires user to enable GPU first.')
    if not torch.cuda.is_bf16_supported():raise RuntimeError('Full-parameter ReID fine-tuning requires an Ampere-or-newer CUDA GPU with BF16 support.')
    out=Path(config.get('run_dir',HERE/'runs/full_finetune'))
    out.mkdir(parents=True,exist_ok=True)
    if (out/'started.json').exists():raise RuntimeError('Run already exists; preserve results and use a new run directory.')
    manifest=Path(config.get('manifest',HERE/'data/manifest.csv'))
    rows=load_rows(manifest)
    if config.get('fragment_protocol'):
        protocol=json.loads(Path(config['fragment_protocol']).read_text(encoding='utf-8'))
        if hashlib.sha256(manifest.read_bytes()).hexdigest()!=protocol['manifest_sha256']:
            raise ValueError('Frozen fragment manifest changed')
        partitions={s:{r['algorithm_id'] for r in rows if r['split']==s} for s in ['train','val','test']}
        if partitions['train']&partitions['test'] or partitions['val']&partitions['test']:
            raise ValueError('Test fragments must not reuse training algorithm IDs')
        if protocol.get('val_from_unused_train_track_frames'):
            if partitions['train']!=partitions['val']:
                raise ValueError('Val must use unused frames from every training trajectory')
            trained={(r['source_local_id'],r['frame']) for r in rows if r['split']=='train'}
            if any((r['source_local_id'],r['frame']) in trained for r in rows if r['split']!='train'):
                raise ValueError('Val/test reused a training frame')
        elif partitions['train']&partitions['val'] or partitions['train']&partitions['test'] or partitions['val']&partitions['test']:
            raise ValueError('An algorithm ID crosses data partitions')
        for r in rows:
            if r['split']=='train' and r.get('label_source') not in {None,'','algorithm_seed_only'}:
                raise ValueError('Human correspondence entered training')
            if protocol.get('val_from_unused_train_track_frames') and r['split']=='val' and r.get('label_source') not in {None,'','algorithm_seed_only'}:
                raise ValueError('Human correspondence entered validation')
    train_only=bool(config.get('train_only')) or not any(r['split']=='val' for r in rows)
    n=num_class(config,rows);config=dict(config,num_class=n)
    seed_all(config['seed']);model=model_for(config).cuda()
    # Full-parameter fine-tuning: undo upstream classifier/bottleneck freezes.
    for parameter in model.parameters():parameter.requires_grad_(True)
    initial_patch=model.base.patch_embed.proj.weight.detach().cpu().clone()
    dtype=torch.bfloat16
    k,decisions=autotune(model,config,dtype);seed_all(config['seed'])
    write(out/'batch_probe.json',decisions)
    torch.save(model.state_dict(),out/'frozen_initial.pth')
    frozen_val=None
    if not train_only:
        frozen_val=validation(model,rows,config,'cuda');write(out/'frozen_validation.json',frozen_val)
    write(out/'started.json',dict(config=config,batch=n*k,dtype=str(dtype),train_only=train_only,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),num_class=n,
        parameter_count=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    head=[];backbone=[]
    for name,p in model.named_parameters():
        if p.requires_grad:(head if 'classifier' in name or 'bottleneck' in name else backbone).append(p)
    optimizer=torch.optim.AdamW([{'params':backbone,'lr':config['backbone_lr']},{'params':head,'lr':config['head_lr']}],weight_decay=config['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    from augmentations import expand_train_views
    train=expand_train_views(rows,config)
    kit_by_pid={r['pid']:r['kit'] for r in train}
    kits=torch.tensor([kit_by_pid.get(i,-1) for i in range(n)],device='cuda')
    sampler_cls=ViewBalancedSampler if config.get('fragment_appearance') else BalancedTemporalSampler
    sampler=sampler_cls(train,k,config['steps_per_epoch'],config['seed'])
    loader=DataLoader(Crops(train,config,True),batch_sampler=sampler,num_workers=config['workers'],pin_memory=True,persistent_workers=config['workers']>0)
    best=-1e9;stale=0;history=[]
    for epoch in range(config['epochs']):
        sampler.epoch=epoch;model.train();freeze_bn_statistics(model)
        fraction=min(1,(epoch+1)/config['warmup_epochs'])*(.1+.9*.5*(1+math.cos(math.pi*epoch/config['epochs'])))
        for group,base in zip(optimizer.param_groups,[config['backbone_lr'],config['head_lr']]):group['lr']=base*fraction
        losses=[]
        for x,y in loader:
            x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=dtype):scores,features=model(x);loss=objective(scores,features,y,kits)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
        mean_loss=float(np.mean(losses))
        record=dict(epoch=epoch+1,loss=mean_loss)
        if train_only:
            score=-mean_loss
            record['selection_metric']='train_loss'
        else:
            val=validation(model,rows,config,'cuda')
            cross=val['cross_algorithm_retrieval']['mAP_macro']
            if cross is not None:
                score=cross;record['selection_metric']='val_cross_algorithm_mAP'
            else:
                score=val['image_retrieval']['mAP_macro'];record['selection_metric']='val_image_retrieval_mAP'
            record['validation']=val
        record['selection_score']=score
        record['test_used_for_selection']=False
        history.append(record);write(out/'history.json',history);print(json.dumps({'epoch':epoch+1,'loss':mean_loss,'selection':score}),flush=True)
        if score>best+1e-5:
            best=score;stale=0;torch.save(model.state_dict(),out/'best.pth')
            write(out/'best_checkpoint.json',record)
            if not train_only:write(out/'best_validation.json',record['validation'])
        else:stale+=1
        if epoch>=4 and stale>=config['patience']:break
    if train_only:
        write(out/'final_comparison.json',dict(train_only=True,best_selection=best,epochs=len(history),
            caveat='No held-out val/test; checkpoint selected by training loss.'))
        return
    model.load_state_dict(torch.load(out/'best.pth',map_location='cpu',weights_only=True))
    backbone_delta=float(torch.norm(model.base.patch_embed.proj.weight.detach().cpu()-initial_patch))
    best_val=json.loads((out/'best_validation.json').read_text(encoding='utf-8'))
    adapted=validation(model,rows,config,'cuda','test',best_val['threshold_from_validation'],False)
    model.load_state_dict(torch.load(out/'frozen_initial.pth',map_location='cpu',weights_only=True))
    frozen=validation(model,rows,config,'cuda','test',frozen_val['threshold_from_validation'],False)
    write(out/'final_comparison.json',dict(frozen=frozen,adapted=adapted,
        backbone_weight_delta=backbone_delta,training_labels='algorithm_seeds_only' if config.get('fragment_protocol') else 'legacy',
        test_used_for_selection=False,production_approved=False,
        caveat='Known nine players, one-minute development clip. Review cross-ID accuracy AND false acceptance; no unseen-match claim.'))


def preflight(config,pretrained=False):
    torch.set_num_threads(2);seed_all(config['seed']);rows=load_rows()
    model=model_for(config,pretrained);model.train();freeze_bn_statistics(model)
    x,y=next(iter(DataLoader(Crops([r for r in rows if r['split']=='train'][:2],config,True),batch_size=2)))
    scores,features=model(x);loss=objective(scores,features,y);loss.backward()
    assert model.base.patch_embed.proj.weight.grad is not None
    assert model.b1[0].attn.qkv.weight.grad is not None
    assert model.b2[0].attn.qkv.weight.grad is not None
    model.eval()
    with torch.no_grad():embedding=model(x)
    assert embedding.shape==(2,3840) and torch.isfinite(embedding).all()
    result=dict(cpu_forward_backward_passed=True,embedding_dimension=3840,
                total_parameters=sum(p.numel() for p in model.parameters()),pretrained_loaded=pretrained,training_run=False)
    write(HERE/('preflight_pretrained.json' if pretrained else 'preflight_cpu.json'),result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--preflight',action='store_true');p.add_argument('--pretrained',action='store_true')
    p.add_argument('--config',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--run-dir',type=Path)
    p.add_argument('--train-only',action='store_true');a=p.parse_args()
    config=json.loads((a.config or HERE/'config.json').read_text(encoding='utf-8'))
    if a.manifest:config['manifest']=str(a.manifest.resolve())
    if a.run_dir:config['run_dir']=str(a.run_dir.resolve())
    if a.train_only:config['train_only']=True
    if a.run:run(config)
    elif a.preflight:preflight(config,a.pretrained)
    else:p.print_help()
