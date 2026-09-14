"""Low-memory server preparation check. Does not instantiate or train a model."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import torch
import torchvision
import yacs

HERE=Path(__file__).resolve().parent
config=json.loads((HERE/'config.json').read_text())
weight=HERE/config['weight_file']
h=hashlib.sha256()
with weight.open('rb') as f:
    for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
assert h.hexdigest().startswith('80ecf9dd')
state=torch.load(weight,map_location='cpu',weights_only=True)
assert state['patch_embed.proj.weight'].shape==(768,3,16,16)
rows=list(csv.DictReader((HERE/'data/manifest.csv').open(encoding='utf-8')))
for r in rows:assert hashlib.sha256((HERE/'data'/r['image']).read_bytes()).hexdigest()==r['sha256']
result=dict(python=sys.version,torch=torch.__version__,torchvision=torchvision.__version__,
            gpu_available=torch.cuda.is_available(),weight_sha256=h.hexdigest(),images_verified=len(rows),
            manifest_sha256=hashlib.sha256((HERE/'data/manifest.csv').read_bytes()).hexdigest(),
            training_run=False,full_model_backward_checked='local CPU preflight; GPU probe deferred until enabled')
(HERE/'server_preparation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
