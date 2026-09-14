#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/football_reid_20260908
PY=/root/miniconda3/bin/python
if [ ! -x "$PY" ]; then PY=$(command -v python3); fi
"$PY" -m venv --system-site-packages .venv
.venv/bin/python -m pip install yacs pillow numpy pytest
mkdir -p weights
.venv/bin/python - <<'PY'
import hashlib,json,urllib.request
from pathlib import Path
import torch
path=Path('weights/jx_vit_base_p16_224-80ecf9dd.pth')
url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth'
if not path.exists():
    temporary=path.with_suffix('.download')
    urllib.request.urlretrieve(url,temporary)
    temporary.replace(path)
digest=hashlib.sha256(path.read_bytes()).hexdigest()
assert digest.startswith('80ecf9dd'),('pretrained checksum mismatch',digest)
state=torch.load(path,map_location='cpu',weights_only=True)
assert 'patch_embed.proj.weight' in state
Path('weights/provenance.json').write_text(json.dumps({'url':url,'sha256':digest,'keys':len(state)},indent=2))
print('pretrained_verified',digest,flush=True)
PY
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests -q
CUDA_VISIBLE_DEVICES='' .venv/bin/python check_environment.py
echo 'CPU_PREPARATION_COMPLETE_NO_GPU_TRAINING_STARTED'
