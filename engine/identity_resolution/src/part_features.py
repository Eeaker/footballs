"""Frozen Sports-OSNet + 6-stripe part pooling. Never trained on this clip's trunks."""
from __future__ import annotations

import csv
import importlib.util
import pickle
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
OSNET_PY = ROOT.parent / '当前方案源码' / 'sports_osnet' / 'reid' / 'torchreid' / 'models' / 'osnet.py'
SPORTS_WEIGHT = ROOT.parent / '当前方案源码' / 'sports_osnet' / 'checkpoints' / 'sports_model.pth.tar-60'
STRIPE_PARTS = 6


def _load_osnet(weight, device):
    spec = importlib.util.spec_from_file_location('sports_osnet_model', OSNET_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.osnet_x1_0(num_classes=1, pretrained=False)
    ckpt = torch.load(weight, map_location='cpu', weights_only=False)
    state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    cleaned = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    current = model.state_dict()
    cleaned = {k: v for k, v in cleaned.items() if k in current and current[k].shape == v.shape}
    model.load_state_dict(cleaned, strict=False)
    return model.to(device).eval()


def _read_bgr(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def part_pool(maps):
    g = F.adaptive_avg_pool2d(maps, 1).flatten(1)
    h = maps.size(2)
    parts = []
    for i in range(STRIPE_PARTS):
        sl = maps[:, :, i * h // STRIPE_PARTS:(i + 1) * h // STRIPE_PARTS, :]
        parts.append(F.adaptive_avg_pool2d(sl, 1).flatten(1))
    feat = torch.cat([g] + parts, 1)
    return F.normalize(feat, dim=1)


def encode_images(paths, device='cuda', batch_size=16, weight=None):
    weight = Path(weight or SPORTS_WEIGHT)
    model = _load_osnet(weight, device)
    transform = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    out = np.zeros((len(paths), 512 * (1 + STRIPE_PARTS)), np.float32)
    pending = []
    def flush():
        if not pending:
            return
        with torch.inference_mode():
            batch = torch.stack([p[0] for p in pending]).to(device)
            maps = model.featuremaps(batch)
            encoded = part_pool(maps).cpu().numpy()
        for feat, (_, i) in zip(encoded, pending):
            out[i] = feat
        pending.clear()
    try:
        for i, path in enumerate(paths):
            bgr = _read_bgr(path)
            if bgr is None:
                raise ValueError(f'Cannot decode {path}')
            rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            pending.append((transform(rgb), i))
            if len(pending) >= batch_size:
                flush()
        flush()
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def replace_audit_features(audit_path, crops_csv, out_pkl, device='cuda', weight=None):
    audit = pickle.loads(Path(audit_path).read_bytes())
    rows = list(csv.DictReader(Path(crops_csv).open(encoding='utf-8')))
    crop_dir = Path(crops_csv).parent / 'crops'
    paths = [crop_dir / Path(r['image']).name for r in rows]
    feats = encode_images(paths, device=device, weight=weight)
    by_id_frame = {}
    for r, feat in zip(rows, feats):
        by_id_frame[int(r['algorithm_id']), int(r['frame'])] = feat
    dim = feats.shape[1]
    n = 0
    for node in audit['nodes']:
        oid = int(node['output_id'])
        samples = []
        for s in node.get('samples') or []:
            key = (oid, int(s[0]))
            if key not in by_id_frame:
                continue
            samples.append((s[0], by_id_frame[key], s[2]))
            n += 1
        node['samples'] = samples
    from features import sha
    weight = Path(weight or SPORTS_WEIGHT)
    info = dict(audit.get('feature_metadata') or {})
    fingerprint = dict(info.get('fingerprint') or {})
    fingerprint.update(checkpoint=sha(weight), encoder='frozen_sports_osnet_pcb6')
    info.update(dimension=int(dim), encoder='frozen_sports_osnet_pcb6', crops=n,
                weight=str(weight.resolve()), fingerprint=fingerprint)
    audit['feature_metadata'] = info
    out_pkl = Path(out_pkl)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open('wb') as f:
        pickle.dump(audit, f)
    return out_pkl, dim, n


def gallery_prototypes(manifest, device='cuda', weight=None):
    rows = list(csv.DictReader(Path(manifest).open(encoding='utf-8')))
    gallery = [r for r in rows if str(r.get('enrolment_gallery', '')).lower() in {'true', '1', 'yes'}]
    if not gallery:
        raise ValueError('No enrolment gallery')
    paths = [Path(r['image']) for r in gallery]
    feats = encode_images(paths, device=device, weight=weight)
    pids = sorted({int(r['pid']) for r in gallery})
    proto = np.array([feats[[i for i, r in enumerate(gallery) if int(r['pid']) == pid]].mean(0) for pid in pids])
    proto /= np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), 1e-9)
    return proto
