"""Tracklet jersey-number vote. Positive evidence only; never a merge veto."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_number(text):
    digits = ''.join(ch for ch in str(text) if ch.isdigit())
    if not digits or len(digits) > 2:
        return None
    n = int(digits)
    return n if 1 <= n <= 99 else None


def vote_track(hits, min_conf=0.85, min_count=2, min_share=0.5):
    """Majority number among high-confidence OCR hits. None if weak or split."""
    kept = []
    for hit in hits:
        n = hit.get('number') if isinstance(hit, dict) else hit[0]
        conf = hit.get('conf') if isinstance(hit, dict) else hit[1]
        if n is None or conf is None or conf < min_conf:
            continue
        n = int(n)
        if 1 <= n <= 99:
            kept.append(n)
    if not kept:
        return None, dict(hits=0, counts={}, winner=None, share=0.0, reason='no_confident_hits')
    counts = Counter(kept)
    winner, n = counts.most_common(1)[0]
    share = n / len(kept)
    if n < min_count:
        return None, dict(hits=len(kept), counts=dict(counts), winner=int(winner), share=share, reason='count')
    if share < min_share:
        return None, dict(hits=len(kept), counts=dict(counts), winner=int(winner), share=share, reason='share')
    return int(winner), dict(hits=len(kept), counts=dict(counts), winner=int(winner), share=share, reason='ok')


def load_votes(path):
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    votes = raw.get('votes', raw)
    out = {}
    for k, v in votes.items():
        n = v.get('number') if isinstance(v, dict) else v
        if n is None:
            continue
        out[int(k)] = int(n)
    return out


def _torso(bgr):
    h, w = bgr.shape[:2]
    y0, y1 = int(h * 0.12), max(int(h * 0.12) + 1, int(h * 0.62))
    x0, x1 = int(w * 0.18), max(int(w * 0.18) + 1, int(w * 0.82))
    return bgr[y0:y1, x0:x1]


def _clahe(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def variants(bgr):
    torso = _torso(bgr)
    up = cv2.resize(torso, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC) if min(torso.shape[:2]) >= 8 else torso
    return [bgr, torso, _clahe(torso), up]


def _best_number(items):
    best = None
    for item in items:
        text, conf = item[1], float(item[2])
        n = parse_number(text)
        if n is None:
            continue
        if best is None or conf > best['conf']:
            best = dict(number=int(n), conf=conf, text=str(text))
    return best


def sample_rows(rows, max_crops=8):
    if len(rows) <= max_crops:
        return rows
    idx = np.unique(np.linspace(0, len(rows) - 1, max_crops).astype(int))
    return [rows[i] for i in idx]


def ocr_crops(crops_csv, reader=None, max_crops=8, min_conf=0.85, min_count=2, min_share=0.5):
    """OCR sampled association crops. Votes are algorithm_id -> number or None."""
    if reader is None:
        import easyocr
        try:
            import torch
            gpu = bool(torch.cuda.is_available())
        except Exception:
            gpu = False
        reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)
    crops_csv = Path(crops_csv)
    crop_dir = crops_csv.parent / 'crops'
    by_id = defaultdict(list)
    with crops_csv.open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            by_id[int(row['algorithm_id'])].append(row)
    hits = {}
    votes = {}
    details = {}
    decoded = 0
    for oid, rows in sorted(by_id.items()):
        chosen = sample_rows(sorted(rows, key=lambda r: int(r['frame'])), max_crops)
        print(f'jersey_ocr id={oid} crops={len(chosen)}/{len(rows)}', flush=True)
        track_hits = []
        raw = []
        for row in chosen:
            path = crop_dir / Path(row['image']).name
            data = np.fromfile(str(path), dtype=np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
            if bgr is None:
                continue
            decoded += 1
            for vi, im in enumerate(variants(bgr)):
                items = reader.readtext(im, allowlist='0123456789', detail=1, paragraph=False)
                hit = _best_number(items)
                rec = dict(algorithm_id=oid, frame=int(row['frame']), variant=vi, image=str(path), hit=hit)
                raw.append(rec)
                if hit:
                    track_hits.append(hit)
        number, info = vote_track(track_hits, min_conf=min_conf, min_count=min_count, min_share=min_share)
        hits[str(oid)] = raw
        details[str(oid)] = dict(info, number=number, crops=len(chosen))
        votes[str(oid)] = dict(number=number, **info, crops=len(chosen))
    if decoded == 0:
        raise ValueError(f'No jersey crops decoded from {crops_csv}')
    return dict(
        min_conf=min_conf, min_count=min_count, min_share=min_share, max_crops=max_crops,
        crops_csv=str(crops_csv.resolve()),
        votes=votes, details=details, hits=hits,
        voted={k: v['number'] for k, v in votes.items() if v.get('number') is not None},
        decoded=decoded,
    )
