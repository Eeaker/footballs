"""Build a train-only crop set from the longest non-quarantine identity tracks."""
import argparse
import csv
import hashlib
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from automatic_seed_selection import select_seeds


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def quality_ok(row, overlap, width, height):
    f, local, x, y, w, h, conf = row
    if conf < .55 or overlap > .15 or w < 16 or h < 40:
        return False
    if x < 2 or y < 2 or x + w > width - 2 or y + h > height - 2:
        return False
    return True


def overlap(row, others):
    _, local, x, y, w, h, *_ = row
    best = 0.
    for other in others:
        if other[1] == local:
            continue
        _, _, xx, yy, ww, hh, *_ = other
        area = max(0, min(x + w, xx + ww) - max(x, xx)) * max(0, min(y + h, yy + hh) - max(y, yy))
        best = max(best, area / max(w * h, 1e-9))
    return best


def prepare(audit_path, video, out, backbone=10, min_crops=8, sample_every=15):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.csv'
    if manifest_path.exists():
        return manifest_path
    audit = pickle.load(Path(audit_path).open('rb'))
    by_id = defaultdict(list)
    by_frame = defaultdict(list)
    kit_votes = defaultdict(Counter)
    nodes_by_id = defaultdict(list)
    skip = {n['output_id'] for n in audit['nodes'] if n.get('quarantine')}
    for node in audit['nodes']:
        for row in node['rows']:
            by_frame[row[0]].append(row)  # Quarantine also counts as an occluding box.
        if node['output_id'] in skip:
            continue
        nodes_by_id[node['output_id']].append(node)
        if node.get('kit', -1) >= 0:
            kit_votes[node['output_id']][node['kit']] += len(node['rows'])
        for row in node['rows']:
            by_id[node['output_id']].append(row)
    selection=select_seeds(audit,max_outfield=min(backbone,8))
    ranked=list(selection['outfield_ids'])
    keeper=selection['keeper']
    if keeper:ranked.append(keeper['id'])
    dump(out/'automatic_selection.json',selection)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError('Cannot open video')
    width, height = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    requested = defaultdict(list)
    selected_ids = []
    rejected = []
    for identity in ranked:
        candidates = []
        for node in nodes_by_id[identity]:
            for row in node['rows']:
                if keeper and identity==keeper['id'] and row[0] not in keeper['sole_frames']:
                    continue
                ov = overlap(row, by_frame[row[0]])
                if not quality_ok(row, ov, width, height):
                    continue
                if min(row[0] - node['start'], node['end'] - row[0]) < 3:
                    continue
                candidates.append((row, ov, node))
        required=2 if keeper and identity==keeper['id'] else min_crops
        if len({r[0][0] // sample_every for r in candidates}) < required:
            rejected.append(dict(id=identity, observations=len(by_id[identity]), quality_bins=len({r[0][0] // sample_every for r in candidates})))
            continue
        pid = len(selected_ids)
        selected_ids.append(identity)
        for row, ov, node in candidates:
            requested[row[0]].append(dict(pid=pid, algorithm_id=identity, source_local_id=row[1], node=node['id'],
                                          frame=row[0], x=row[2], y=row[3], w=row[4], h=row[5],
                                          confidence=row[6], overlap=ov))
    if len(selected_ids) < 2:
        cap.release()
        raise ValueError(f'Need at least 2 backbone identities with {min_crops} crops, got {len(selected_ids)}')
    best = {}
    for frame in range(total):
        ok, image = cap.read()
        if not ok:
            raise ValueError(f'Video ended at frame {frame}')
        for record in requested.get(frame, []):
            x, y, w, h = (record[k] for k in 'xywh')
            crop = image[int(y):int(y + h), int(x):int(x + w)]
            if crop.size == 0:
                continue
            sharp = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            quality = record['confidence'] * (1 - record['overlap']) * min(1, h / 100) * (.5 + .5 * min(1, sharp / 100))
            key = (record['pid'], frame // sample_every)
            if key not in best or quality > best[key][0]:
                best[key] = (quality, dict(record, sharpness=sharp, split='train'), crop.copy())
        if frame % 4000 == 0:
            print(json.dumps(dict(train_frame=frame, bins=len(best))), flush=True)
    cap.release()
    images = out / 'images' / 'train'
    images.mkdir(parents=True, exist_ok=True)
    rows = []
    for quality, record, crop in sorted(best.values(), key=lambda v: (v[1]['pid'], v[1]['frame'])):
        record['kit'] = kit_votes[record['algorithm_id']].most_common(1)[0][0] if kit_votes[record['algorithm_id']] else -1
        record['identity'] = f'backbone_{record["pid"]}'
        record['source_frame'] = record['frame']
        record['original_v3_id'] = record['algorithm_id']
        path = images / f'p{record["pid"]}_f{record["frame"]:06d}_id{record["algorithm_id"]}.jpg'
        ok, encoded = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        assert ok
        encoded.tofile(str(path))
        record['image'] = path.relative_to(out).as_posix()
        record['sha256'] = sha(path)
        record['quality_score'] = quality
        record['enrolment_gallery'] = False
        rows.append(record)
    for pid in range(len(selected_ids)):
        group = [r for r in rows if r['pid'] == pid]
        required=2 if keeper and selected_ids[pid]==keeper['id'] else min_crops
        if len(group) < required:
            raise ValueError(f'Identity {pid} has {len(group)} crops')
        for index in np.linspace(0, len(group) - 1, min(12, len(group))).astype(int):
            group[index]['enrolment_gallery'] = True
    with manifest_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dump(out / 'dataset_report.json', dict(
        windows={'train': [0, max(r['frame'] for r in rows)]},
        source_offset=0, backbone=len(selected_ids), selected_ids=selected_ids,
        per_identity={f'pid_{pid}': sum(r['pid'] == pid for r in rows) for pid in range(len(selected_ids))},
        rejected_short=rejected, total=len(rows),
        label_source='algorithm_only_simultaneous_non_goal_full_tracks', val=None, test=None,
        audit_sha256=sha(audit_path), manifest_sha256=sha(manifest_path),
    ))
    dump(out / 'backbone_ids.json', dict(selected_ids=selected_ids, min_crops=min_crops, backbone=len(selected_ids)))
    print(json.dumps(dict(backbone=len(selected_ids), selected_ids=selected_ids, total=len(rows)), ensure_ascii=False), flush=True)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backbone', type=int, default=10)
    args = parser.parse_args()
    prepare(args.audit, args.video, args.output, args.backbone)


if __name__ == '__main__':
    main()
