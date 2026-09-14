"""Full-video detection, jersey evidence and adjacent association.

Clip-length constants stay out of this module. Merge score/margin/gap match
the existing first-stage rules and are not relaxed here.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = ROOT.parents[1]
TRACK = ROOT / 'vendor' / 'tracking'
SPORTS = ROOT / 'vendor' / 'sports_osnet'
WEIGHTS = Path(os.getenv('FOOTBALL_INSIGHT_DETECTOR_WEIGHTS', str(SYSTEM_ROOT / 'models' / 'yolov8x.pt'))).resolve()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')


def write_mot(path, rows):
    with Path(path).open('w', encoding='utf8') as handle:
        for frame, identity, x, y, w, h, conf in sorted(rows):
            handle.write(f'{frame+1},{identity},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{conf:.5f},-1,-1,-1\n')


def unit(x):
    x = np.asarray(x, dtype=float)
    return x / max(np.linalg.norm(x), 1e-9)


def runs(frames):
    out = []
    for frame in sorted(set(frames)):
        if not out or frame != out[-1][-1] + 1:
            out.append([])
        out[-1].append(frame)
    return out


def statistics(rows):
    tracks = defaultdict(list)
    for row in rows:
        tracks[int(row[1])].append(int(row[0]))
    lengths = [len(v) for v in tracks.values()]
    continuous = [max(map(len, runs(v))) for v in tracks.values()]
    return dict(
        rows=len(rows), ids=len(tracks),
        median_observed_s=float(np.median(lengths)) / 30 if lengths else 0,
        maximum_observed_s=max(lengths, default=0) / 30,
        ids_at_least_5s=sum(n >= 150 for n in lengths),
        ids_at_least_10s=sum(n >= 300 for n in lengths),
        maximum_contiguous_s=max(continuous, default=0) / 30,
        top10_mean_observed_s=float(np.mean(sorted(lengths, reverse=True)[:10])) / 30 if lengths else 0,
        max_simultaneous=max(Counter(int(r[0]) for r in rows).values(), default=0),
    )


def detect(video, calibration, out, conf=0.10, device='0'):
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f'Detector weights not found: {WEIGHTS}')
    if not (SPORTS / 'checkpoints' / 'sports_model.pth.tar-60').is_file():
        raise FileNotFoundError('Bundled Sports-OSNet checkpoint is missing')
    sys.path.insert(0, str(TRACK))
    import tracking_core as core
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'fresh.pkl'
    if cache.exists():
        return cache
    video = Path(video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {video}')
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f'Invalid source FPS: {fps}')
    calib = json.loads(Path(calibration).read_text(encoding='utf8'))
    if int(calib['video_metadata']['proc_total_frames']) != frames:
        raise ValueError('Calibration frame count does not match the video')
    if not calib.get('validation', {}).get('passed'):
        raise ValueError('Calibration validation did not pass')
    args = SimpleNamespace(
        video=str(video), weights=str(WEIGHTS),
        tracker_config=str(TRACK / 'config' / 'botsort_buffer.yaml'),
        vid_stride=1, reid_stride=10, reid_interval_sec=.33, reid_max_iou=.4,
        conf=conf, imgsz=1280, device=str(device),
    )
    started = time.perf_counter()
    reid = core.SportsOSNetReIDExtractor(SPORTS, SPORTS / 'checkpoints' / 'sports_model.pth.tar-60')
    detections, ball, tracklets, total = core.stage1_detect_track(args, reid)
    if total != frames:
        raise ValueError(f'Detector processed {total} frames, video has {frames}')
    result = dict(
        detections=detections, ball=ball, tracklets=dict(tracklets),
        total_frames=total, fps=float(fps), source_frame_offset=0, args=vars(args),
        elapsed_seconds=time.perf_counter() - started,
    )
    with cache.open('wb') as handle:
        pickle.dump(result, handle)
    summary = {k: result[k] for k in ('total_frames', 'fps', 'source_frame_offset', 'args', 'elapsed_seconds')}
    summary.update(detections=len(detections), local_ids=len(tracklets))
    write_json(out / 'fresh.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return cache


def extract_dense(video, detections, destination, total_frames):
    destination = Path(destination)
    if destination.exists():
        return pickle.load(destination.open('rb'))
    by_frame = defaultdict(list)
    for row in detections:
        by_frame[row[0]].append(row)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {video}')
    evidence = {}
    try:
        for frame in range(total_frames):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'Video ended at frame {frame}')
            rows = by_frame.get(frame, [])
            for row in rows:
                _, identity, x, y, w, h, conf = row
                crop = image[max(0, int(y + .12 * h)):min(image.shape[0], int(y + .52 * h)),
                             max(0, int(x + .25 * w)):min(image.shape[1], int(x + .75 * w))]
                probs = np.zeros(3)
                if crop.size:
                    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    hue, sat, val = cv2.split(hsv)
                    valid = (sat > 65) & (val > 45)
                    probs = np.array([
                        np.mean(valid & (hue >= 15) & (hue < 38)),
                        np.mean(valid & (hue >= 88) & (hue <= 135)),
                        np.mean(valid & ((hue < 12) | (hue > 170))),
                    ])
                order = np.argsort(probs)
                label = int(order[-1]) if probs[order[-1]] >= .30 and probs[order[-1]] - probs[order[-2]] >= .18 else -1
                overlap = 0.
                for other in rows:
                    if other[1] == identity:
                        continue
                    _, _, ox, oy, ow, oh, *_ = other
                    inter = max(0, min(x + w, ox + ow) - max(x, ox)) * max(0, min(y + h, oy + oh) - max(y, oy))
                    overlap = max(overlap, inter / max(w * h, 1))
                evidence[(frame, identity)] = dict(label=label, strength=float(max(probs)), probs=probs, overlap=overlap)
            if frame % 2000 == 0:
                print(json.dumps(dict(dense_frame=frame, observations=len(evidence))), flush=True)
    finally:
        cap.release()
    with destination.open('wb') as handle:
        pickle.dump(evidence, handle)
    return evidence


def endpoint_velocity(node, side):
    rows = node['rows'][-20:] if side == 'end' else node['rows'][:20]
    t = np.array([x[0] for x in rows], float)
    t -= t.mean()
    pts = np.array([node['positions'][x[0]] for x in rows])
    return (t[:, None] * pts).sum(0) / max(float(t @ t), 1e-9)


def choose_edges(candidates, min_score=.8, margin=.06):
    outgoing, incoming = defaultdict(list), defaultdict(list)
    for edge in candidates:
        outgoing[edge['from']].append(edge)
        incoming[edge['to']].append(edge)
    for table in (outgoing, incoming):
        for edges in table.values():
            edges.sort(key=lambda e: (-e['score'], e['from'], e['to']))
    selected = []
    for edge in candidates:
        a, b = outgoing[edge['from']], incoming[edge['to']]
        edge['out_margin'] = edge['score'] - (a[1]['score'] if len(a) > 1 else 0)
        edge['in_margin'] = edge['score'] - (b[1]['score'] if len(b) > 1 else 0)
        edge['accepted'] = bool(edge is a[0] and edge is b[0] and edge['score'] >= min_score and min(edge['out_margin'], edge['in_margin']) >= margin)
        if edge['accepted']:
            selected.append(edge)
    return selected


def associate(cache, calibration, video, out, score=.74, margin=.06, dense=True):
    sys.path.insert(0, str(TRACK))
    from tracking_lib.metric_field_gate import filter_detections_by_dynamic_pitch
    from tracking_lib.precision_association import (
        AssociationConfig, find_kit_outlier_frames, quarantine_tracklet_frames,
        split_tracklets_on_persistent_kit_change, associate_tracklets,
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    audit_path = out / 'audit_data.pkl'
    if audit_path.exists():
        return audit_path
    data = pickle.load(Path(cache).open('rb'))
    detections = data['detections']
    total_frames = int(data['total_frames'])
    gated = filter_detections_by_dynamic_pitch(detections, calibration)
    tracks = copy.deepcopy(data['tracklets'])
    from tracking_lib.field_filter import restrict_tracklet_frames
    restrict_tracklet_frames(tracks, gated.detections)
    tracks = {i: t for i, t in tracks.items() if t['frames']}
    cfg = AssociationConfig(
        min_track_frames=1, min_presence_ratio=0, maximum_merge_gap_seconds=5,
        appearance_weight=.85, colour_weight=.15, short_reid_min=.94, medium_reid_min=.99, long_reid_min=.99,
        short_score_min=.92, medium_score_min=.99, long_score_min=.99,
    )
    rejected = {i: find_kit_outlier_frames(t, cfg) for i, t in tracks.items()}
    quarantine = quarantine_tracklet_frames(tracks, gated.track_positions_m, rejected)
    split = split_tracklets_on_persistent_kit_change(quarantine.tracklets, quarantine.metric_positions, cfg)
    remap = {key: split.frame_id_remap.get((q, key[1]), q) for key, q in quarantine.frame_id_remap.items()}
    baseline = associate_tracklets(split.tracklets, total_frames, 30, split.metric_positions, cfg)
    base_rows = [(r[0], baseline.local_to_global[remap[(r[1], r[0])]], *r[2:]) for r in gated.detections]
    write_mot(out / 'fresh_v5.txt', base_rows)
    evidence = None
    if dense:
        evidence = extract_dense(video, detections, Path(cache).with_name(Path(cache).stem + '_dense.pkl'), total_frames)
        grouped = defaultdict(list)
        for row in gated.detections:
            grouped[row[1]].append(row)
    else:
        grouped = defaultdict(list)
        for row in gated.detections:
            grouped[remap[(row[1], row[0])]].append(row)
    nodes = []
    row_to_node = {}
    for local, rows in sorted(grouped.items()):
        by_f = {r[0]: r for r in rows}
        tk = tracks[local] if dense else split.tracklets[local]
        pos = {int(f): np.array([x, y]) for f, x, y in (gated.track_positions_m[local] if dense else split.metric_positions[local])}
        groups = runs(by_f)
        if dense:
            ordered = sorted(by_f)
            labels = []
            for j, f in enumerate(ordered):
                votes = [evidence[(g, local)]['label'] for g in ordered[max(0, j - 2):j + 3] if abs(g - f) <= 3 and evidence[(g, local)]['overlap'] < .35]
                votes = [v for v in votes if v >= 0]
                count = Counter(votes)
                labels.append(count.most_common(1)[0][0] if count and count.most_common(1)[0][1] >= 2 else -1)
            groups = []
            last_label = -1
            for j, f in enumerate(ordered):
                changed = labels[j] >= 0 and last_label >= 0 and labels[j] != last_label
                jump = False
                if j:
                    prev = ordered[j - 1]
                    gap = f - prev
                    left, right = by_f[prev], by_f[f]
                    pixel_jump = np.linalg.norm(np.array([left[2] + left[4] / 2, left[3] + left[5]]) - np.array([right[2] + right[4] / 2, right[3] + right[5]]))
                    jump = gap <= 2 and pixel_jump > max(left[5], right[5]) * 1.2 and np.linalg.norm(pos[f] - pos[prev]) > 3.
                if not groups or f - groups[-1][-1] > 10 or changed or jump:
                    groups.append([])
                groups[-1].append(f)
                if labels[j] >= 0:
                    last_label = labels[j]
        run_specs = [(run, bool(tk.get('association_quarantine'))) for run in groups]
        if dense:
            run_specs = []
            for run in groups:
                votes = [evidence[(f, local)]['label'] for f in run if evidence[(f, local)]['label'] >= 0 and evidence[(f, local)]['overlap'] < .35]
                dominant = Counter(votes).most_common(1)[0][0] if votes else -1
                bad = []
                for f in run:
                    ev = evidence[(f, local)]
                    if ev['overlap'] >= .5 or (dominant >= 0 and ev['label'] >= 0 and ev['label'] != dominant):
                        bad.append(f)
                good = sorted(set(run) - set(bad))
                if good:
                    run_specs.append((good, False))
                run_specs.extend((piece, True) for piece in runs(bad))
        for run, is_quarantine in run_specs:
            node = dict(
                id=len(nodes), local=local, source_local=int(by_f[run[0]][1]), rows=[by_f[f] for f in run],
                start=run[0], end=run[-1], positions=pos, quarantine=is_quarantine,
                samples=[s for s in tk['appearance_samples'] if s[0] in set(run)],
            )
            if dense:
                votes = [evidence[(f, local)]['label'] for f in run if evidence[(f, local)]['label'] >= 0 and evidence[(f, local)]['overlap'] < .35]
                node['kit'] = Counter(votes).most_common(1)[0][0] if votes else -1
            else:
                node['kit'] = -1
            nodes.append(node)
            for f in run:
                row_to_node[(by_f[f][1], f)] = node['id']
    candidates = []
    for left in nodes:
        for right in nodes:
            gap = right['start'] - left['end']
            if not 1 <= gap <= 60 or left['quarantine'] or right['quarantine']:
                continue
            if left['kit'] >= 0 and right['kit'] >= 0 and left['kit'] != right['kit']:
                continue
            distance = float(np.linalg.norm(right['positions'][right['start']] - left['positions'][left['end']]))
            if distance > 1. + 9. * gap / 30:
                continue
            lv, rv = endpoint_velocity(left, 'end'), endpoint_velocity(right, 'start')
            residual = max(
                float(np.linalg.norm(left['positions'][left['end']] + lv * gap - right['positions'][right['start']])),
                float(np.linalg.norm(right['positions'][right['start']] - rv * gap - left['positions'][left['end']])),
            )
            if residual > .7 + 3. * gap / 30:
                continue
            ls, rs = left['samples'][-3:], right['samples'][:3]
            appearance = colour = None
            if ls and rs:
                appearance = float(unit(np.mean([s[1] for s in ls], 0)) @ unit(np.mean([s[1] for s in rs], 0)))
                colour = float(unit(np.mean([s[2] for s in ls], 0)) @ unit(np.mean([s[2] for s in rs], 0)))
                if appearance < .80 or colour < .35:
                    continue
            else:
                if gap > 10 or left['source_local'] != right['source_local'] or distance > .8:
                    continue
            motion = math.exp(-residual / (.6 + 2. * gap / 30))
            score_value = .5 * motion + .35 * (appearance if appearance is not None else .85) + .15 * (colour if colour is not None else .7)
            candidates.append(dict(
                from_=left['id'], to=right['id'], gap=gap, score=score_value, distance_m=distance,
                residual_m=residual, appearance=appearance, colour=colour,
            ))
    for edge in candidates:
        edge['from'] = edge.pop('from_')
    selected = choose_edges(candidates, score, margin)
    predecessor = {e['to']: e['from'] for e in selected}
    roots = {}
    for node in nodes:
        root = node['id']
        while root in predecessor:
            root = predecessor[root]
        roots[node['id']] = root + 1
    out_rows = [(r[0], roots[row_to_node[(r[1], r[0])]], *r[2:]) for r in gated.detections]
    assert len(set((r[0], r[1]) for r in out_rows)) == len(out_rows) == len(gated.detections)
    write_mot(out / 'tracking_long.txt', out_rows)
    write_mot(out / 'tracking_local.txt', gated.detections)
    report = dict(
        fresh_local=statistics(detections), metric_local=statistics(gated.detections), fresh_v5=statistics(base_rows),
        adjacent_safe=statistics(out_rows), gate=gated.report, quarantine=quarantine.report, persistent_split=split.report,
        parameters=dict(score=score, margin=margin, max_gap_frames=60, dense=dense), nodes=len(nodes), links=len(selected),
        same_frame_conflicts=0, source_frames_zero_based=[0, total_frames - 1],
    )
    q_ids = {roots[n['id']] for n in nodes if n['quarantine']}
    report['eligible_tracks'] = statistics([r for r in out_rows if r[1] not in q_ids])
    report['quarantine_output'] = dict(ids=len(q_ids), rows=sum(r[1] in q_ids for r in out_rows), training_included=False)
    for node in nodes:
        node['output_id'] = roots[node['id']]
    with audit_path.open('wb') as handle:
        pickle.dump(dict(nodes=nodes, rows=out_rows, selected=selected), handle)
    write_json(out / 'edges.json', candidates)
    write_json(out / 'summary.json', report)
    print(json.dumps({k: report[k] for k in ['fresh_local', 'metric_local', 'fresh_v5', 'adjacent_safe', 'nodes', 'links']}, indent=2), flush=True)
    return audit_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['detect', 'associate'], required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--conf', type=float, default=.10)
    parser.add_argument('--score', type=float, default=.74)
    parser.add_argument('--margin', type=float, default=.06)
    parser.add_argument('--device', default='0')
    args = parser.parse_args(argv)
    if args.stage == 'detect':
        detect(args.video, args.calibration, args.output, args.conf, args.device)
    else:
        if args.cache is None:
            raise ValueError('associate requires --cache')
        associate(args.cache, args.calibration, args.video, args.output, args.score, args.margin, True)


if __name__ == '__main__':
    main()
