"""Run the full first-stage + train-only ReID + second-stage chain on a new match."""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import cv2
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def mark(out, stage, **info):
    path = Path(out) / 'progress.json'
    data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(stages=[])
    data['stages'] = [s for s in data.get('stages', []) if s.get('name') != stage]
    data['stages'].append(dict(name=stage, **info))
    data['updated'] = datetime.now().isoformat()
    dump(path, data)
    print(json.dumps(dict(stage=stage, **info), ensure_ascii=False), flush=True)


def export_final_identities(audit_path, mapping_path, out_path, min_frames=10, fps=30.0, accepted_edges_path=None):
    """Keep observations while separating publishable players from isolated IDs."""
    audit = pickle.loads(Path(audit_path).read_bytes())
    mapping = {int(k): int(v) for k, v in json.loads(Path(mapping_path).read_text(encoding='utf-8')).items()}
    quarantine_obs = set()
    quarantine_ids = set()
    for node in audit['nodes']:
        if not node.get('quarantine'):
            continue
        quarantine_ids.add(int(node['output_id']))
        for row in node['rows']:
            quarantine_obs.add((int(row[0]), int(node['output_id'])))
    kept = []
    dropped_quarantine = 0
    for row in audit['rows']:
        frame, oid = int(row[0]), int(row[1])
        if (frame, oid) in quarantine_obs:
            dropped_quarantine += 1
            continue
        kept.append((frame, mapping.get(oid, oid), *row[2:]))
    counts = defaultdict(int)
    for row in kept:
        counts[row[1]] += 1
    keep_ids = {identity for identity, n in counts.items() if n >= min_frames}
    final = [row for row in kept if row[1] in keep_ids]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open('w', encoding='utf-8') as handle:
        for frame, identity, x, y, w, h, conf in sorted(final):
            handle.write(f'{frame+1},{identity},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{conf:.6f},-1,-1,-1\n')
    groups = defaultdict(set)
    for source, canonical in mapping.items():
        if source not in quarantine_ids:
            groups[canonical].add(source)
    review_ids = set()
    if accepted_edges_path and Path(accepted_edges_path).is_file():
        for edge in json.loads(Path(accepted_edges_path).read_text(encoding='utf-8')):
            if edge.get('requires_review'):
                for key in ('from_oid','to_oid','from_','to','from'):
                    if edge.get(key) is not None:review_ids.add(int(edge[key]))
    published = {canonical: sorted(sources) for canonical, sources in groups.items()
                 if canonical in keep_ids and len(sources) >= 2 and not (sources & review_ids)}
    isolated = sorted((set(mapping) | set(keep_ids)) - {source for sources in published.values() for source in sources} - quarantine_ids)
    lengths = sorted((counts[i] for i in keep_ids), reverse=True)
    return dict(
        remaining_ids=len(keep_ids),
        remaining_rows=len(final),
        min_frames=min_frames,
        dropped_quarantine_rows=dropped_quarantine,
        dropped_short_unmerged_ids=sum(n < min_frames for n in counts.values()),
        dropped_short_unmerged_rows=sum(n for n in counts.values() if n < min_frames),
        input_rows=len(audit['rows']),
        top10_seconds=[round(n / fps, 1) for n in lengths[:10]],
        ids_at_least_10s=sum(n >= round(10 * fps) for n in lengths),
        remaining_id_list=sorted(keep_ids),
        published_players=[dict(canonical_id=canonical,source_ids=sources,confirmed=True)
                           for canonical,sources in sorted(published.items())],
        isolated_ids=isolated,
        quarantine_ids=sorted(quarantine_ids),
    )


def execute(video, calibration, out, device='cuda', backbone=10, min_frames=10):
    from first_stage import detect, associate
    from identity.pipeline import refine, export_result
    sys.path.insert(0, str(ROOT / 'scripts'))
    from prepare_backbone_train import prepare
    from features import extract_features, build_reference_bank
    from second_stage import run_second_stage
    from render_video import render_video_func

    if device != 'cuda':
        raise RuntimeError('Production identity resolution is BF16-only and requires --device cuda.')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'run_complete.json').exists():
        return json.loads((out / 'run_complete.json').read_text(encoding='utf-8'))
    video = Path(video)
    calibration = Path(calibration)
    probe = cv2.VideoCapture(str(video))
    fps = float(probe.get(cv2.CAP_PROP_FPS))
    probe.release()
    if fps <= 0:
        raise ValueError('Input video has invalid FPS metadata')
    dump(out / 'run_request.json', dict(
        video=str(video), calibration=str(calibration), backbone=backbone,
        min_frames=min_frames, device=device, started=datetime.now().isoformat(),
    ))

    detect_dir = out / 'detect'
    cache = detect(video, calibration, detect_dir, conf=.10, device='0' if device == 'cuda' else 'cpu')
    mark(out, 'detect', cache=str(cache))

    associate_dir = out / 'associate'
    audit = associate(cache, calibration, video, associate_dir, score=.74, margin=.06, dense=True)
    mark(out, 'associate', audit=str(audit))

    refine_dir = out / 'refine'
    if not (refine_dir / 'audit_data.pkl').exists():
        dense_path = detect_dir / 'fresh_dense.pkl'
        result = refine(pickle.loads(Path(audit).read_bytes()), pickle.loads(dense_path.read_bytes()))
        export_result(result, refine_dir, pickle.loads(Path(audit).read_bytes())['rows'], source_offset=0)
    refined = refine_dir / 'audit_data.pkl'
    mark(out, 'refine', audit=str(refined))

    train_data = out / 'train_data'
    manifest = prepare(refined, video, train_data, backbone=backbone)
    mark(out, 'prepare_train', manifest=str(manifest))

    train_dir = out / 'train'
    best = train_dir / 'best.pth'
    if not best.exists():
        sys.path.insert(0, str(ROOT / 'reid_model'))
        import train_adapt
        config = json.loads((ROOT / 'reid_model' / 'config.json').read_text(encoding='utf-8'))
        config.update(dict(
            manifest=str(manifest.resolve()), run_dir=str(train_dir.resolve()),
            data_root=str(train_data.resolve()), train_only=True, num_class=backbone,
            batch_identities=backbone, autotune_k=[2, 4], workers=0,
        ))
        train_adapt.run(config)
    if not best.exists():
        raise RuntimeError('Training finished without best.pth')
    mark(out, 'train', weight=str(best))

    model_config = json.loads((ROOT / 'configs' / 'reid_config.json').read_text(encoding='utf-8'))
    started = json.loads((train_dir / 'started.json').read_text(encoding='utf-8'))
    model_config['num_class'] = int(started.get('num_class', backbone))
    quality = json.loads((ROOT / 'configs' / 'pipeline_config.json').read_text(encoding='utf-8'))['inference_quality']
    quality = dict(quality)
    quality.pop('observation_policy', None)
    quality.update(sample_every_frames=max(1, round(fps / 2)), short_track_frames=max(1, round(1.5 * fps)))

    features_dir = out / 'features'
    cache_features = extract_features(refined, video, best, model_config, features_dir, quality, device, 8)
    mark(out, 'features', cache=str(cache_features))

    reference = features_dir / 'reference_bank.npz'
    if not reference.exists():
        build_reference_bank(dict(
            manifest=str(manifest.resolve()), reid_weights=str(best.resolve()),
            audit_data=str(refined.resolve()), allow_new_match_references=True,
        ), model_config, reference, device, 8)
    association_dir = out / 'association'
    tracking_path = association_dir / 'tracking_reid_trajectory.txt'
    if tracking_path.exists():
        result = dict(output_tracking=str(tracking_path),
                      report=json.loads((association_dir / 'association_report.json').read_text(encoding='utf-8')))
    else:
        second_config = json.loads((ROOT / 'configs' / 'pipeline_config.json').read_text(encoding='utf-8'))['second_stage']
        second_config = dict(second_config)
        second_config.update(reference_file=str(reference.resolve()), allow_new_match_references=True, use_training_references=True,
                             fps=fps, max_gap=round(60 * fps), local_window_frames=round(2 * fps))
        result = run_second_stage(cache_features, association_dir, second_config)
    mark(out, 'second_stage', tracking=result['output_tracking'], report=result['report'])

    filtered_dir = out / 'filtered'
    filtered_dir.mkdir(parents=True, exist_ok=True)
    filtered_tracking = filtered_dir / 'tracking_filtered.txt'
    stats = export_final_identities(refined, association_dir / 'trajectory_mapping.json', filtered_tracking, min_frames, fps,
                                    association_dir / 'accepted_edges.json')
    dump(filtered_dir / 'filter_report.json', {k: v for k, v in stats.items() if k not in {'remaining_id_list','published_players'}})
    dump(filtered_dir / 'remaining_ids.json', dict(remaining_ids=stats['remaining_ids'], ids=stats['remaining_id_list']))
    dump(filtered_dir / 'identity_catalog.json', dict(
        schema_version=1,
        policy='confirmed_multi_id_groups_only',
        published_players=stats['published_players'],
        isolated_ids=stats['isolated_ids'],
        quarantine_ids=stats['quarantine_ids'],
    ))
    shutil.copy2(result['output_tracking'], filtered_dir / 'tracking_before_filter.txt')
    mark(out, 'filter', remaining_ids=stats['remaining_ids'], remaining_rows=stats['remaining_rows'],
         dropped_quarantine_rows=stats['dropped_quarantine_rows'],
         dropped_short_unmerged_ids=stats['dropped_short_unmerged_ids'])

    review = out / 'review.mp4'
    if review.exists():
        render_info = dict(skipped=True, output=str(review))
    else:
        render_info = render_video_func(video, filtered_tracking, review, fps, 1,
                                        quarantine_ids=set(), show_confidence=True, show_legend=True)
    mark(out, 'render', **render_info)
    dump(out / 'run_complete.json', dict(status='complete', filtered=stats, render=render_info, association=result['report']))
    return dict(status='complete', output=str(out), filtered=stats, association=result['report'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=['cuda'], default='cuda')
    parser.add_argument('--backbone', type=int, default=10)
    parser.add_argument('--min-frames', type=int, default=10)
    args = parser.parse_args(argv)
    out = args.output or (ROOT.parent / 'reid_pipeline_runs' / datetime.now().strftime('july24_%Y%m%d_%H%M%S'))
    print(json.dumps(execute(args.video, args.calibration, out, args.device, args.backbone, args.min_frames), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
