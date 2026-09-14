"""Freeze algorithm-seed train crops, then add unused-track val and annotated-fragment test."""
import argparse
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_backbone_train import dump, overlap, quality_ok, sha


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def abs_image(row, data_root):
    image = Path(row["image"])
    if image.is_absolute():
        return str(image)
    return str((data_root / image).resolve())


def ints(row, keys):
    for k in keys:
        if k in row and row[k] != "":
            row[k] = int(row[k])
    return row


def harvest(audit, video, ids, skip_keys, mapping, train_rows, split, label_source, dest_root, sample_every=15):
    byframe = defaultdict(list)
    for n in audit["nodes"]:
        for r in n["rows"]:
            byframe[r[0]].append(r)
    cap = cv2.VideoCapture(str(video))
    width, height = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(7))
    requests = defaultdict(list)
    rejected = defaultdict(int)
    for n in audit["nodes"]:
        oid = n["output_id"]
        if oid not in ids or n.get("quarantine"):
            continue
        for row in n["rows"]:
            if (row[1], row[0]) in skip_keys:
                continue
            ov = overlap(row, byframe[row[0]])
            if not quality_ok(row, ov, width, height):
                rejected[oid] += 1
                continue
            requests[row[0]].append((n, row, ov))
    best = {}
    for frame in range(total):
        ok, image = cap.read()
        assert ok
        for n, row, ov in requests.get(frame, ()):
            oid = n["output_id"]
            pid = mapping[oid]
            _, local, x, y, w, h, conf = row
            crop = image[int(y) : int(y + h), int(x) : int(x + w)].copy()
            if crop.size == 0:
                continue
            sharp = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            score = conf * (1 - ov) * (0.5 + 0.5 * min(1, sharp / 100))
            key = (oid, frame // sample_every)
            prototype = next(r for r in train_rows if r["pid"] == pid)
            rec = {k: "" for k in prototype}
            rec.update(
                pid=pid,
                identity=prototype["identity"],
                algorithm_id=oid,
                original_v3_id=oid,
                source_local_id=local,
                node=n["id"],
                frame=frame,
                source_frame=frame,
                x=x,
                y=y,
                w=w,
                h=h,
                confidence=conf,
                overlap=ov,
                sharpness=sharp,
                kit=prototype["kit"],
                quality_score=score,
                enrolment_gallery=False,
                label_source=label_source,
                nearest_training_gap=min(abs(frame - t["frame"]) for t in train_rows if t["pid"] == pid),
                split=split,
            )
            if key not in best or score > best[key][0]:
                best[key] = (score, rec, crop)
    cap.release()
    out = []
    dest_root.mkdir(parents=True, exist_ok=True)
    for _, rec, crop in sorted(best.values(), key=lambda v: (v[1]["pid"], v[1]["frame"])):
        path = dest_root / f'p{rec["pid"]}_f{rec["frame"]:06d}_id{rec["algorithm_id"]}.jpg'
        ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        assert ok
        encoded.tofile(str(path))
        rec["image"] = str(path.resolve())
        rec["sha256"] = sha(path)
        out.append(rec)
    return out, dict(rejected)


def prepare(out, train_manifest, train_data_root, audit_path, video, review_path):
    out = Path(out)
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    if (data / "fragment_protocol.json").exists():
        raise ValueError("Protocol already frozen")
    train = list(csv.DictReader(Path(train_manifest).open(encoding="utf-8")))
    assert all(r["split"] == "train" for r in train)
    train_root = Path(train_data_root)
    for r in train:
        ints(r, ["pid", "frame", "source_local_id", "algorithm_id", "node", "original_v3_id", "kit", "source_frame"])
        r["image"] = abs_image(r, train_root)
        r["label_source"] = "algorithm_seed_only"
        r["enrolment_gallery"] = str(r.get("enrolment_gallery", "")).lower() in {"true", "1", "yes"}
        r["nearest_training_gap"] = -1
    train_path = data / "train_manifest.csv"
    write_csv(train_path, train)
    frozen = sha(train_path)
    review = __import__("json").loads(Path(review_path).read_text(encoding="utf-8"))
    seed_ids = {r["algorithm_id"] for r in train}
    mapping = {}
    for pid in sorted({r["pid"] for r in train}):
        trunk = next(r["algorithm_id"] for r in train if r["pid"] == pid)
        groups = [g for g in review["groups"].values() if trunk in g]
        if len(groups) != 1:
            raise ValueError(("Selected algorithm identity lacks evaluation mapping", trunk))
        for oid in groups[0]:
            if oid in mapping and mapping[oid] != pid:
                raise ValueError("Two selected classes belong to same reviewed person")
            mapping[oid] = pid
    audit = pickle.load(Path(audit_path).open("rb"))
    skip = {(r["source_local_id"], r["frame"]) for r in train}
    val, val_rej = harvest(
        audit, video, seed_ids, skip, {r["algorithm_id"]: r["pid"] for r in train}, train,
        "val", "algorithm_seed_only", data / "images" / "val", sample_every=15,
    )
    test, test_rej = harvest(
        audit, video, {oid for oid in mapping if oid not in seed_ids} - set(review["invalid_ids"]),
        skip, mapping, train, "test", "user_review_evaluation_only", data / "images" / "test", sample_every=15,
    )
    if {r["pid"] for r in val} != {r["pid"] for r in train}:
        missing = {r["pid"] for r in train} - {r["pid"] for r in val}
        raise ValueError(("Val missing unused frames for training pids", missing))
    rows = train + val + test
    if len({(r["source_local_id"], r["frame"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate source frames")
    if len({r["sha256"] for r in rows}) != len(rows):
        raise ValueError("Duplicate crop hashes")
    assert sha(train_path) == frozen
    write_csv(data / "manifest.csv", rows)
    sets = {s: {r["algorithm_id"] for r in rows if r["split"] == s} for s in ("train", "val", "test")}
    report = dict(
        seed_ids=sorted(seed_ids),
        train_manifest_sha256=frozen,
        manifest_sha256=sha(data / "manifest.csv"),
        review_sha256=sha(review_path),
        review_used_after_training_freeze=True,
        val_from_unused_train_track_frames=True,
        whole_id_disjoint=False,
        train_val_share_algorithm_ids=True,
        test_uses_annotated_fragments=True,
        counts={s: sum(r["split"] == s for r in rows) for s in sets},
        coverage={s: len({r["pid"] for r in rows if r["split"] == s}) for s in sets},
        partitions={str(pid): dict(val="unused_frames_of_training_track", test=sorted(sets["test"] & {oid for oid, p in mapping.items() if p == pid}))
                    for pid in sorted({r["pid"] for r in train})},
        quality_rejected=dict(val=val_rej, test=test_rej),
        invalid_ids_excluded=review["invalid_ids"],
        no_identity_expansion_into_training=True,
        caveat="Val is unused frames of the same algorithm trunks. Test is user-reviewed fragments only.",
    )
    dump(data / "fragment_protocol.json", report)
    import json
    c = json.loads((ROOT / "reid_model" / "config.json").read_text(encoding="utf-8"))
    c.update(
        num_class=len(seed_ids),
        manifest=str((data / "manifest.csv").resolve()),
        data_root=str(data.resolve()),
        run_dir=str((out / "training").resolve()),
        weight_file=str((ROOT / "reid_model" / "weights" / "jx_vit_base_p16_224-80ecf9dd.pth").resolve()),
        workers=0,
        eval_batch=8,
        autotune_k=[2, 4],
        epochs=20,
        patience=5,
        steps_per_epoch=20,
        seed=20260910,
        train_only=False,
        fragment_protocol=str((data / "fragment_protocol.json").resolve()),
        fragment_appearance=dict(views=["original", "light", "heavy"]),
    )
    dump(data / "training_config.json", c)
    print(__import__("json").dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--train-data-root", type=Path, required=True)
    p.add_argument("--audit", type=Path, default=ROOT / "data" / "audit_data.pkl")
    p.add_argument("--video", type=Path, default=ROOT / "data" / "video.mp4")
    p.add_argument("--review", type=Path, default=ROOT / "configs" / "review_20260910.json")
    a = p.parse_args()
    prepare(a.output, a.train_manifest, a.train_data_root, a.audit, a.video, a.review)
