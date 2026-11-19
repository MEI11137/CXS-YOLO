#!/usr/bin/env python
"""Image-level bootstrap confidence intervals for a single-class YOLO detector.

No pycocotools is required: the script implements the COCO-style AP for one class
(all-point interpolation, IoU 0.50:0.95, max 100 detections per image), which is
what the paper reports.

Typical use (from the repository root, with the environment that has torch):

    python tools/bootstrap_ci.py ^
        --weights best.pt ^
        --data data.yaml ^
        --split test ^
        --imgsz 640 ^
        --conf 0.001 ^
        --n 1000 ^
        --out bootstrap_ci.json

The network runs once; each bootstrap replicate only re-uses the saved predictions.
"""

import argparse
import json
import os
import random
import sys

import numpy as np

IOU_THRS = np.linspace(0.5, 0.95, 10)


def xywhn_to_xyxy(box, w, h):
    x, y, bw, bh = box
    return np.array([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0][:, None], a[:, 1][:, None], a[:, 2][:, None], a[:, 3][:, None]
    bx1, by1, bx2, by2 = b[:, 0][None, :], b[:, 1][None, :], b[:, 2][None, :], b[:, 3][None, :]
    inter = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None) * \
            np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def ap_single_class(preds, gts, n_images, iou_thr):
    """preds: list of (img_id, xyxy, score); gts: dict img_id -> xyxy array."""
    per_img = {}
    for img_id, box, score in preds:
        per_img.setdefault(img_id, []).append((score, box))
    n_gt = sum(len(gts.get(i, [])) for i in range(n_images))
    if n_gt == 0:
        return float("nan")
    records = []
    for img_id in range(n_images):
        gt = gts.get(img_id, np.zeros((0, 4)))
        dt = sorted(per_img.get(img_id, []), key=lambda t: -t[0])[:100]
        if len(dt) == 0:
            continue
        dboxes = np.stack([d[1] for d in dt])
        ious = iou_matrix(dboxes, gt) if len(gt) else np.zeros((len(dboxes), 0))
        used = np.zeros(len(gt), dtype=bool)
        for k, (score, _) in enumerate(dt):
            best, best_j = 0.0, -1
            for j in range(len(gt)):
                if used[j]:
                    continue
                if ious[k, j] > best:
                    best, best_j = ious[k, j], j
            matched = best >= iou_thr
            if matched:
                used[best_j] = True
            records.append((score, 1.0 if matched else 0.0))
    if not records:
        return 0.0
    records.sort(key=lambda t: -t[0])
    tp = np.cumsum([r[1] for r in records])
    fp = np.cumsum([1 - r[1] for r in records])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(preds, gts, n_images):
    aps = [ap_single_class(preds, gts, n_images, t) for t in IOU_THRS]
    return {
        "AP50": aps[0],
        "AP": float(np.mean(aps)),
    }


def run_inference(args):
    from ultralytics import YOLO
    from pathlib import Path

    model = YOLO(args.weights)
    root = Path(args.images)
    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    print(f"images: {len(files)}")
    preds, gts = [], {}
    for i, f in enumerate(files):
        r = model.predict(str(f), imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)[0]
        h, w = r.orig_shape
        label = Path(args.labels) / (f.stem + ".txt")
        g = []
        if label.exists():
            for line in label.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    g.append(xywhn_to_xyxy([float(x) for x in parts[1:5]], w, h))
        gts[i] = np.array(g) if g else np.zeros((0, 4))
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            for box, s in zip(xyxy, conf):
                preds.append((i, box, float(s)))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(files)}")
    return preds, gts, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True, help="folder with the test images")
    ap.add_argument("--labels", required=True, help="folder with the YOLO txt labels")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="bootstrap_ci.json")
    args = ap.parse_args()

    preds, gts, n = run_inference(args)
    point = evaluate(preds, gts, n)
    print("point estimate: AP {:.2f}  AP50 {:.2f}".format(point["AP"] * 100, point["AP50"] * 100))

    rng = random.Random(args.seed)
    ap_boot, ap50_boot = [], []
    for _ in range(args.n):
        idx = [rng.randrange(n) for _ in range(n)]
        # re-index images so that each replicate is a resample with replacement
        remap = {}
        sub_preds, sub_gts = [], {}
        for new_id, old_id in enumerate(idx):
            remap.setdefault(old_id, []).append(new_id)
        for old_id, news in remap.items():
            for nid in news:
                sub_gts[nid] = gts[old_id]
        for (img_id, box, score) in preds:
            for nid in remap.get(img_id, []):
                sub_preds.append((nid, box, score))
        res = evaluate(sub_preds, sub_gts, n)
        if not np.isnan(res["AP"]):
            ap_boot.append(res["AP"] * 100)
            ap50_boot.append(res["AP50"] * 100)

    out = {
        "point": {"AP": point["AP"] * 100, "AP50": point["AP50"] * 100},
        "bootstrap": {
            "n": len(ap_boot),
            "AP_95CI": [float(np.percentile(ap_boot, 2.5)), float(np.percentile(ap_boot, 97.5))],
            "AP50_95CI": [float(np.percentile(ap50_boot, 2.5)), float(np.percentile(ap50_boot, 97.5))],
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
