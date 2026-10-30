"""Score cached detections from baseline.py with pycocotools; no inference.

Usage:
    python src/evaluate.py --run results/baseline-val300

For each cache/<variant>.json: COCO AP (50, 50:95, small/medium/large), AR@100,
and precision/recall/F1 at IoU 0.5 for each sweep threshold. Ground-truth boxes
marked iscrowd/ignore (tiny boxes) are neither required nor counted as misses.
"""
import argparse, contextlib, io, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils import load_config, save_metrics

STAT_NAMES = ["AP50:95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
              "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


def iou(a, b):
    """IoU between xywh boxes a (N,4) and b (M,4)."""
    ax1, ay1, bx1, by1 = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
    ax2, ay2, bx2, by2 = ax1 + a[:, 2], ay1 + a[:, 3], bx1 + b[:, 2], by1 + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(ax1[:, None], bx1), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(ay1[:, None], by1), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1) - inter)


def match_at(gt_by_img, dets, image_ids, thr, iou_thr=0.5):
    """Greedy score-ordered matching; detections on ignore boxes are discarded."""
    dets_by_img = defaultdict(list)
    for d in dets:
        if d["score"] >= thr:
            dets_by_img[d["image_id"]].append(d)
    tp = fp = n_gt = 0
    for img_id in image_ids:
        gts = gt_by_img.get(img_id, [])
        ds = sorted(dets_by_img[img_id], key=lambda d: -d["score"])
        n_gt += sum(not g["iscrowd"] for g in gts)
        if not ds:
            continue
        if not gts:
            fp += len(ds)
            continue
        ious = iou(np.array([d["bbox"] for d in ds]), np.array([g["bbox"] for g in gts]))
        taken = np.zeros(len(gts), bool)
        for i in range(len(ds)):
            cand = [j for j in np.argsort(-ious[i]) if ious[i, j] >= iou_thr and (gts[j]["iscrowd"] or not taken[j])]
            if not cand:
                fp += 1
            elif gts[cand[0]]["iscrowd"]:
                continue
            else:
                taken[cand[0]] = True
                tp += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / n_gt if n_gt else 0.0
    return {"threshold": thr, "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0, "tp": tp, "fp": fp, "fn": n_gt - tp}


def coco_stats(gt, dets, image_ids):
    if not dets:
        return {k: 0.0 for k in STAT_NAMES}
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(dets), "bbox")
        ev.params.imgIds = image_ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return {k: round(float(v), 4) for k, v in zip(STAT_NAMES, ev.stats)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    run_dir = ap.parse_args().run
    cfg = load_config(run_dir / "config.yaml")
    image_ids = json.loads((run_dir / "run.json").read_text())["image_ids"]

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(Path(cfg["manifests"]) / f"{cfg['split']}.json"))
    gt_by_img = defaultdict(list)
    for a in gt.loadAnns(gt.getAnnIds(imgIds=image_ids)):
        gt_by_img[a["image_id"]].append(a)

    metrics = {"split": cfg["split"], "n_images": len(image_ids), "variants": {}}
    for cache in sorted((run_dir / "cache").glob("*.json")):
        dets = json.loads(cache.read_text())
        sweep = [match_at(gt_by_img, dets, image_ids, t) for t in cfg["sweep_thresholds"]]
        best = max(sweep, key=lambda s: s["f1"])
        metrics["variants"][cache.stem] = {**coco_stats(gt, dets, image_ids), "best_f1": best, "sweep": sweep}

    print(f"{'variant':28} {'AP50':>6} {'AP':>6} {'APs':>6} {'APm':>6} {'AR100':>6} | best F1 @ thr (P / R)")
    for name, m in metrics["variants"].items():
        b = m["best_f1"]
        print(f"{name:28} {m['AP50']:6.3f} {m['AP50:95']:6.3f} {m['AP_small']:6.3f} {m['AP_medium']:6.3f} "
              f"{m['AR100']:6.3f} | {b['f1']:.3f} @ {b['threshold']} ({b['precision']:.3f} / {b['recall']:.3f})")
    save_metrics(run_dir, metrics)


if __name__ == "__main__":
    main()
