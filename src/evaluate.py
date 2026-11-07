"""Score cached detections from baseline.py with pycocotools; no inference.

Usage:
    python src/evaluate.py --run results/baseline-val300

For each cache/<variant>.json: COCO AP (50, 50:95, small/medium/large), AR@100,
and precision/recall/F1 at IoU 0.5 for each sweep threshold, plus AP and P/R/F1 under
looser matching (IoU 0.3; detection centre inside the label). Ground-truth boxes
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


CRITERIA = ["iou50", "iou30", "centre"]  # centre: detection centre inside the label box


def qualifies(d_xywh, g_xywh, criterion):
    """(ious, ok) for detections (N,4) vs labels (M,4); ok[i, j] if detection i may match label j."""
    ious = iou(d_xywh, g_xywh)
    if criterion == "centre":
        cx, cy = d_xywh[:, 0] + d_xywh[:, 2] / 2, d_xywh[:, 1] + d_xywh[:, 3] / 2
        ok = ((cx[:, None] >= g_xywh[:, 0]) & (cx[:, None] <= g_xywh[:, 0] + g_xywh[:, 2])
              & (cy[:, None] >= g_xywh[:, 1]) & (cy[:, None] <= g_xywh[:, 1] + g_xywh[:, 3]))
    else:
        ok = ious >= {"iou50": 0.5, "iou30": 0.3}[criterion]
    return ious, ok


def match_pairs(ds, gts, criterion="iou50"):
    """Like match_image, but also returns the matched label index per detection (None if unmatched)."""
    taken = [False] * len(gts)
    if not ds or not gts:
        return ["fp"] * len(ds), taken, [None] * len(ds)
    ious, ok = qualifies(np.array([d["bbox"] for d in ds]), np.array([g["bbox"] for g in gts]), criterion)
    labels, assign = [], []
    for i in range(len(ds)):
        cand = [j for j in np.argsort(-ious[i]) if ok[i, j] and (gts[j]["iscrowd"] or not taken[j])]
        if not cand:
            labels.append("fp")
            assign.append(None)
        elif gts[cand[0]]["iscrowd"]:
            labels.append("ignored")
            assign.append(cand[0])
        else:
            taken[cand[0]] = True
            labels.append("tp")
            assign.append(cand[0])
    return labels, taken, assign


def match_image(ds, gts, criterion="iou50"):
    """Greedy matching of score-sorted detections ds to gts.

    Each detection takes the unmatched (or ignore) label it overlaps most among
    those that satisfy the criterion. Returns one label per detection:
    "tp", "fp" or "ignored" (matched an ignore region), plus a per-gt hit list.
    """
    labels, taken, _ = match_pairs(ds, gts, criterion)
    return labels, taken


def _by_image(dets, thr):
    dets_by_img = defaultdict(list)
    for d in dets:
        if d["score"] >= thr:
            dets_by_img[d["image_id"]].append(d)
    for ds in dets_by_img.values():
        ds.sort(key=lambda d: -d["score"])
    return dets_by_img


def match_at(gt_by_img, dets, image_ids, thr, criterion="iou50"):
    """Precision/recall/F1 at a score threshold; detections on ignore boxes are discarded."""
    dets_by_img = _by_image(dets, thr)
    tp = fp = n_gt = 0
    for img_id in image_ids:
        gts = gt_by_img.get(img_id, [])
        labels, _ = match_image(dets_by_img.get(img_id, []), gts, criterion)
        n_gt += sum(not g["iscrowd"] for g in gts)
        tp += labels.count("tp")
        fp += labels.count("fp")
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / n_gt if n_gt else 0.0
    return {"threshold": thr, "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0, "tp": tp, "fp": fp, "fn": n_gt - tp}


def match_ap(gt_by_img, dets, image_ids, criterion):
    """AP (101-point interpolation, as pycocotools) under any matching criterion, up to 100 dets/image."""
    dets_by_img = _by_image(dets, 0.0)
    scores, hits, n_gt = [], [], 0
    for img_id in image_ids:
        gts = gt_by_img.get(img_id, [])
        ds = dets_by_img.get(img_id, [])[:100]
        labels, _ = match_image(ds, gts, criterion)
        n_gt += sum(not g["iscrowd"] for g in gts)
        for d, label in zip(ds, labels):
            if label != "ignored":
                scores.append(d["score"])
                hits.append(label == "tp")
    if not n_gt:
        return 0.0
    order = np.argsort(-np.array(scores), kind="mergesort")
    tp = np.cumsum(np.array(hits)[order])
    fp = np.cumsum(~np.array(hits)[order])
    rc, pr = tp / n_gt, tp / np.maximum(tp + fp, np.spacing(1))
    pr = np.maximum.accumulate(pr[::-1])[::-1]
    idx = np.searchsorted(rc, np.linspace(0, 1, 101), side="left")
    return round(float(np.mean([pr[i] if i < len(pr) else 0.0 for i in idx])), 4)


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
        # Looser matching, to separate box-style mismatch from missed people; same operating point as best F1.
        by_criterion = {c: {"AP": match_ap(gt_by_img, dets, image_ids, c),
                            "at_best_thr": match_at(gt_by_img, dets, image_ids, best["threshold"], c)}
                        for c in CRITERIA}
        metrics["variants"][cache.stem] = {**coco_stats(gt, dets, image_ids), "best_f1": best, "sweep": sweep,
                                           "by_criterion": by_criterion}

    print(f"{'variant':28} {'AP50':>6} {'AP':>6} {'APs':>6} {'APm':>6} {'AR100':>6} | best F1 @ thr (P / R)")
    for name, m in metrics["variants"].items():
        b = m["best_f1"]
        print(f"{name:28} {m['AP50']:6.3f} {m['AP50:95']:6.3f} {m['AP_small']:6.3f} {m['AP_medium']:6.3f} "
              f"{m['AR100']:6.3f} | {b['f1']:.3f} @ {b['threshold']} ({b['precision']:.3f} / {b['recall']:.3f})")
        for c, r in m["by_criterion"].items():
            a = r["at_best_thr"]
            print(f"  {c:26} AP {r['AP']:.3f} | P {a['precision']:.3f} R {a['recall']:.3f} F1 {a['f1']:.3f}")
    save_metrics(run_dir, metrics)


if __name__ == "__main__":
    main()
