"""Preliminary conformal check: choose a score threshold on calib for a target
recall, then measure the recall it achieves on other splits.

Usage:
    python src/recall_calibration.py --calib results/baseline-calib --eval results/baseline-val-full results/baseline-test

Each non-ignored calib person gets a score s_i = the highest detection score
with IoU >= 0.5 against it (0 if none above the cache floor). For target recall
r, the threshold is the k-th smallest s_i with k = floor((n + 1) * (1 - r)),
the split-conformal quantile. This assumes exchangeable people; tiles within a
40-frame block are correlated, so treat the guarantee as approximate.
"""
import argparse, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate import iou, match_at
from utils import load_config


def load(run, variant):
    cfg = load_config(run / "config.yaml")
    coco = json.loads((Path(cfg["manifests"]) / f"{cfg['split']}.json").read_text())
    image_ids = json.loads((run / "run.json").read_text())["image_ids"]
    gt, dets = defaultdict(list), defaultdict(list)
    for a in coco["annotations"]:
        gt[a["image_id"]].append(a)
    for d in json.loads((run / "cache" / f"{variant}.json").read_text()):
        dets[d["image_id"]].append(d)
    return cfg["split"], image_ids, gt, dets


def person_scores(image_ids, gt, dets):
    scores = []
    for i in image_ids:
        people = [g for g in gt[i] if not g["iscrowd"]]
        if not people:
            continue
        if not dets[i]:
            scores += [0.0] * len(people)
            continue
        ious = iou(np.array([g["bbox"] for g in people]), np.array([d["bbox"] for d in dets[i]]))
        s = np.array([d["score"] for d in dets[i]])
        scores += [float(s[row >= 0.5].max()) if (row >= 0.5).any() else 0.0 for row in ious]
    return np.sort(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--eval", type=Path, nargs="+", required=True)
    ap.add_argument("--variant", default="person_s1")
    ap.add_argument("--targets", type=float, nargs="+", default=[0.7, 0.8, 0.9, 0.95])
    args = ap.parse_args()

    _, calib_ids, calib_gt, calib_dets = load(args.calib, args.variant)
    s = person_scores(calib_ids, calib_gt, calib_dets)
    n = len(s)
    evals = [load(r, args.variant) for r in args.eval]

    rows = []
    print(f"calib: {n} people, {np.mean(s > 0):.3f} have a detection >= cache floor with IoU >= 0.5")
    print(f"{'target':>6} {'thr':>6} | " + " | ".join(f"{e[0]:>5} recall  prec" for e in evals))
    for r in args.targets:
        k = math.floor((n + 1) * (1 - r))
        thr = float(s[k - 1]) if k >= 1 else 0.0
        if thr == 0.0:
            row = {"target": r, "threshold": None, "note": "unattainable above cache floor"}
            print(f"{r:6.2f}    n/a | unattainable: too many calib people have no detection")
        else:
            row = {"target": r, "threshold": round(thr, 4)}
            cells = []
            for split, ids, gt, dets in evals:
                flat = [d for i in ids for d in dets[i]]
                m = match_at(gt, flat, ids, thr)
                row[split] = {"recall": m["recall"], "precision": m["precision"]}
                cells.append(f"{m['recall']:12.3f} {m['precision']:5.3f}")
            print(f"{r:6.2f} {thr:6.3f} | " + " | ".join(cells))
        rows.append(row)

    result = json.dumps({"calib_people": n, "variant": args.variant, "rows": rows}, indent=1)
    (args.calib / "recall_calibration.json").write_text(result)
    (Path("results/metrics") / f"{args.calib.name}-recall-calibration.json").write_text(result)


if __name__ == "__main__":
    main()
