"""Per-block AP, block bootstrap and box counts by size, from cached detections.

Usage:
    python src/analyze.py --run results/baseline-test --variant person_s1 --threshold 0.3

Blocks are the 40-frame groups used by make_splits.py. The bootstrap resamples
whole blocks with replacement, since tiles within a block are not independent.
COCOeval runs once; resampled AP is recomputed from its per-image matches with
the same 101-point interpolation as pycocotools.accumulate().
"""
import argparse, contextlib, io, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from evaluate import match_at
from utils import load_config

AREAS = ["all", "small", "medium", "large"]  # COCOeval areaRng order


def size_counts(anns):
    keep = [a["area"] for a in anns if not a["iscrowd"]]
    return {"all": len(keep), "small": sum(a < 32 ** 2 for a in keep),
            "medium": sum(32 ** 2 <= a < 96 ** 2 for a in keep), "large": sum(a >= 96 ** 2 for a in keep)}


class FastAP:
    """Recompute COCO AP for any multiset of images from one COCOeval.evaluate()."""

    def __init__(self, gt, dets, image_ids):
        with contextlib.redirect_stdout(io.StringIO()):
            self.ev = COCOeval(gt, gt.loadRes(dets), "bbox")
            self.ev.params.imgIds = image_ids
            self.ev.evaluate()
        self.n_img = len(self.ev.params.imgIds)
        self.pos = {img_id: i for i, img_id in enumerate(self.ev.params.imgIds)}

    def ap(self, image_ids, area="all", max_det=100):
        """Returns (AP50:95, AP50); None if there are no positives."""
        a = AREAS.index(area)
        es = [self.ev.evalImgs[a * self.n_img + self.pos[i]] for i in image_ids]
        es = [e for e in es if e is not None]
        if not es:
            return None
        scores = np.concatenate([e["dtScores"][:max_det] for e in es])
        order = np.argsort(-scores, kind="mergesort")
        dtm = np.concatenate([e["dtMatches"][:, :max_det] for e in es], axis=1)[:, order]
        dt_ig = np.concatenate([e["dtIgnore"][:, :max_det] for e in es], axis=1)[:, order]
        n_pos = np.count_nonzero(np.concatenate([e["gtIgnore"] for e in es]) == 0)
        if n_pos == 0:
            return None
        tp = np.cumsum(np.logical_and(dtm, ~dt_ig), axis=1).astype(float)
        fp = np.cumsum(np.logical_and(dtm == 0, ~dt_ig), axis=1).astype(float)
        rec_thrs = self.ev.params.recThrs
        aps = []
        for t in range(tp.shape[0]):  # one row per IoU threshold 0.50:0.05:0.95
            rc = tp[t] / n_pos
            pr = (tp[t] / (tp[t] + fp[t] + np.spacing(1))).tolist()
            for i in range(len(pr) - 1, 0, -1):
                pr[i - 1] = max(pr[i - 1], pr[i])
            q = np.zeros(len(rec_thrs))
            idx = np.searchsorted(rc, rec_thrs, side="left")
            for ri, pi in enumerate(idx):
                if pi < len(pr):
                    q[ri] = pr[pi]
            aps.append(q.mean())
        return float(np.mean(aps)), float(aps[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--block", type=int, default=40)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.run / "config.yaml")
    image_ids = json.loads((args.run / "run.json").read_text())["image_ids"]
    manifest = Path(cfg["manifests"]) / f"{cfg['split']}.json"
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(manifest))
    dets = json.loads((args.run / "cache" / f"{args.variant}.json").read_text())

    fast = FastAP(gt, dets, image_ids)
    gt_by_img = defaultdict(list)
    for a in gt.loadAnns(gt.getAnnIds(imgIds=image_ids)):
        gt_by_img[a["image_id"]].append(a)
    blocks = defaultdict(list)
    for i in image_ids:
        blocks[(int(gt.imgs[i]["source_frame"][3:]) - 1) // args.block].append(i)

    def summarize(ids):
        out = {}
        for area in AREAS:
            r = fast.ap(ids, area)
            out[f"AP50:95_{area}"], out[f"AP50_{area}"] = r if r else (None, None)
        m = match_at(gt_by_img, dets, ids, args.threshold)
        out.update(precision=m["precision"], recall=m["recall"], f1=m["f1"])
        return out

    overall = summarize(image_ids)
    per_block = {}
    for b in sorted(blocks):
        ids = blocks[b]
        per_block[f"{b * args.block + 1}-{(b + 1) * args.block}"] = {
            "tiles": len(ids), "boxes": size_counts([a for i in ids for a in gt_by_img[i]]), **summarize(ids)}

    rng = np.random.default_rng(args.seed)
    keys = sorted(blocks)
    boot = defaultdict(list)
    for _ in range(args.n_boot):
        ids = [i for b in rng.choice(keys, size=len(keys)) for i in blocks[b]]
        for k, v in summarize(ids).items():
            if v is not None:
                boot[k].append(v)
    ci = {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
          for k, v in boot.items()}

    all_counts = {}
    for split in ["train", "val", "calib", "test"]:
        anns = json.loads((Path(cfg["manifests"]) / f"{split}.json").read_text())["annotations"]
        all_counts[split] = size_counts(anns)

    result = {"run": args.run.name, "variant": args.variant, "split": cfg["split"],
              "threshold": args.threshold, "n_blocks": len(keys), "n_boot": args.n_boot,
              "overall": overall, "ci95_block_bootstrap": ci, "per_block": per_block,
              "box_counts_eval_images": size_counts([a for i in image_ids for a in gt_by_img[i]]),
              "box_counts_by_split": all_counts}

    print(f"{cfg['split']} ({len(image_ids)} tiles, {len(keys)} blocks), {args.variant} @ {args.threshold}")
    for k in ["AP50_all", "AP50:95_all", "AP50_small", "AP50:95_small", "recall", "precision", "f1"]:
        lo, hi = ci.get(k, [float("nan")] * 2)
        print(f"  {k:15} {overall[k]:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  {'frames':10} {'tiles':>5} {'boxes s/m/l':>12} {'AP50':>6} {'AP':>6} {'APs':>6} {'P':>5} {'R':>5}")
    for name, r in per_block.items():
        c = r["boxes"]
        aps = f"{r['AP50:95_small']:.3f}" if r["AP50:95_small"] is not None else "   n/a"
        print(f"  {name:10} {r['tiles']:5} {c['small']:4}/{c['medium']:3}/{c['large']:3} "
              f"{r['AP50_all']:6.3f} {r['AP50:95_all']:6.3f} {aps:>6} {r['precision']:5.2f} {r['recall']:5.2f}")
    print("  box counts (non-ignored) small/medium/large:",
          {s: f"{c['small']}/{c['medium']}/{c['large']}" for s, c in all_counts.items()})

    out = args.run / f"analysis_{args.variant}.json"
    out.write_text(json.dumps(result, indent=1))
    (Path("results/metrics") / f"{args.run.name}-analysis.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
