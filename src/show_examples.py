"""Draw detection errors for tiles in a frame range, as a grid image for the report.

Usage:
    python src/show_examples.py --run results/baseline-test --variant person_s1 --frames 1321-1360 --kind fp

Colours: green = matched person, yellow = missed person, red = false box,
grey = ignore region. --kind picks the tiles with the most false boxes (fp) or
missed people (fn). Output: results/figures/<run>_<frames>_<kind>.jpg
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from evaluate import iou
from utils import load_config


def classify(gts, dets, iou_thr=0.5):
    """Same greedy matching as evaluate.match_at, but returns per-box labels."""
    dets = sorted(dets, key=lambda d: -d["score"])
    det_label, gt_hit = [], [False] * len(gts)
    if dets and gts:
        ious = iou(np.array([d["bbox"] for d in dets]), np.array([g["bbox"] for g in gts]))
    for i in range(len(dets)):
        cand = [] if not gts else [j for j in np.argsort(-ious[i])
                                   if ious[i, j] >= iou_thr and (gts[j]["iscrowd"] or not gt_hit[j])]
        if not cand:
            det_label.append("fp")
        elif gts[cand[0]]["iscrowd"]:
            det_label.append("ignored")
        else:
            gt_hit[cand[0]] = True
            det_label.append("tp")
    return dets, det_label, gt_hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--frames", required=True, help="e.g. 1321-1360")
    ap.add_argument("--kind", choices=["fp", "fn"], default="fp")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    cfg = load_config(args.run / "config.yaml")
    coco = json.loads((Path(cfg["manifests"]) / f"{cfg['split']}.json").read_text())
    lo, hi = map(int, args.frames.split("-"))
    images = [im for im in coco["images"] if lo <= int(im["source_frame"][3:]) <= hi]
    gt_by_img, det_by_img = defaultdict(list), defaultdict(list)
    for a in coco["annotations"]:
        gt_by_img[a["image_id"]].append(a)
    for d in json.loads((args.run / "cache" / f"{args.variant}.json").read_text()):
        if d["score"] >= args.threshold:
            det_by_img[d["image_id"]].append(d)

    scored = []
    for im in images:
        gts = gt_by_img[im["id"]]
        dets, labels, hit = classify(gts, det_by_img[im["id"]])
        n_err = labels.count("fp") if args.kind == "fp" else sum(not h and not g["iscrowd"] for g, h in zip(gts, hit))
        scored.append((n_err, im, gts, dets, labels, hit))
    scored.sort(key=lambda s: (-s[0], s[1]["id"]))

    tiles = []
    for n_err, im, gts, dets, labels, hit in scored[:args.n]:
        img = Image.open(Path(cfg["data_root"]) / im["file_name"]).convert("RGB")
        draw = ImageDraw.Draw(img)
        for g, h in zip(gts, hit):
            x, y, w, h_ = g["bbox"]
            colour = "grey" if g["iscrowd"] else "lime" if h else "yellow"
            draw.rectangle([x, y, x + w, y + h_], outline=colour, width=3)
        for d, label in zip(dets, labels):
            if label == "fp":
                x, y, w, h_ = d["bbox"]
                draw.rectangle([x, y, x + w, y + h_], outline="red", width=3)
                draw.text((x + 2, max(y - 12, 0)), f"{d['score']:.2f}", fill="red")
        draw.text((6, 6), f"{im['source_frame']}  {args.kind}={n_err}", fill="white")
        tiles.append(img.resize((480, 480)))

    cols = 3
    sheet = Image.new("RGB", (480 * cols, 480 * ((len(tiles) + cols - 1) // cols)), "black")
    for i, t in enumerate(tiles):
        sheet.paste(t, (480 * (i % cols), 480 * (i // cols)))
    out = Path("results/figures") / f"{args.run.name}_{args.frames}_{args.kind}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)
    print(f"{out}: {len(tiles)} tiles, errors per tile {[s[0] for s in scored[:args.n]]}")


if __name__ == "__main__":
    main()
