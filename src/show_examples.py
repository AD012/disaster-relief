"""Draw detections against labels for a few tiles, as a grid image for the report.

Usage:
    python src/show_examples.py --run results/baseline-test --variant person_s1 --kind tp --one-per-block
    python src/show_examples.py --run results/baseline-test --variant person_s1 --frames 1321-1360 --kind fp

Colours: green = person found (box drawn is the detection, with its score),
yellow = missed person (label box), red = false box (with score), grey = ignore
region. --kind picks the tiles with the most found people (tp), false boxes
(fp) or missed people (fn). --one-per-block keeps at most one tile per 40-frame
block so the sheet is not near-duplicate frames.
Output: results/figures/<run>_<frames>_<kind>.jpg
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from evaluate import match_image
from utils import load_config

TILE, COLS = 480, 3
LEGEND = [("lime", "person found (score)"), ("yellow", "person missed"),
          ("red", "false box (score)"), ("grey", "ignored (tiny)")]


def legend_bar(width, font):
    bar = Image.new("RGB", (width, 36), "black")
    draw, x = ImageDraw.Draw(bar), 12
    for colour, text in LEGEND:
        draw.rectangle([x, 10, x + 16, 26], outline=colour, width=3)
        draw.text((x + 24, 9), text, fill="white", font=font)
        x += 34 + int(draw.textlength(text, font=font))
    return bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--frames", default=None, help="e.g. 1321-1360; default all")
    ap.add_argument("--kind", choices=["tp", "fp", "fn"], default="fp")
    ap.add_argument("--one-per-block", action="store_true")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    cfg = load_config(args.run / "config.yaml")
    coco = json.loads((Path(cfg["manifests"]) / f"{cfg['split']}.json").read_text())
    lo, hi = map(int, args.frames.split("-")) if args.frames else (0, 10 ** 9)
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
        dets = sorted(det_by_img[im["id"]], key=lambda d: -d["score"])
        labels, hit = match_image(dets, gts)
        missed = sum(not h and not g["iscrowd"] for g, h in zip(gts, hit))
        n = {"tp": labels.count("tp"), "fp": labels.count("fp"), "fn": missed}[args.kind]
        if args.kind == "tp":  # prefer clean successes: penalise errors on the same tile
            n -= labels.count("fp") + missed
        scored.append((n, im, gts, dets, labels, hit))
    scored.sort(key=lambda s: (-s[0], s[1]["id"]))
    if args.one_per_block:
        seen, kept = set(), []
        for s in scored:
            block = (int(s[1]["source_frame"][3:]) - 1) // 40
            if block not in seen:
                seen.add(block)
                kept.append(s)
        scored = kept
    chosen = scored[:args.n]

    font = ImageFont.load_default(size=16)
    tiles = []
    for n, im, gts, dets, labels, hit in chosen:
        img = Image.open(Path(cfg["data_root"]) / im["file_name"]).convert("RGB")
        draw = ImageDraw.Draw(img)
        for g, h in zip(gts, hit):
            if g["iscrowd"] or not h:
                x, y, w, h_ = g["bbox"]
                draw.rectangle([x, y, x + w, y + h_], outline="grey" if g["iscrowd"] else "yellow", width=3)
        for d, label in zip(dets, labels):
            if label in ("tp", "fp"):
                colour = "lime" if label == "tp" else "red"
                x, y, w, h_ = d["bbox"]
                draw.rectangle([x, y, x + w, y + h_], outline=colour, width=3)
                draw.text((x + 2, max(y - 18, 0)), f"{d['score']:.2f}", fill=colour, font=font)
        tile = img.resize((TILE, TILE))
        ImageDraw.Draw(tile).text((6, 4), f"{im['source_frame']} ({cfg['split']})", fill="white", font=font,
                                  stroke_width=2, stroke_fill="black")
        tiles.append(tile)

    rows = (len(tiles) + COLS - 1) // COLS
    sheet = Image.new("RGB", (TILE * COLS, TILE * rows + 36), "black")
    for i, t in enumerate(tiles):
        sheet.paste(t, (TILE * (i % COLS), TILE * (i // COLS)))
    sheet.paste(legend_bar(TILE * COLS, font), (0, TILE * rows))
    out = Path("results/figures") / f"{args.run.name}_{args.frames or 'all'}_{args.kind}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)
    print(f"{out}: {len(tiles)} tiles, {args.kind} score per tile {[s[0] for s in chosen]}")


if __name__ == "__main__":
    main()
