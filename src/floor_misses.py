"""People with no qualifying box at a run's score floor: counts per block and a crop sheet.

Usage:
    python src/floor_misses.py --runs results/baseline-calib-f001 results/baseline-val-full-f001 --n-crops 3

A person is a floor miss if no cached box (score above the run's floor) has
IoU >= 0.3 with their label, so no threshold can ever find them. Writes
results/metrics/floor-misses.json and results/figures/floor-misses.jpg (crops
with 1.5x context, upscaled, label box in yellow).
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from conformal import BLOCK, Split, person_found_scores
from utils import load_config


def misses(split):
    """(image, label) pairs for non-ignored people with no qualifying box, in manifest order."""
    s, _, _ = person_found_scores(split, "iou30")
    people = [(im, g) for im in split.images for g in split.gts[im["id"]] if not g["iscrowd"]]
    return [p for p, score in zip(people, s) if score == 0.0]


def frame_ranges(frames):
    nums = sorted({int(f[3:]) for f in frames})
    ranges, start = [], nums[0]
    for a, b in zip(nums, nums[1:] + [None]):
        if b != a + 1:
            ranges.append(f"{start}" if start == a else f"{start}-{a}")
            if b is not None:
                start = b
    return ranges


def crop(cfg_root, im, g, size=220):
    img = Image.open(Path(cfg_root) / im["file_name"]).convert("RGB")
    x, y, w, h = g["bbox"]
    side = max(w, h) * 2.5 + 20
    cx, cy = x + w / 2, y + h / 2
    box = (max(0, cx - side / 2), max(0, cy - side / 2), min(im["width"], cx + side / 2), min(im["height"], cy + side / 2))
    c = img.crop(box)
    ImageDraw.Draw(c).rectangle([x - box[0], y - box[1], x - box[0] + w, y - box[1] + h], outline="yellow", width=2)
    return c.resize((size, round(size * c.height / c.width)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", default=[Path("results/baseline-calib-f001"), Path("results/baseline-val-full-f001")])
    ap.add_argument("--n-crops", type=int, default=3)
    args = ap.parse_args()

    report, picks = {}, []
    for run in args.runs:
        sp = Split(run)
        ms = misses(sp)
        total = sum(not g["iscrowd"] for im in sp.images for g in sp.gts[im["id"]])
        blocks = defaultdict(list)
        for im, g in ms:
            blocks[sp.block[im["id"]]].append((im, g))
        report[sp.name] = {"run": str(run), "floor": sp.floor, "people": total, "floor_misses": len(ms), "per_block": {
            f"{b * BLOCK + 1}-{(b + 1) * BLOCK}": {"floor_misses": len(v), "frames": frame_ranges([im["source_frame"] for im, _ in v]),
                                                   "median_box_wh": sorted((round(g["bbox"][2]), round(g["bbox"][3])) for _, g in v)[len(v) // 2]}
            for b, v in sorted(blocks.items())}}
        for im, g in sorted(ms, key=lambda p: -p[1]["area"])[:max(1, args.n_crops // len(args.runs) + 1)]:
            picks.append((sp, im, g))

    picks = picks[:args.n_crops]
    font = ImageFont.load_default(size=13)
    crops = []
    for sp, im, g in picks:
        c = crop(load_config(sp.run / "config.yaml")["data_root"], im, g)
        ImageDraw.Draw(c).text((4, 2), f"{sp.name} {im['source_frame']}  {round(g['bbox'][2])}x{round(g['bbox'][3])}px",
                               fill="white", font=font, stroke_width=2, stroke_fill="black")
        crops.append(c)
    h = max(c.height for c in crops)
    sheet = Image.new("RGB", (sum(c.width for c in crops) + 6 * (len(crops) - 1), h), "black")
    x = 0
    for c in crops:
        sheet.paste(c, (x, 0))
        x += c.width + 6
    out = Path("results/figures/floor-misses.jpg")
    sheet.save(out, quality=92)
    report["crops"] = [{"split": sp.name, "frame": im["source_frame"], "file": im["file_name"], "bbox": g["bbox"]} for sp, im, g in picks]
    Path("results/metrics/floor-misses.json").write_text(json.dumps(report, indent=1))
    for name, r in report.items():
        if name == "crops":
            continue
        print(f"{name} (floor {r['floor']}): {r['floor_misses']} of {r['people']} people have no box with IoU >= 0.3")
        for blk, v in r["per_block"].items():
            print(f"  block {blk}: {v['floor_misses']} (frames {', '.join(v['frames'])}; median box {v['median_box_wh']})")
    print(out)



if __name__ == "__main__":
    main()
