"""Labelled example sheet from a pipeline replay (for the PoC checkpoint / report).

Usage:
    python src/poc_examples.py --run results/poc-replay

Picks one frame per row by a fixed rule, and writes the rule as a caption under
each frame: best frame of the easy block, most false boxes in the first block,
most missed people in the weakest block (shown as is), and best frame of the
weakest block. --note FRAME=TEXT adds an observation under a frame (mark it
unconfirmed when it is). Colours: green = kept box on a person, red = false box,
cyan = inflated box, yellow = missed person. Output: results/figures/poc-examples.jpg
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from evaluate import match_pairs
from pipeline import TILE, draw_tile, strip
from utils import load_config

LEGEND = [("lime", "kept box on a person"), ("red", "false box"), ("cyan", "inflated box (margin m)"), ("yellow", "missed person")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("results/poc-replay"))
    ap.add_argument("--note", action="append", default=[], help="FRAME=TEXT, e.g. gss1337='... (unconfirmed)'")
    args = ap.parse_args()
    notes = dict(n.split("=", 1) for n in args.note)
    cfg = load_config(args.run / "config.yaml")
    coco = json.loads(Path(cfg["input"]["manifest"]).read_text())
    gts = defaultdict(list)
    for a in coco["annotations"]:
        gts[a["image_id"]].append(a)
    recs = [json.loads(p.read_text()) for p in sorted((args.run / "frames").glob("*.json"))]
    by_block = defaultdict(list)
    for r in recs:
        by_block[r["block"]].append(r)
    blocks = cfg["input"]["blocks"]
    easy, hard_fp, weak = blocks[2], blocks[0], blocks[1]
    missed = lambda r: r["people"] - r["found"]
    picks = [(max(by_block[easy], key=lambda r: r["found"]), "best frame of the easy block (best case)"),
             (max(by_block[hard_fp], key=lambda r: (r["fp"], r["found"])), f"most false boxes in {hard_fp} (worst case by rule)"),
             (max(by_block[weak], key=lambda r: (missed(r), r["people"])),
              f"most missed people in {weak} (worst case by rule, shown as is)"),
             (max(by_block[weak], key=lambda r: (r["found"] - missed(r), r["found"])), f"best frame of {weak} (best case)")]

    font = ImageFont.load_default(size=14)
    strips = []
    for r, why in picks:
        tiles = []
        for t in r["tiles"]:
            img = Image.open(t["file"]).convert("RGB")
            labels, _, _ = match_pairs(t["detections"], gts[t["image_id"]], "iou30")
            tiles.append(draw_tile(img, t["detections"], gts[t["image_id"]], labels, font))
        s = strip(tiles, font, f"{r['frame']}  block {r['block']}: found {r['found']}/{r['people']}, false boxes {r['fp']}")
        caption = [f"Selected as: {why}"] + ([f"Note: {notes[r['frame']]}"] if r["frame"] in notes else [])
        cap = Image.new("RGB", (s.width, 6 + 20 * len(caption)), "black")
        for i, line in enumerate(caption):
            ImageDraw.Draw(cap).text((6, 3 + 20 * i), line, fill="#d9d8d3", font=font)
        out_strip = Image.new("RGB", (s.width, s.height + cap.height), "black")
        out_strip.paste(s, (0, 0))
        out_strip.paste(cap, (0, s.height))
        strips.append(out_strip)
    width = max(s.width for s in strips)
    sheet = Image.new("RGB", (width, sum(s.height for s in strips) + 30), "black")
    y = 0
    for s in strips:
        sheet.paste(s, (0, y))
        y += s.height
    draw, x = ImageDraw.Draw(sheet), 8
    for colour, text in LEGEND:
        draw.rectangle([x, y + 8, x + 14, y + 22], outline=colour, width=3)
        draw.text((x + 20, y + 7), text, fill="white", font=font)
        x += 34 + int(draw.textlength(text, font=font))
    out = Path("results/figures/poc-examples.jpg")
    sheet.save(out, quality=90)
    print(out, [(r["frame"], why) for r, why in picks])


if __name__ == "__main__":
    main()
