"""Phase 4 proof-of-concept: replay ordered frames as a stream through the calibrated detector.

Usage:
    python src/pipeline.py --config configs/poc.yaml

Per frame: detect on each tile -> optional NMS -> keep score >= lambda -> inflate
kept boxes by the calibrated margin -> flag the frame if anything is kept.
lambda and the margin come from the Phase 3 calibration file for the configured
method, target and NMS setting. Writes, under results/<run_name>/:
  frames/<frame>.json   kept boxes, scores, inflated boxes, flagged
  annotated/<frame>.jpg tiles side by side with kept (green) and inflated (cyan) boxes
  summary.json          operating point (method, lambda, tau, calibration sha256, git commit),
                        throughput, consistency check, and per-block recall when labels exist
"""
import argparse, hashlib, json, time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from conformal import BLOCK, apply_threshold
from conformal_nms import nms
from detector import Detector
from evaluate import match_pairs, qualifies
from utils import load_config, start_run

TILE = 320


def load_operating_point(op):
    path = Path(op["calibration_nms"] if op["nms"] is not None else op["calibration_plain"])
    data = json.loads(path.read_text())
    target = f"{op['target']:.2f}"
    if op["nms"] is not None:
        if op["method"] != "crc-frame" or not data["passed"] or float(data["chosen_tau"]) != float(op["nms"]):
            raise SystemExit(f"NMS tau {op['nms']} is not the passed, chosen variant in {path}")
        r = data["confirm"][target][f"nms{op['nms']:g}"]["val"]
    else:
        r = next(x for x in data["targets"] if abs(x["target"] - op["target"]) < 1e-9)[op["method"]]
    if r.get("lambda") is None:
        raise SystemExit(f"{op['method']} has no lambda at target {target} in {path}")
    return {"method": op["method"] + (f" + NMS tau={op['nms']:g}" if op["nms"] is not None else ""),
            "target": op["target"], "alpha": round(1 - op["target"], 4), "nms_tau": op["nms"],
            "lambda": r["lambda"], "box_margin": r["box_margin"], "calibration_file": str(path),
            "calibration_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "calibration_git": data["git"]}


def frames_from_manifest(inp):
    coco = json.loads(Path(inp["manifest"]).read_text())
    gts = defaultdict(list)
    for a in coco["annotations"]:
        gts[a["image_id"]].append(a)
    order = []
    for blk in inp["blocks"]:
        lo, hi = map(int, blk.split("-"))
        tiles = defaultdict(list)
        for im in coco["images"]:
            n = int(im["source_frame"][3:])
            if lo <= n <= hi:
                tiles[n].append(im)
        for n in sorted(tiles):
            order.append({"frame": f"gss{n}", "block": blk,
                          "tiles": [{"image": im, "path": Path(inp["data_root"]) / im["file_name"], "gts": gts[im["id"]]}
                                    for im in sorted(tiles[n], key=lambda im: im["id"])]})
    return order


def frames_from_folder(folder):
    return [{"frame": p.stem, "block": "folder", "tiles": [{"image": None, "path": p, "gts": None}]}
            for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png")]


def process_tile(det, img, op):
    t0 = time.perf_counter()
    raw = sorted(det.detect(img), key=lambda d: -d["score"])
    seconds = time.perf_counter() - t0
    cand = nms(raw, op["nms_tau"]) if op["nms_tau"] is not None else raw
    kept = apply_threshold(cand, op["lambda"], op["box_margin"], img.width, img.height)
    return raw, kept, seconds


def draw_tile(img, kept, gts=None, labels=None, font=None):
    """Kept boxes green (red if a labelled false box), inflated boxes cyan, missed labels yellow."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    if gts is not None:
        found = person_found(kept, gts)
        for g, f in zip([g for g in gts if not g["iscrowd"]], found):
            if not f:
                x, y, w, h = g["bbox"]
                draw.rectangle([x, y, x + w, y + h], outline="yellow", width=3)
    for i, d in enumerate(kept):
        x, y, w, h = d["inflated_bbox"]
        draw.rectangle([x, y, x + w, y + h], outline="cyan", width=1)
        colour = "red" if labels is not None and labels[i] == "fp" else "lime"
        x, y, w, h = d["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=colour, width=3)
        draw.text((x + 2, max(y - 16, 0)), f"{d['score']:.2f}", fill=colour, font=font)
    return img


def person_found(kept, gts):
    people = [g for g in gts if not g["iscrowd"]]
    if not people:
        return []
    if not kept:
        return [False] * len(people)
    _, ok = qualifies(np.array([d["bbox"] for d in kept]), np.array([g["bbox"] for g in people]), "iou30")
    return [bool(v) for v in ok.any(axis=0)]


def strip(tiles, font, caption):
    out = Image.new("RGB", (TILE * len(tiles), TILE + 22), "black")
    for i, t in enumerate(tiles):
        out.paste(t.resize((TILE, TILE)), (TILE * i, 22))
    ImageDraw.Draw(out).text((6, 3), caption, fill="white", font=font)
    return out


def block_metrics(records):
    """Per-block frame-averaged and pooled recall, false boxes per tile (labels required)."""
    per = defaultdict(lambda: {"frame_fracs": [], "found": 0, "people": 0, "fp": 0, "tiles": 0, "frames": 0})
    for r in records:
        if r["found"] is None:
            continue
        b = per[r["block"]]
        b["frames"] += 1
        b["tiles"] += r["n_tiles"]
        b["fp"] += r["fp"]
        if r["people"]:
            b["frame_fracs"].append(r["found"] / r["people"])
            b["found"] += r["found"]
            b["people"] += r["people"]
    out = {}
    for blk, b in per.items():
        fr = float(np.mean(b["frame_fracs"])) if b["frame_fracs"] else None
        out[blk] = {"frames": b["frames"], "people": b["people"], "frame_recall": round(fr, 4) if fr is not None else None,
                    "recall": round(b["found"] / b["people"], 4) if b["people"] else None,
                    "fp_per_tile": round(b["fp"] / b["tiles"], 4),
                    "meets_0.80": fr is not None and fr >= 0.80, "meets_0.90": fr is not None and fr >= 0.90}
    return out


def consistency(cache_path, records, op, floor):
    """Kept detections must equal the Phase 3 cache after the same NMS and threshold."""
    cache = defaultdict(list)
    for d in json.loads(Path(cache_path).read_text()):
        if d["score"] > floor:
            cache[d["image_id"]].append(d)
    tiles = mismatched = raw_mismatched = 0
    for r in records:
        for t in r["tiles"]:
            if t["image_id"] is None:
                continue
            tiles += 1
            ref = sorted(cache[t["image_id"]], key=lambda d: -d["score"])
            ref_kept = [d for d in (nms(ref, op["nms_tau"]) if op["nms_tau"] is not None else ref) if d["score"] >= op["lambda"]]
            same = lambda a, b: len(a) == len(b) and all(
                abs(x["score"] - y["score"]) <= 1e-4 and max(abs(p - q) for p, q in zip(x["bbox"], y["bbox"])) <= 0.02
                for x, y in zip(a, b))
            mismatched += not same(ref_kept, t["detections"])
            raw_mismatched += len(ref) != t["n_raw"]
    return {"cache": str(cache_path), "tiles_checked": tiles, "tiles_with_kept_mismatch": mismatched,
            "tiles_with_raw_count_mismatch": raw_mismatched, "passed": mismatched == 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/poc.yaml")
    cfg = load_config(ap.parse_args().config)

    op = load_operating_point(cfg["operating_point"])
    run_dir = start_run(cfg, operating_point=op)
    d = cfg["detector"]
    det = Detector(d["model"], d["prompt"], d["scale"], d["floor"], adapter=d["adapter"])
    frames = frames_from_folder(cfg["input"]["folder"]) if cfg["input"].get("folder") else frames_from_manifest(cfg["input"])
    if not frames:
        raise SystemExit("no frames selected (frame ranges may fall in split buffers, or the folder is empty)")
    font = ImageFont.load_default(size=14)
    (run_dir / "frames").mkdir(exist_ok=True)
    (run_dir / "annotated").mkdir(exist_ok=True)

    records, det_seconds, t_start = [], 0.0, time.perf_counter()
    for i, fr in enumerate(frames, 1):
        tiles_out, drawn, found = [], [], 0
        people = fp = 0
        labelled = all(t["gts"] is not None for t in fr["tiles"])
        for t in fr["tiles"]:
            img = Image.open(t["path"]).convert("RGB")
            raw, kept, secs = process_tile(det, img, op)
            det_seconds += secs
            labels = None
            if labelled:
                labels, _, _ = match_pairs(kept, t["gts"], "iou30")
                fp += labels.count("fp")
                f = person_found(kept, t["gts"])
                found += sum(f)
                people += len(f)
            tiles_out.append({"file": str(t["path"]), "image_id": t["image"]["id"] if t["image"] else None,
                              "n_raw": len(raw), "detections": kept})
            if cfg["save_annotated"]:
                drawn.append(draw_tile(img, kept, font=font))
        rec = {"frame": fr["frame"], "block": fr["block"], "flagged": any(t["detections"] for t in tiles_out),
               "n_tiles": len(tiles_out), "tiles": tiles_out, "found": found if labelled else None,
               "people": people if labelled else None, "fp": fp}
        (run_dir / "frames" / f"{fr['frame']}.json").write_text(json.dumps(rec))
        if cfg["save_annotated"]:
            n_kept = sum(len(t["detections"]) for t in tiles_out)
            strip(drawn, font, f"{fr['frame']}  block {fr['block']}  kept {n_kept}  {'FLAGGED' if rec['flagged'] else ''}").save(
                run_dir / "annotated" / f"{fr['frame']}.jpg", quality=88)
        records.append(rec)
        if i % 20 == 0 or i == len(frames):
            print(f"{i}/{len(frames)} frames, {det_seconds / sum(r['n_tiles'] for r in records):.2f} s/tile")

    n_tiles = sum(r["n_tiles"] for r in records)
    summary = {"operating_point": op, "git": json.loads((run_dir / "run.json").read_text())["git"],
               "adapter": det.adapter_info, "frames": len(records), "tiles": n_tiles,
               "detections_kept": sum(len(t["detections"]) for r in records for t in r["tiles"]),
               "flagged_frames": sum(r["flagged"] for r in records),
               "seconds_per_tile_detector": round(det_seconds / n_tiles, 3),
               "seconds_per_tile_total": round((time.perf_counter() - t_start) / n_tiles, 3),
               "per_block": block_metrics(records), "near_duplicate_skipping": "not implemented (optional; skipped)"}
    if cfg.get("consistency_cache"):
        summary["consistency"] = consistency(cfg["consistency_cache"], records, op, d["floor"])
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_block",)}, indent=1))
    print(json.dumps(summary["per_block"], indent=1))


if __name__ == "__main__":
    main()
