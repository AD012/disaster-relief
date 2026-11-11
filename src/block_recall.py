"""Per-block recall at a fixed threshold, from cached detections.

Usage:
    python src/block_recall.py --lam 0.25 --runs results/baseline-calib-f001 results/baseline-val-full-f001 results/baseline-test

Found = some kept box with IoU >= 0.3. Reports frame-averaged recall (headline)
and pooled person recall per 40-frame block, plus min and counts of blocks at
or above 0.80 and 0.90, for calib+val together and for test.
Writes results/metrics/block-recall-lambda<lam>.json.
"""
import argparse, json
from pathlib import Path

import numpy as np

from conformal import BLOCK, Split, person_found_scores


def block_rows(split, lam):
    s, frames, blocks = person_found_scores(split, "iou30")
    rows = []
    for b in np.unique(blocks):
        m = blocks == b
        fr = frames[m]
        found = s[m] >= lam
        rows.append({"split": split.name, "block": f"{b * BLOCK + 1}-{(b + 1) * BLOCK}", "frames": len(np.unique(fr)),
                     "people": int(m.sum()), "frame_recall": round(float(np.mean([found[fr == f].mean() for f in np.unique(fr)])), 4),
                     "recall": round(float(found.mean()), 4)})
    return rows


def summary(rows):
    fr = [r["frame_recall"] for r in rows]
    worst = min(rows, key=lambda r: r["frame_recall"])
    return {"blocks": len(rows), "min_frame_recall": min(fr), "worst": f"{worst['split']} {worst['block']}",
            "n_ge_0.80": sum(v >= 0.80 for v in fr), "n_ge_0.90": sum(v >= 0.90 for v in fr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=0.25)
    ap.add_argument("--runs", type=Path, nargs="+",
                    default=[Path("results/baseline-calib-f001"), Path("results/baseline-val-full-f001"), Path("results/baseline-test")])
    args = ap.parse_args()

    rows = [r for run in args.runs for r in block_rows(Split(run), args.lam)]
    groups = {"calib+val": [r for r in rows if r["split"] in ("calib", "val")], "test": [r for r in rows if r["split"] == "test"]}
    result = {"lambda": args.lam, "found": "IoU >= 0.3, any kept box", "rows": rows,
              "summary": {g: summary(v) for g, v in groups.items() if v}}
    Path(f"results/metrics/block-recall-lambda{args.lam:g}.json").write_text(json.dumps(result, indent=1))

    print(f"| Split | Block (frames) | Frames | People | Frame recall | Pooled recall |\n|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['split']} | {r['block']} | {r['frames']} | {r['people']} | {r['frame_recall']:.3f} | {r['recall']:.3f} |")
    for g, s in result["summary"].items():
        print(f"{g}: {s['blocks']} blocks, min frame recall {s['min_frame_recall']:.3f} ({s['worst']}), "
              f">= 0.80: {s['n_ge_0.80']}, >= 0.90: {s['n_ge_0.90']}")


if __name__ == "__main__":
    main()
