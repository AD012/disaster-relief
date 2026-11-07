"""Check that a lower-floor cache reproduces a higher-floor cache above the higher floor.

Usage:
    python src/compare_caches.py results/baseline-calib/cache/person_s1.json results/baseline-calib-f001/cache/person_s1.json --floor 0.05

Both caches come from the same frozen model, and post-processing keeps boxes with
score > floor, so filtering the new cache at the old floor should give the same
boxes. Boxes match if every coordinate is within --box-tol px and the score is
within --score-tol. Differences beyond tolerance are counted (MPS kernels are
not bit-deterministic across runs).
"""
import argparse, json
from collections import defaultdict

import numpy as np


def by_image(path, floor):
    out = defaultdict(list)
    for d in json.loads(open(path).read()):
        if d["score"] > floor:
            out[d["image_id"]].append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--score-tol", type=float, default=1e-4)
    ap.add_argument("--box-tol", type=float, default=0.02)  # caches round boxes to 0.01 px
    args = ap.parse_args()

    old, new = by_image(args.old, args.floor), by_image(args.new, args.floor)
    matched = unmatched_old = unmatched_new = 0
    max_score_diff = max_box_diff = 0.0
    images_differing = []
    for img in sorted(set(old) | set(new)):
        a, b = old.get(img, []), new.get(img, [])
        used = [False] * len(b)
        bad = 0
        for d in a:
            best, best_j = None, None
            for j, e in enumerate(b):
                if used[j]:
                    continue
                box_diff = float(np.max(np.abs(np.array(d["bbox"]) - np.array(e["bbox"]))))
                score_diff = abs(d["score"] - e["score"])
                if box_diff <= args.box_tol and score_diff <= args.score_tol and (best is None or score_diff < best[1]):
                    best, best_j = (box_diff, score_diff), j
            if best_j is None:
                unmatched_old += 1
                bad += 1
            else:
                used[best_j] = True
                matched += 1
                max_box_diff, max_score_diff = max(max_box_diff, best[0]), max(max_score_diff, best[1])
        extra = used.count(False)
        unmatched_new += extra
        if bad or extra:
            images_differing.append(img)
    total_old = sum(len(v) for v in old.values())
    print(json.dumps({"floor": args.floor, "boxes_old": total_old, "boxes_new_above_floor": sum(len(v) for v in new.values()),
                      "matched": matched, "unmatched_old": unmatched_old, "unmatched_new": unmatched_new,
                      "images_differing": len(images_differing), "max_score_diff": round(max_score_diff, 6),
                      "max_box_diff_px": round(max_box_diff, 4)}, indent=1))


if __name__ == "__main__":
    main()
