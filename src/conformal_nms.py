"""NMS variant of the Phase 3 conformal baseline (separately labelled; rules in docs/log.md).

Usage:
    python src/conformal_nms.py

Class-agnostic greedy NMS is applied to each tile's cached detections before
thresholding. For each overlap threshold tau, lambda (crc-frame) and the box
margin are recalibrated on calib with NMS applied. tau is chosen on val only,
at target 0.90; test is scored once with the chosen tau as confirmation.
Writes results/metrics/conformal-nms.json and docs/results-phase3-nms.md.
"""
import argparse, copy, json
from pathlib import Path

import numpy as np

from conformal import (CRITERIA, Split, block_stats, bootstrap, calibrate_box_margin, calibrate_crc,
                       matched_margins, person_found_scores, pooled)
from evaluate import iou
from utils import git_info

TAUS = [0.3, 0.4, 0.5, 0.6, 0.7]
TARGETS = [0.80, 0.90]
CROWDED = 3  # frames with at least this many people


def nms(dets, tau):
    """Greedy NMS on score-sorted detections; keeps a box unless a higher-scoring kept box overlaps it with IoU > tau."""
    if not dets:
        return []
    boxes = np.array([d["bbox"] for d in dets])
    ious = iou(boxes, boxes)
    keep, suppressed = [], np.zeros(len(dets), bool)
    for i in range(len(dets)):
        if suppressed[i]:
            continue
        keep.append(dets[i])
        suppressed |= ious[i] > tau
    return keep


def with_nms(split, tau):
    sp = copy.copy(split)
    sp.dets = {k: nms(v, tau) for k, v in split.dets.items()}
    return sp


def crowded_recall(scores, lam):
    """Frame-averaged and pooled recall over frames with >= CROWDED people (primary criterion)."""
    s, frames, _ = scores
    uniq, counts = np.unique(frames, return_counts=True)
    crowded = uniq[counts >= CROWDED]
    m = np.isin(frames, crowded)
    found = s >= lam
    return {"frames": int(len(crowded)), "people": int(m.sum()),
            "frame_recall": float(np.mean([found[frames == f].mean() for f in crowded])),
            "recall": float(found[m].mean())}


def evaluate_variant(calib, split, target, rng, n_boot):
    cs = person_found_scores(calib, "iou30")
    lam = calibrate_crc(cs[0], cs[1], round(1 - target, 4))
    margin = calibrate_box_margin([m for _, m in matched_margins(calib, lam)])
    if lam < split.floor - 1e-9:
        return {"lambda": lam, "skipped": f"lambda below {split.name} floor {split.floor}"}
    scores = {c: person_found_scores(split, c) for c in CRITERIA}
    st = block_stats(split, scores, lam, margin)
    p = pooled(st, CRITERIA)
    out = {"lambda": round(lam, 4), "box_margin": round(margin, 4),
           "frame_recall": p["iou30"]["frame_recall"], "recall": p["iou30"]["recall"],
           "precision": p["iou30"]["precision"], "fp_per_tile": p["iou30"]["fp_per_tile"],
           "box_coverage": p["box"]["coverage"], "crowded": crowded_recall(scores["iou30"], lam), "counts": p["counts"]}
    if n_boot:
        out["ranges"] = bootstrap(st, ["iou30"], n_boot, rng)
    return out


def row_md(label, r, ranges=False):
    if "skipped" in r:
        return f"| {label} | {r['lambda']:.3f} | needs lower-floor run | | | | | |"
    rg = r.get("ranges", {})
    f = lambda k, v: f"{v:.3f} [{rg[k][0]:.3f}, {rg[k][1]:.3f}]" if ranges and k in rg else f"{v:.3f}"
    c = r["crowded"]
    return (f"| {label} | {r['lambda']:.3f} | {f('iou30.frame_recall', r['frame_recall'])} | {r['recall']:.3f} | "
            f"{c['frame_recall']:.3f} ({c['frames']} frames) | {f('iou30.precision', r['precision'])} | "
            f"{f('iou30.fp_per_tile', r['fp_per_tile'])} | {r['box_coverage']:.3f} (m={r['box_margin']:.3f}) |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=Path, default=Path("results/baseline-calib-f001"))
    ap.add_argument("--val", type=Path, default=Path("results/baseline-val-full-f001"))
    ap.add_argument("--test", type=Path, default=Path("results/baseline-test"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    calib, val, test = Split(args.calib), Split(args.val), Split(args.test)

    sweep = {}
    for tau in [None] + TAUS:
        c, v = (calib, val) if tau is None else (with_nms(calib, tau), with_nms(val, tau))
        sweep["none" if tau is None else f"{tau:g}"] = {f"{t:.2f}": evaluate_variant(c, v, t, rng, 0) for t in TARGETS}

    base = sweep["none"]["0.90"]
    eligible = [tau for tau in TAUS if sweep[f"{tau:g}"]["0.90"]["frame_recall"] >= base["frame_recall"] - 0.01
                and sweep[f"{tau:g}"]["0.90"]["crowded"]["frame_recall"] >= base["crowded"]["frame_recall"] - 0.02]
    chosen = min(eligible, key=lambda t: (sweep[f"{t:g}"]["0.90"]["fp_per_tile"], -t)) if eligible else None
    passed = chosen is not None and sweep[f"{chosen:g}"]["0.90"]["fp_per_tile"] <= 0.8 * base["fp_per_tile"]

    confirm = {}
    if chosen is not None:
        for t in TARGETS:
            confirm[f"{t:.2f}"] = {
                "none": {s.name: evaluate_variant(calib, s, t, rng, args.n_boot) for s in (val, test)},
                f"nms{chosen:g}": {s.name: evaluate_variant(with_nms(calib, chosen), with_nms(s, chosen), t, rng, args.n_boot)
                                   for s in (val, test)}}

    result = {"git": git_info(), "taus": TAUS, "crowded_min_people": CROWDED, "sweep_val": sweep, "eligible": eligible,
              "chosen_tau": chosen, "passed": passed, "confirm": confirm}
    Path("results/metrics/conformal-nms.json").write_text(json.dumps(result, indent=1, default=float))

    hdr = ("| Variant | λ | Frame recall | Pooled recall | Frame recall, 3+ people | Precision | False boxes/tile | Box coverage |\n"
           "|---|---|---|---|---|---|---|---|")
    lines = ["# Phase 3 variant: NMS before conformal calibration", "",
             f"Generated by `src/conformal_nms.py` from commit `{result['git']['commit'][:7]}` (dirty={result['git']['dirty']}). "
             "Selection and pass rules were fixed in `docs/log.md` before running. Found = some kept box with IoU ≥ 0.3; "
             "precision and false boxes use one-to-one matching at IoU ≥ 0.3. λ and the box margin are recalibrated on calib "
             "for every variant.", ""]
    for t in TARGETS:
        lines += [f"## Val sweep, target {t:.2f}", "", hdr]
        lines += [row_md("no NMS" if k == "none" else f"NMS τ={k}", v[f"{t:.2f}"]) for k, v in sweep.items()]
        lines.append("")
    lines += ["## Selection", "",
              f"- Eligible τ (val frame recall ≥ no-NMS − 0.01, and 3+-people frame recall ≥ no-NMS − 0.02): {eligible or 'none'}.",
              f"- Chosen τ: **{chosen}**; fewest val false boxes per tile among eligible.",
              f"- Pass rule (≥ 20% fewer val false boxes per tile at target 0.90): **{'passed' if passed else 'failed'}**.", ""]
    for t, block in confirm.items():
        lines += [f"## Confirmation at target {t} (ranges: block bootstrap, descriptive)", "", hdr.replace("| Variant |", "| Variant / split |")]
        for variant, by_split in block.items():
            for name, r in by_split.items():
                lines.append(row_md(f"{'no NMS' if variant == 'none' else 'NMS τ=' + variant[3:]} / {name}", r, ranges=True))
        lines.append("")
    Path("docs/results-phase3-nms.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
