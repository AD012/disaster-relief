"""Conformal recall and box guarantees on cached detections (Phase 3).

Usage:
    python src/conformal.py   # defaults: calib and val at floor 0.01 (-f001 runs), test at its frozen 0.05 floor

Recall control is conformal risk control (Angelopoulos et al., 2022) with frames
as the exchangeable unit: the loss for a frame is the fraction of its people
not found, and the threshold is the largest lambda with
(n / (n + 1)) * mean_loss(lambda) + 1 / (n + 1) <= alpha. A person is found if
any kept box (score >= lambda) satisfies the criterion (primary: IoU >= 0.3).
"worst-block" calibrates each calib block separately and keeps the lowest lambda.

The box margin m is the split-conformal quantile of the per-pair inflation
needed for a matched detection to contain its label (one-to-one, IoU >= 0.3).

Writes results/metrics/conformal-baseline.json, results/figures/conformal-recall-vs-fp.png
and docs/results-phase3.md. Ranges are 2.5-97.5% block-bootstrap percentiles;
with 6-7 blocks per split they are descriptive, not inferential.
"""
import argparse, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate import match_pairs, qualifies
from utils import git_info, load_config

TARGETS = [0.70, 0.80, 0.90, 0.95]
CRITERIA = ["iou30", "centre", "iou50"]  # first is primary
METHODS = ["crc-frame", "worst-block"]
BLOCK = 40
BOX_COVERAGE = 0.90


class Split:
    """Labels and cached detections for one run, grouped by tile, frame and block."""

    def __init__(self, run, variant="person_s1"):
        self.run = Path(run)
        cfg = load_config(self.run / "config.yaml")
        self.name, self.floor = cfg["split"], cfg["cache_threshold"]
        coco = json.loads((Path(cfg["manifests"]) / f"{cfg['split']}.json").read_text())
        ids = set(json.loads((self.run / "run.json").read_text())["image_ids"])
        self.images = [im for im in coco["images"] if im["id"] in ids]
        self.gts, self.dets = defaultdict(list), defaultdict(list)
        for a in coco["annotations"]:
            if a["image_id"] in ids:
                self.gts[a["image_id"]].append(a)
        for d in json.loads((self.run / "cache" / f"{variant}.json").read_text()):
            self.dets[d["image_id"]].append(d)
        for ds in self.dets.values():
            ds.sort(key=lambda d: -d["score"])
        self.frame = {im["id"]: im["source_frame"] for im in self.images}
        self.block = {im["id"]: (int(im["source_frame"][3:]) - 1) // BLOCK for im in self.images}


def person_found_scores(split, criterion):
    """Per non-ignored person: (found score, frame, block). Found score = max score of any qualifying box, 0 if none."""
    s, frames, blocks = [], [], []
    for im in split.images:
        people = [g for g in split.gts[im["id"]] if not g["iscrowd"]]
        ds = split.dets[im["id"]]
        if not people:
            continue
        if ds:
            _, ok = qualifies(np.array([d["bbox"] for d in ds]), np.array([g["bbox"] for g in people]), criterion)
            scores = np.array([d["score"] for d in ds])
            best = [float(scores[ok[:, j]].max()) if ok[:, j].any() else 0.0 for j in range(len(people))]
        else:
            best = [0.0] * len(people)
        s += best
        frames += [split.frame[im["id"]]] * len(people)
        blocks += [split.block[im["id"]]] * len(people)
    return np.array(s), np.array(frames), np.array(blocks)


def calibrate_crc(s, frames, alpha):
    """Largest lambda with (n/(n+1)) * mean frame miss rate + 1/(n+1) <= alpha; None if unreachable."""
    uniq, inv, counts = np.unique(frames, return_inverse=True, return_counts=True)
    n = len(uniq)
    w = 1.0 / counts[inv]  # each frame's people share weight 1
    order = np.argsort(s)
    s_sorted, cum_w = s[order], np.cumsum(w[order])
    for lam in np.unique(s[s > 0])[::-1]:
        k = np.searchsorted(s_sorted, lam, side="left")  # people with s < lam are missed
        risk = cum_w[k - 1] / n if k else 0.0
        if n / (n + 1) * risk + 1 / (n + 1) <= alpha:
            return float(lam)
    return None


def calibrate_worst_block(s, frames, blocks, alpha):
    """Calibrate within each block; keep the lowest lambda (None if any block is unreachable)."""
    lams = {int(b): calibrate_crc(s[blocks == b], frames[blocks == b], alpha) for b in np.unique(blocks)}
    return (None if None in lams.values() else min(lams.values())), lams


def kept(split, image_id, lam):
    return [d for d in split.dets[image_id] if d["score"] >= lam]


def margin_needed(d, g):
    """Smallest m such that d grown by m*w left/right and m*h top/bottom contains g."""
    dx, dy, dw, dh = d
    gx, gy, gw, gh = g
    return max(0.0, (dx - gx) / dw, (gx + gw - dx - dw) / dw, (dy - gy) / dh, (gy + gh - dy - dh) / dh)


def inflate(bbox, m, width, height):
    x, y, w, h = bbox
    x0, y0 = max(0.0, x - m * w), max(0.0, y - m * h)
    x1, y1 = min(width, x + w + m * w), min(height, y + h + m * h)
    return [x0, y0, x1 - x0, y1 - y0]


def matched_margins(split, lam):
    """(block, margin needed) for every one-to-one IoU>=0.3 pair among boxes kept at lam."""
    out = []
    for im in split.images:
        ds, gts = kept(split, im["id"], lam), split.gts[im["id"]]
        labels, _, assign = match_pairs(ds, gts, "iou30")
        out += [(split.block[im["id"]], margin_needed(d["bbox"], gts[j]["bbox"]))
                for d, label, j in zip(ds, labels, assign) if label == "tp"]
    return out


def calibrate_box_margin(margins, coverage=BOX_COVERAGE):
    m = np.sort(margins)
    k = math.ceil((len(m) + 1) * coverage)
    return float(m[k - 1]) if k <= len(m) else math.inf


def apply_threshold(dets, lam, margin=0.0, width=None, height=None):
    """Kept detections at lam, each with an inflated box when margin > 0 (the pipeline's view)."""
    out = []
    for d in dets:
        if d["score"] >= lam:
            box = inflate(d["bbox"], margin, width, height) if margin else list(d["bbox"])
            out.append({**d, "inflated_bbox": [round(v, 2) for v in box]})
    return out


def block_stats(split, scores, lam, margin=None):
    """Per-block counts at lam, for pooled metrics and block bootstrap."""
    st = defaultdict(lambda: defaultdict(float))
    for c, (s, frames, blocks) in scores.items():
        found = s >= lam
        for b in np.unique(blocks):
            sel = blocks == b
            st[int(b)][f"people"] = float(sel.sum())
            st[int(b)][f"found_{c}"] = float(found[sel].sum())
            fr = frames[sel]
            uniq = np.unique(fr)
            st[int(b)]["frames"] = float(len(uniq))
            st[int(b)][f"frame_recall_sum_{c}"] = float(sum(found[sel][fr == f].mean() for f in uniq))
    for im in split.images:
        b = split.block[im["id"]]
        ds, gts = kept(split, im["id"], lam), split.gts[im["id"]]
        st[b]["tiles"] += 1
        for c in scores:
            labels, _, _ = match_pairs(ds, gts, c)
            st[b][f"tp_{c}"] += labels.count("tp")
            st[b][f"fp_{c}"] += labels.count("fp")
        if margin is not None:
            infl = [{**d, "bbox": inflate(d["bbox"], margin, im["width"], im["height"])} for d in ds]
            labels, _, assign = match_pairs(infl, gts, "iou30")
            st[b]["fp_inflated"] += labels.count("fp")
            st[b]["tp_inflated"] += labels.count("tp")
            labels, _, assign = match_pairs(ds, gts, "iou30")
            for d, d_infl, label, j in zip(ds, infl, labels, assign):
                if label == "tp":
                    st[b]["pairs"] += 1
                    st[b]["covered"] += margin_needed(d["bbox"], gts[j]["bbox"]) <= margin
                    st[b]["area_orig"] += d["bbox"][2] * d["bbox"][3]  # matched boxes only, after edge clipping
                    st[b]["area_inflated"] += d_infl["bbox"][2] * d_infl["bbox"][3]
    return {b: dict(v) for b, v in st.items()}


def pooled(stats, criteria):
    tot = defaultdict(float)
    for v in stats.values():
        for k, x in v.items():
            tot[k] += x
    out = {}
    for c in criteria:
        tp, fp = tot[f"tp_{c}"], tot[f"fp_{c}"]
        out[c] = {"recall": tot[f"found_{c}"] / tot["people"], "frame_recall": tot[f"frame_recall_sum_{c}"] / tot["frames"],
                  "precision": tp / (tp + fp) if tp + fp else 0.0, "fp_per_tile": fp / tot["tiles"]}
    if tot["pairs"]:
        out["box"] = {"coverage": tot["covered"] / tot["pairs"], "pairs": int(tot["pairs"]),
                      "fp_per_tile_inflated": tot["fp_inflated"] / tot["tiles"],
                      "fp_per_tile_extra": (tot["fp_inflated"] - tot[f"fp_iou30"]) / tot["tiles"],
                      "area_ratio": tot["area_inflated"] / tot["area_orig"]}
    out["counts"] = {"people": int(tot["people"]), "frames": int(tot["frames"]), "tiles": int(tot["tiles"]),
                     "blocks": len(stats)}
    return out


def bootstrap(stats, criteria, n_boot, rng):
    keys = sorted(stats)
    draws = defaultdict(list)
    for _ in range(n_boot):
        sample = {i: stats[k] for i, k in enumerate(rng.choice(keys, size=len(keys)))}
        p = pooled(sample, criteria)
        for c in criteria:
            for m, v in p[c].items():
                draws[f"{c}.{m}"].append(v)
        if "box" in p:
            for m in ("coverage", "fp_per_tile_extra"):
                draws[f"box.{m}"].append(p["box"][m])
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
            for k, v in draws.items()}


def per_block(stats, c="iou30"):
    return {f"{b * BLOCK + 1}-{(b + 1) * BLOCK}": {"people": int(v["people"]), "frames": int(v["frames"]),
                                                   "frame_recall": round(v[f"frame_recall_sum_{c}"] / v["frames"], 4),
                                                   "recall": round(v[f"found_{c}"] / v["people"], 4)}
            for b, v in sorted(stats.items())}


def frame_recall(s, frames, lam):
    found = s >= lam
    return float(np.mean([found[frames == f].mean() for f in np.unique(frames)]))


def lobo(scores_by_split, alpha, target):
    """Leave-one-block-out over the given splits for both methods (primary criterion); diagnostic only."""
    s = np.concatenate([x[0] for x in scores_by_split.values()])
    frames = np.concatenate([x[1] for x in scores_by_split.values()])
    keys = np.concatenate([[f"{name} {b * BLOCK + 1}-{(b + 1) * BLOCK}" for b in x[2]] for name, x in scores_by_split.items()])
    own = {k: calibrate_crc(s[keys == k], frames[keys == k], alpha) for k in np.unique(keys)}
    out = {}
    for method in METHODS:
        folds = {}
        for k in np.unique(keys):
            if method == "crc-frame":
                lam = calibrate_crc(s[keys != k], frames[keys != k], alpha)
            else:
                others = [v for j, v in own.items() if j != k]
                lam = None if None in others else min(others)
            held = keys == k
            folds[k] = {"lambda": lam, "people": int(held.sum()),
                        "frame_recall": None if lam is None else round(frame_recall(s[held], frames[held], lam), 4)}
        recs = {k: f["frame_recall"] for k, f in folds.items() if f["frame_recall"] is not None}
        short = [target - r for r in recs.values() if r < target]
        lams = [f["lambda"] for f in folds.values() if f["lambda"] is not None]
        out[method] = {"folds": folds, "n_blocks": len(folds), "n_reachable": len(recs),
                       "n_meeting": sum(r >= target for r in recs.values()),
                       "median_shortfall_below": round(float(np.median(short)), 4) if short else 0.0,
                       "lambda_range": [min(lams), max(lams)] if lams else None,
                       "worst": min(recs, key=recs.get) if recs else None}
    return out


def fp_breakdown(split, lam):
    """False boxes (one-to-one, IoU >= 0.3) by best IoU with any person: duplicate >= 0.3, loose 0.1-0.3, background < 0.1."""
    out = defaultdict(int)
    for im in split.images:
        ds, gts = kept(split, im["id"], lam), split.gts[im["id"]]
        labels, _, _ = match_pairs(ds, gts, "iou30")
        people = np.array([g["bbox"] for g in gts if not g["iscrowd"]])
        out["tiles"] += 1
        for d, label in zip(ds, labels):
            if label != "fp":
                continue
            best = float(qualifies(np.array([d["bbox"]]), people, "iou30")[0].max()) if len(people) else 0.0
            out["duplicate" if best >= 0.3 else "loose" if best >= 0.1 else "background"] += 1
    out["total"] = out["duplicate"] + out["loose"] + out["background"]
    return dict(out)


def curve(split, scores, grid):
    pts = []
    for lam in grid:
        p = pooled(block_stats(split, {"iou30": scores["iou30"]}, lam), ["iou30"])["iou30"]
        pts.append({"lambda": round(float(lam), 3), "frame_recall": round(p["frame_recall"], 4),
                    "recall": round(p["recall"], 4), "fp_per_tile": round(p["fp_per_tile"], 4)})
    return pts


def plot(curves, marks, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = {"val": "#2a78d6", "test": "#eb6834"}  # categorical slots 1-2 (validated default palette)
    ink, ink2, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=surface)
    ax.set_facecolor(surface)
    for name, pts in curves.items():
        ax.plot([p["fp_per_tile"] for p in pts], [p["frame_recall"] for p in pts], color=colours[name], lw=2, label=f"SARD {name}")
        for m in marks[name]:
            ax.plot(m["fp_per_tile"], m["frame_recall"], "o", ms=8, color=colours[name], mec=surface, mew=2, zorder=3)
            ax.annotate(f"{m['target']:.2f}", (m["fp_per_tile"], m["frame_recall"]), textcoords="offset points",
                        xytext=(8, -14) if name == "test" else (-30, 6), fontsize=8, color=ink2)
    for t in TARGETS:
        ax.axhline(t, color="#d9d8d3", lw=1, zorder=0)
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlim(-0.005, 60)
    ax.set_xticks([0, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50])
    ax.set_xticklabels(["0", "0.05", "0.1", "0.5", "1", "2", "5", "10", "50"])
    ax.set_xlabel("False boxes per tile (symmetric log scale)", color=ink2)
    ax.set_ylabel("Frame-averaged recall (found: IoU ≥ 0.3)", color=ink2)
    ax.set_title("Recall vs false boxes, zero-shot Grounding DINO (crc-frame)\n"
                 "Dots: thresholds calibrated on calib for each target recall (labels)", fontsize=10, color=ink, loc="left")
    ax.tick_params(colors=ink2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ad")
    ax.legend(frameon=False, labelcolor=ink)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=surface)


def fmt(v, r=None):
    if v is None:
        return "n/a"
    s = f"{v:.3f}"
    return f"{s} [{r[0]:.3f}, {r[1]:.3f}]" if r else s


def write_doc(result, out):
    ev = result["eval"]
    lines = ["# Phase 3 results: conformal recall control on the zero-shot baseline", "",
             f"Generated by `src/conformal.py` from commit `{result['git']['commit'][:7]}` (dirty={result['git']['dirty']}). "
             f"Calib: `{result['calib']}` (floor {result['floor']}); "
             + "; ".join(f"{n}: `{e['run']}` (floor {e['floor']})" for n, e in ev.items()) + ".", "",
             "## Definitions", "",
             "- **Found:** a person is found at threshold λ if some kept box (score ≥ λ) has IoU ≥ 0.3 with their label. "
             "Centre-inside and IoU ≥ 0.5 variants use the same any-box rule.",
             "- **Frame-averaged recall (headline):** the mean over frames of the fraction of that frame's people found. "
             "This is the quantity the conformal guarantee controls: expected per-frame miss rate ≤ α = 1 − target.",
             "- **Pooled person recall (secondary):** people found divided by all people. Crowded frames weigh more, so it is usually lower.",
             "- **Precision and false boxes:** one-to-one greedy matching at IoU ≥ 0.3.",
             f"- **Ranges:** 2.5–97.5% block-bootstrap percentiles ({result['n_boot']} resamples of whole 40-frame blocks). "
             f"They are **descriptive**: val has {ev['val']['counts']['blocks']} blocks and test has {ev['test']['counts']['blocks']}.",
             "- **Test is confirmatory only.** Nothing is chosen on test. `worst-block` was introduced *after* the `crc-frame` "
             "test shortfall had been seen, so its test result is not a clean held-out confirmation of that design choice.", ""]
    for method in METHODS:
        lines += [f"## Target vs achieved recall, method `{method}`", "",
                  "| Target | λ | Val frame recall | Val pooled | Val worst block (frame) | Test frame recall | Test pooled | Test worst block (frame) | Test gap (frame) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for row in result["targets"]:
            r = row[method]
            if r["lambda"] is None:
                lines.append(f"| {row['target']:.2f} | unreachable at calib floor | | | | | | | |")
                continue
            cells = []
            for name in ("val", "test"):
                e = r[name]
                if "skipped" in e:
                    cells += ["needs lower-floor run", "", ""]
                    continue
                w = e["worst_block"]
                cells += [fmt(e["iou30"]["frame_recall"], e["ranges"]["iou30.frame_recall"]), f"{e['iou30']['recall']:.3f}",
                          f"{w['frame_recall']:.3f} ({w['frames']}, {w['frames_n']} frames)"]
            t = r["test"]
            gap = "" if "skipped" in t else f"{t['iou30']['frame_recall'] - row['target']:+.3f}"
            lines.append(f"| {row['target']:.2f} | {r['lambda']:.3f} | " + " | ".join(cells) + f" | {gap} |")
        lines += ["", f"Cost at each threshold, method `{method}`:", "",
                  "| Target | Val precision | Val false boxes/tile | Test precision | Test false boxes/tile | Test pooled recall: centre-in / IoU ≥ 0.5 |",
                  "|---|---|---|---|---|---|"]
        for row in result["targets"]:
            r = row[method]
            if r["lambda"] is None:
                continue
            cells = []
            for name in ("val", "test"):
                e = r[name]
                cells += (["needs lower-floor run", ""] if "skipped" in e else
                          [fmt(e["iou30"]["precision"], e["ranges"]["iou30.precision"]),
                           fmt(e["iou30"]["fp_per_tile"], e["ranges"]["iou30.fp_per_tile"])])
            t = r["test"]
            cells.append("" if "skipped" in t else f"{t['centre']['recall']:.3f} / {t['iou50']['recall']:.3f}")
            lines.append(f"| {row['target']:.2f} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## False boxes at the 0.90 target, by type", "",
              "One-to-one matching at IoU ≥ 0.3. Each false box is classified by its best IoU with any person: "
              "duplicate ≥ 0.3 (that person was already matched), loose 0.1–0.3, background < 0.1.", "",
              "| Method | Split | λ | Loose box on a person | Duplicate | Background | Total | Per tile |", "|---|---|---|---|---|---|---|---|"]
    for row in result["targets"]:
        if row["target"] != 0.90:
            continue
        for method in METHODS:
            r = row[method]
            for name in ("val", "test"):
                fb = r.get("fp_breakdown", {}).get(name)
                if fb is None:
                    lines.append(f"| `{method}` | {name} | {'unreachable' if r['lambda'] is None else 'needs lower-floor run'} | | | | | |")
                    continue
                lines.append(f"| `{method}` | {name} | {r['lambda']:.3f} | {fb['loose']} | {fb['duplicate']} | {fb['background']} | "
                             f"{fb['total']} | {fb['total'] / fb['tiles']:.3f} |")
    lines += ["", f"## Box guarantee (target coverage {BOX_COVERAGE:.2f}, method `crc-frame`)", "",
              "| Target | Margin m | Calib pairs | Val coverage | Test coverage | Matched box area × (test) | Test extra false boxes/tile |",
              "|---|---|---|---|---|---|---|"]
    for row in result["targets"]:
        r = row["crc-frame"]
        if r["lambda"] is None or any("skipped" in r[n] for n in ("val", "test")):
            continue
        v, t = r["val"], r["test"]
        lines.append(f"| {row['target']:.2f} | {r['box_margin']:.3f} | {r['box_calib_pairs']} | "
                     f"{fmt(v['box']['coverage'], v['ranges']['box.coverage'])} | {fmt(t['box']['coverage'], t['ranges']['box.coverage'])} | "
                     f"{t['box']['area_ratio']:.2f} | {fmt(t['box']['fp_per_tile_extra'], t['ranges']['box.fp_per_tile_extra'])} |")
    lines += ["", "## Leave-one-block-out over calib + val (diagnostic)", "",
              "Each of the 13 calib and val blocks is held out in turn; λ is calibrated on the other 12 and applied to the held-out block. "
              "Achieved = frame-averaged recall in the held-out block. Median shortfall = the median of (target − achieved) over blocks below target.", "",
              "| Target | Method | λ range over folds | Blocks meeting target | Median shortfall (blocks below) | Worst block |",
              "|---|---|---|---|---|---|"]
    for row in result["targets"]:
        for method in METHODS:
            lb = row["lobo"][method]
            if not lb["n_reachable"]:
                lines.append(f"| {row['target']:.2f} | `{method}` | unreachable | | | |")
                continue
            worst = lb["worst"]
            lines.append(f"| {row['target']:.2f} | `{method}` | {lb['lambda_range'][0]:.3f}–{lb['lambda_range'][1]:.3f} | "
                         f"{lb['n_meeting']} of {lb['n_blocks']}" + (f" ({lb['n_blocks'] - lb['n_reachable']} unreachable)" if lb['n_reachable'] < lb['n_blocks'] else "")
                         + f" | {lb['median_shortfall_below']:.3f} | {lb['folds'][worst]['frame_recall']:.3f} ({worst}) |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=Path, default=Path("results/baseline-calib-f001"))
    ap.add_argument("--eval", type=Path, nargs="+", default=[Path("results/baseline-val-full-f001"), Path("results/baseline-test")])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    calib = Split(args.calib)
    evals = {sp.name: sp for sp in (Split(r) for r in args.eval)}
    calib_scores = {c: person_found_scores(calib, c) for c in CRITERIA}
    eval_scores = {name: {c: person_found_scores(sp, c) for c in CRITERIA} for name, sp in evals.items()}
    s, frames, blocks = calib_scores["iou30"]

    result = {"git": git_info(), "calib": str(args.calib), "eval": {}, "n_boot": args.n_boot, "box_coverage": BOX_COVERAGE,
              "floor": calib.floor, "calib_people_above_floor": round(float(np.mean(s > 0)), 4), "targets": []}
    for name, sp in evals.items():
        result["eval"][name] = {"run": str(sp.run), "floor": sp.floor,
                                "counts": pooled(block_stats(sp, eval_scores[name], 0.99), CRITERIA)["counts"]}
    marks = defaultdict(list)
    for target in TARGETS:
        alpha = round(1 - target, 4)
        lam_wb, lam_blocks = calibrate_worst_block(s, frames, blocks, alpha)
        row = {"target": target, "alpha": alpha}
        for method, lam in [("crc-frame", calibrate_crc(s, frames, alpha)), ("worst-block", lam_wb)]:
            r = {"lambda": lam}
            if method == "worst-block":
                r["per_calib_block_lambda"] = lam_blocks
            if lam is not None:
                margins = [m for _, m in matched_margins(calib, lam)]
                margin = calibrate_box_margin(margins)
                r.update(box_margin=round(margin, 4), box_calib_pairs=len(margins), fp_breakdown={})
                for name, sp in evals.items():
                    if lam < sp.floor - 1e-9:  # detections below this run's floor were never saved
                        r[name] = {"skipped": f"lambda {lam:.3f} below this run's floor {sp.floor}; needs a lower-floor run"}
                        continue
                    st = block_stats(sp, eval_scores[name], lam, margin)
                    p = pooled(st, CRITERIA)
                    blk = per_block(st)
                    worst = min(blk, key=lambda k: blk[k]["frame_recall"])
                    p.update(ranges=bootstrap(st, CRITERIA, args.n_boot, rng), per_block=blk,
                             worst_block={**blk[worst], "frames_n": blk[worst]["frames"], "frames": worst})
                    r[name] = p
                    if target == 0.90:
                        r["fp_breakdown"][name] = fp_breakdown(sp, lam)
                    if method == "crc-frame":
                        marks[name].append({"target": target, "frame_recall": p["iou30"]["frame_recall"],
                                            "fp_per_tile": p["iou30"]["fp_per_tile"]})
            row[method] = r
        lobo_scores = {"calib": calib_scores["iou30"], **({"val": eval_scores["val"]["iou30"]} if "val" in eval_scores else {})}
        row["lobo"] = lobo(lobo_scores, alpha, target)
        result["targets"].append(row)

    result["curves"] = {name: curve(sp, eval_scores[name], np.round(np.arange(sp.floor, 0.71, 0.01), 3))
                        for name, sp in evals.items()}
    plot(result["curves"], marks, Path("results/figures/conformal-recall-vs-fp.png"))

    out = Path("results/metrics/conformal-baseline.json")
    out.write_text(json.dumps(result, indent=1, default=float))
    write_doc(result, Path("docs/results-phase3.md"))
    print(Path("docs/results-phase3.md").read_text())


if __name__ == "__main__":
    main()
