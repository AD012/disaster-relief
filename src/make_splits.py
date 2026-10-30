"""Re-split SARD by contiguous frame blocks to avoid near-duplicate leakage.

Usage:
    python src/make_splits.py --src data/manifests/sard/roboflow --out data/manifests/sard

The Roboflow splits assign frames one at a time, so neighbouring (near-identical)
frames land in different splits. Here frames are grouped into blocks of
--block consecutive frame numbers, whole blocks are assigned to a split, and
--buffer frames are dropped on the non-train side of every boundary between
different splits. Boxes with min side < --tiny px become ignore regions
(iscrowd=1, ignore=1). Writes train/val/calib/test.json and splits.json.
"""
import argparse, json, random, sys
from collections import defaultdict
from pathlib import Path

EVAL_SPLITS = ["test", "calib", "val"]  # filled in this order; the rest is train


def frame_number(source_frame):
    return int(source_frame.removeprefix("gss"))


def load_tiles(src):
    """Merge the Roboflow manifests into {frame_number: [(image, [annotations])]}."""
    frames, categories = defaultdict(list), None
    for split in ["train", "val", "test"]:
        coco = json.loads((src / f"{split}.json").read_text())
        categories = coco["categories"]
        anns = defaultdict(list)
        for a in coco["annotations"]:
            anns[a["image_id"]].append(a)
        for im in coco["images"]:
            frames[frame_number(im["source_frame"])].append((im, anns[im["id"]]))
    return frames, categories


def assign_blocks(frames, block, eval_frac, seed):
    blocks = defaultdict(list)
    for n in frames:
        blocks[(n - 1) // block].append(n)
    tiles = {b: sum(len(frames[n]) for n in ns) for b, ns in blocks.items()}
    target = eval_frac * sum(tiles.values())

    order = sorted(blocks)
    random.Random(seed).shuffle(order)
    counts, split_of = defaultdict(int), {}
    for b in order:
        split = next((s for s in EVAL_SPLITS if counts[s] < target), "train")
        split_of[b] = split
        counts[split] += tiles[b]
    return blocks, split_of


def apply_buffer(blocks, split_of, buffer):
    """Drop the `buffer` frames nearest each boundary between different splits."""
    dropped = set()
    ids = sorted(blocks)
    for a, b in zip(ids, ids[1:]):
        if split_of[a] == split_of[b]:
            continue
        if split_of[b] != "train":  # drop from the later block unless it is train
            dropped.update(sorted(blocks[b])[:buffer])
        else:
            dropped.update(sorted(blocks[a])[-buffer:])
    return dropped


def build_coco(entries, categories, tiny):
    images, annotations = [], []
    for img_id, (im, anns) in enumerate(entries, 1):
        images.append({**im, "id": img_id})
        for a in anns:
            is_tiny = min(a["bbox"][2:]) < tiny
            annotations.append({**a, "id": len(annotations) + 1, "image_id": img_id,
                                "iscrowd": int(is_tiny), "ignore": int(is_tiny)})
    return {"images": images, "annotations": annotations, "categories": categories}


def check_separation(frames_by_split, buffer):
    """Every pair of frames from different splits must be > buffer frame numbers apart."""
    labelled = sorted((n, s) for s, ns in frames_by_split.items() for n in ns)
    gaps = [b[0] - a[0] for a, b in zip(labelled, labelled[1:]) if a[1] != b[1]]
    return min(gaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/manifests/sard/roboflow"))
    ap.add_argument("--out", type=Path, default=Path("data/manifests/sard"))
    ap.add_argument("--block", type=int, default=40)
    ap.add_argument("--buffer", type=int, default=5)
    ap.add_argument("--eval-frac", type=float, default=0.11)
    ap.add_argument("--tiny", type=float, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frames, categories = load_tiles(args.src)
    blocks, split_of = assign_blocks(frames, args.block, args.eval_frac, args.seed)
    dropped = apply_buffer(blocks, split_of, args.buffer)

    frames_by_split = defaultdict(list)
    for b, ns in blocks.items():
        frames_by_split[split_of[b]].extend(n for n in ns if n not in dropped)

    summary = {"seed": args.seed, "block": args.block, "buffer": args.buffer,
               "eval_frac": args.eval_frac, "tiny_min_side": args.tiny,
               "blocks": {str(b): split_of[b] for b in sorted(blocks)},
               "dropped_frames": sorted(dropped), "counts": {}}
    total_tiles = sum(len(t) for t in frames.values())
    for split in ["train", "val", "calib", "test"]:
        entries = [e for n in sorted(frames_by_split[split]) for e in frames[n]]
        coco = build_coco(entries, categories, args.tiny)
        (args.out / f"{split}.json").write_text(json.dumps(coco))
        n_ignore = sum(a["iscrowd"] for a in coco["annotations"])
        summary["counts"][split] = {"frames": len(frames_by_split[split]),
                                    "images": len(coco["images"]),
                                    "boxes": len(coco["annotations"]), "ignored_boxes": n_ignore}
        print(f"{split}: {len(frames_by_split[split])} frames, {len(coco['images'])} tiles "
              f"({len(coco['images']) / total_tiles:.1%}), {len(coco['annotations'])} boxes "
              f"({n_ignore} ignored)")

    dropped_tiles = sum(len(frames[n]) for n in dropped)
    print(f"dropped: {len(dropped)} frames, {dropped_tiles} tiles ({dropped_tiles / total_tiles:.1%})")
    (args.out / "splits.json").write_text(json.dumps(summary, indent=1))

    min_gap = check_separation(frames_by_split, args.buffer)
    print(f"min frame distance between different splits: {min_gap}")
    if min_gap <= args.buffer:
        sys.exit(f"FAIL: splits are only {min_gap} frames apart (need > {args.buffer})")


if __name__ == "__main__":
    main()
