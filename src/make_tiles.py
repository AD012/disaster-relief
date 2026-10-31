"""Cut native-resolution tiles from a full-image COCO manifest and sample a subset.

Usage:
    python src/make_tiles.py heridal --manifest full_test.json --n 300

Tiles are --tile px squares on a grid whose last row/column is aligned to the
image edge (smaller images give smaller tiles). Boxes are clipped to the tile;
a box with less than --min-visible of its area inside, or a min side below
--tiny px after clipping, becomes an ignore region (iscrowd=1). The sample has
round(n * pos_frac) tiles with at least one person and the rest without.

Writes crops to data/processed/<dataset>/tiles<n>/ and the manifest to
data/manifests/<dataset>/tiles<n>.json (file_name relative to data/processed/<dataset>).
"""
import argparse, json, random
from pathlib import Path

from PIL import Image

RAW, PROCESSED, MANIFESTS = Path("data/raw"), Path("data/processed"), Path("data/manifests")


def grid(length, tile):
    if length <= tile:
        return [0]
    return sorted(set(list(range(0, length - tile, tile)) + [length - tile]))


def tile_boxes(anns, x0, y0, tw, th, min_visible, tiny):
    out = []
    for a in anns:
        x, y, w, h = a["bbox"]
        cx0, cy0 = max(x, x0), max(y, y0)
        cx1, cy1 = min(x + w, x0 + tw), min(y + h, y0 + th)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        cw, ch = cx1 - cx0, cy1 - cy0
        ignore = bool(a["iscrowd"]) or cw * ch < min_visible * w * h or min(cw, ch) < tiny
        out.append({**a, "bbox": [round(cx0 - x0, 2), round(cy0 - y0, 2), round(cw, 2), round(ch, 2)],
                    "area": round(cw * ch, 2), "iscrowd": int(ignore), "ignore": int(ignore)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--manifest", required=True, help="file in data/manifests/<dataset>/")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--pos-frac", type=float, default=0.85)
    ap.add_argument("--min-visible", type=float, default=0.5)
    ap.add_argument("--tiny", type=float, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    coco = json.loads((MANIFESTS / args.dataset / args.manifest).read_text())
    by_img = {im["id"]: [] for im in coco["images"]}
    for a in coco["annotations"]:
        by_img[a["image_id"]].append(a)

    candidates = []
    for im in coco["images"]:
        for y0 in grid(im["height"], args.tile):
            for x0 in grid(im["width"], args.tile):
                tw, th = min(args.tile, im["width"]), min(args.tile, im["height"])
                boxes = tile_boxes(by_img[im["id"]], x0, y0, tw, th, args.min_visible, args.tiny)
                candidates.append((im, x0, y0, tw, th, boxes))
    pos = [c for c in candidates if any(not b["iscrowd"] for b in c[5])]
    neg = [c for c in candidates if not any(not b["iscrowd"] for b in c[5])]
    rng = random.Random(args.seed)
    n_pos = min(round(args.n * args.pos_frac), len(pos))
    chosen = rng.sample(pos, n_pos) + rng.sample(neg, min(args.n - n_pos, len(neg)))
    chosen.sort(key=lambda c: (c[0]["id"], c[2], c[1]))

    name = f"tiles{args.n}"
    out_dir = PROCESSED / args.dataset / name
    out_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for im, x0, y0, tw, th, boxes in chosen:
        src = RAW / args.dataset / im["file_name"]
        file_name = f"{name}/{Path(im['file_name']).stem}_x{x0}_y{y0}.jpg"
        if not (PROCESSED / args.dataset / file_name).exists():
            Image.open(src).convert("RGB").crop((x0, y0, x0 + tw, y0 + th)).save(
                PROCESSED / args.dataset / file_name, quality=95)
        img_id = len(images) + 1
        images.append({"id": img_id, "file_name": file_name, "width": tw, "height": th,
                       "source_image": im["file_name"], "tile_origin": [x0, y0]})
        for b in boxes:
            annotations.append({**b, "id": len(annotations) + 1, "image_id": img_id})

    out = MANIFESTS / args.dataset / f"{name}.json"
    out.write_text(json.dumps({"images": images, "annotations": annotations, "categories": coco["categories"]}))
    persons = sum(not a["iscrowd"] for a in annotations)
    print(f"{args.dataset}: {len(candidates)} candidate tiles ({len(pos)} with people); sampled {len(images)} "
          f"({n_pos} with people) from {len({c[0]['id'] for c in chosen})} images, {persons} persons, "
          f"{len(annotations) - persons} ignored -> {out}")


if __name__ == "__main__":
    main()
