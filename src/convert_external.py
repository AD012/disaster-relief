"""Convert HERIDAL (Ultralytics NDJSON export) or VisDrone-DET to a full-image COCO manifest.

Usage:
    python src/convert_external.py heridal  --split test
    python src/convert_external.py visdrone --split val

Output: data/manifests/<dataset>/full_<split>.json, with file_name relative to
data/raw/<dataset>. All person classes map to one category, "human" (id 1), to
match SARD. VisDrone: pedestrian (1) and people (2) are persons; ignored
regions (0) become iscrowd=1 boxes; other classes are dropped.
"""
import argparse, json
from pathlib import Path

from PIL import Image

RAW, MANIFESTS = Path("data/raw"), Path("data/manifests")
CATEGORIES = [{"id": 1, "name": "human"}]


def ann(ann_id, img_id, x, y, w, h, crowd=False, **extra):
    return {"id": ann_id, "image_id": img_id, "category_id": 1, "bbox": [x, y, w, h],
            "area": w * h, "iscrowd": int(crowd), "ignore": int(crowd), **extra}


def heridal(split):
    root = RAW / "heridal"
    rows = [json.loads(l) for l in (root / "heridal.ndjson").read_text().splitlines() if l.strip()]
    images, annotations = [], []
    for r in rows:
        if r["type"] != "image" or r["split"] != split:
            continue
        w, h = r["width"], r["height"]
        img_id = len(images) + 1
        images.append({"id": img_id, "file_name": f"images/{split}/{Path(r['file']).name}", "width": w, "height": h})
        for _, cx, cy, bw, bh in (r.get("annotations") or {}).get("boxes", []):
            annotations.append(ann(len(annotations) + 1, img_id, round((cx - bw / 2) * w, 2),
                                   round((cy - bh / 2) * h, 2), round(bw * w, 2), round(bh * h, 2)))
    return images, annotations


def visdrone(split):
    root = RAW / "visdrone" / "unzipped" / f"VisDrone2019-DET-{split}"
    images, annotations = [], []
    for img_path in sorted((root / "images").glob("*.jpg")):
        w, h = Image.open(img_path).size
        img_id = len(images) + 1
        images.append({"id": img_id, "file_name": str(img_path.relative_to(RAW / "visdrone")), "width": w, "height": h})
        for line in (root / "annotations" / f"{img_path.stem}.txt").read_text().splitlines():
            vals = [int(v) for v in line.strip().rstrip(",").split(",")[:8]]
            x, y, bw, bh, _, cat, trunc, occl = vals
            if cat in (1, 2):
                annotations.append(ann(len(annotations) + 1, img_id, x, y, bw, bh,
                                       visdrone_class=cat, truncation=trunc, occlusion=occl))
            elif cat == 0:
                annotations.append(ann(len(annotations) + 1, img_id, x, y, bw, bh, crowd=True))
    return images, annotations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["heridal", "visdrone"])
    ap.add_argument("--split", required=True)
    args = ap.parse_args()

    images, annotations = {"heridal": heridal, "visdrone": visdrone}[args.dataset](args.split)
    out = MANIFESTS / args.dataset / f"full_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"images": images, "annotations": annotations, "categories": CATEGORIES}))
    persons = sum(not a["iscrowd"] for a in annotations)
    print(f"{args.dataset} {args.split}: {len(images)} images, {persons} persons, "
          f"{len(annotations) - persons} ignore regions -> {out}")


if __name__ == "__main__":
    main()
