"""Convert the Roboflow SARD export (YOLO format) to one COCO JSON per split.

Usage:
    python src/convert_sard.py --root data/raw/sard-roboflow-tiles --out data/manifests/sard/roboflow

file_name in the output is relative to --root, so manifests stay valid wherever
the raw data lives. Each image also records its source_frame (e.g. "gss1008"),
the original SARD video frame its tile came from, for grouping by frame.
"""
import argparse, json, re
from pathlib import Path

import yaml
from PIL import Image

SPLITS = {"train": "train", "valid": "val", "test": "test"}  # folder -> data.yaml key


def source_frame(stem):
    return re.sub(r"_jpg\.rf\..*$", "", stem)


def convert_split(root, folder, names):
    images, annotations, stats = [], [], {"empty": 0, "clipped": 0, "dropped": 0}
    for img_id, img_path in enumerate(sorted((root / folder / "images").glob("*.jpg")), 1):
        w, h = Image.open(img_path).size
        images.append({"id": img_id, "file_name": str(img_path.relative_to(root)),
                       "width": w, "height": h, "source_frame": source_frame(img_path.stem)})

        label_path = root / folder / "labels" / f"{img_path.stem}.txt"
        rows = [l.split() for l in label_path.read_text().splitlines() if l.strip()]
        stats["empty"] += not rows
        for cls, cx, cy, bw, bh in rows:
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            x0, y0 = max(cx - bw / 2, 0), max(cy - bh / 2, 0)
            x1, y1 = min(cx + bw / 2, w), min(cy + bh / 2, h)
            if (x1 - x0, y1 - y0) != (bw, bh):
                stats["clipped"] += 1
            if x1 - x0 < 1 or y1 - y0 < 1:
                stats["dropped"] += 1
                continue
            annotations.append({"id": len(annotations) + 1, "image_id": img_id,
                                "category_id": int(cls) + 1,
                                "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
                                "area": round((x1 - x0) * (y1 - y0), 2), "iscrowd": 0})

    categories = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
    return {"images": images, "annotations": annotations, "categories": categories}, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw/sard-roboflow-tiles"))
    ap.add_argument("--out", type=Path, default=Path("data/manifests/sard/roboflow"))
    args = ap.parse_args()

    names = yaml.safe_load((args.root / "data.yaml").read_text())["names"]
    args.out.mkdir(parents=True, exist_ok=True)
    for folder, split in SPLITS.items():
        coco, stats = convert_split(args.root, folder, names)
        out_path = args.out / f"{split}.json"
        out_path.write_text(json.dumps(coco))
        print(f"{split}: {len(coco['images'])} images, {len(coco['annotations'])} boxes, "
              f"{stats['empty']} empty, {stats['clipped']} clipped, {stats['dropped']} dropped -> {out_path}")


if __name__ == "__main__":
    main()
