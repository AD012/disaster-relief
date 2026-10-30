"""Run zero-shot Grounding DINO once per (prompt, scale) and cache raw detections.

Usage:
    python src/baseline.py --config configs/baseline.yaml

Detections above cache_threshold are saved in COCO results format to
results/<run_name>/cache/<variant>.json. Existing cache files are skipped, so an
interrupted run resumes. Score the cache with src/evaluate.py.
"""
import argparse, json, random, re, time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from utils import get_device, load_config, set_seed, start_run


def variant_name(prompt, scale):
    return f"{re.sub(r'[^a-z0-9]+', '-', prompt.lower()).strip('-')}_s{scale:g}"


def select_images(cfg):
    coco = json.loads((Path(cfg["manifests"]) / f"{cfg['split']}.json").read_text())
    images = sorted(coco["images"], key=lambda im: im["id"])
    if cfg["subset"]:
        images = sorted(random.Random(cfg["seed"]).sample(images, cfg["subset"]), key=lambda im: im["id"])
    return images


@torch.no_grad()
def detect(model, proc, img, prompt, scale, cfg, dev):
    size = {"shortest_edge": round(800 * scale), "longest_edge": round(1333 * scale)}
    inputs = proc(images=img, text=prompt, return_tensors="pt", size=size).to(dev)
    out = model(**inputs)
    return proc.post_process_grounded_object_detection(
        out, inputs.input_ids, threshold=cfg["cache_threshold"], text_threshold=cfg["text_threshold"],
        target_sizes=[img.size[::-1]])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    cfg = load_config(ap.parse_args().config)

    set_seed(cfg["seed"])
    images = select_images(cfg)
    run_dir = start_run(cfg, image_ids=[im["id"] for im in images])
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    dev = get_device()
    proc = AutoProcessor.from_pretrained(cfg["model"])
    model = GroundingDinoForObjectDetection.from_pretrained(cfg["model"]).to(dev).eval()

    timings = {}
    for prompt in cfg["prompts"]:
        for scale in cfg["scales"]:
            name = variant_name(prompt, scale)
            out_path = cache_dir / f"{name}.json"
            if out_path.exists():
                print(f"{name}: cached, skipping")
                continue
            dets, t0 = [], time.perf_counter()
            for i, im in enumerate(images, 1):
                img = Image.open(Path(cfg["data_root"]) / im["file_name"]).convert("RGB")
                res = detect(model, proc, img, prompt, scale, cfg, dev)
                for (x0, y0, x1, y1), s in zip(res["boxes"].tolist(), res["scores"].tolist()):
                    dets.append({"image_id": im["id"], "category_id": 1, "score": round(s, 4),
                                 "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)]})
                if i % 50 == 0 or i == len(images):
                    print(f"{name}: {i}/{len(images)} ({(time.perf_counter() - t0) / i:.2f} s/img)")
            timings[name] = round((time.perf_counter() - t0) / len(images), 3)
            out_path.write_text(json.dumps(dets))

    if timings:
        timing_path = run_dir / "timings.json"
        old = json.loads(timing_path.read_text()) if timing_path.exists() else {}
        timing_path.write_text(json.dumps({**old, **timings}, indent=1))


if __name__ == "__main__":
    main()
