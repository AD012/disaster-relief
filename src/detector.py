"""Grounding DINO person detector with the frozen Phase 2 settings, plus an optional adapter.

    det = Detector(adapter=None)          # zero-shot, identical to baseline.py
    boxes = det.detect(pil_image)         # [{"bbox": [x, y, w, h], "score": s}], score > floor

adapter: a directory with adapter_config.json is loaded as PEFT/LoRA weights;
any other path is a torch state dict (e.g. head weights) loaded with
strict=False, and the missing/unexpected key counts are kept in .adapter_info.
Detections are rounded exactly as baseline.py writes its cache.
"""
from pathlib import Path

import torch
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from baseline import detect as _detect
from utils import get_device

FROZEN = {"model": "IDEA-Research/grounding-dino-tiny", "prompt": "person.", "scale": 1.0,
          "cache_threshold": 0.05, "text_threshold": 0.05}


class Detector:
    def __init__(self, model=FROZEN["model"], prompt=FROZEN["prompt"], scale=FROZEN["scale"],
                 floor=FROZEN["cache_threshold"], text_threshold=FROZEN["text_threshold"], adapter=None, device=None):
        self.prompt, self.scale = prompt, scale
        self.cfg = {"cache_threshold": floor, "text_threshold": text_threshold}
        self.device = device or get_device()
        self.proc = AutoProcessor.from_pretrained(model)
        net = GroundingDinoForObjectDetection.from_pretrained(model)
        self.adapter_info = None
        if adapter:
            net, self.adapter_info = load_adapter(net, Path(adapter))
        self.model = net.to(self.device).eval()

    def detect(self, image):
        res = _detect(self.model, self.proc, image, self.prompt, self.scale, self.cfg, self.device)
        return [{"bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)], "score": round(s, 4)}
                for (x0, y0, x1, y1), s in zip(res["boxes"].tolist(), res["scores"].tolist())]


def load_adapter(net, path):
    if (path / "adapter_config.json").exists():
        from peft import PeftModel
        return PeftModel.from_pretrained(net, path), {"type": "peft", "path": str(path)}
    state = torch.load(path, map_location="cpu")
    missing, unexpected = net.load_state_dict(state, strict=False)
    return net, {"type": "state_dict", "path": str(path), "loaded": len(state) - len(unexpected),
                 "unexpected": len(unexpected), "missing": len(missing)}
