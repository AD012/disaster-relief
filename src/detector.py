"""Grounding DINO person detector with the frozen Phase 2 settings, plus an optional adapter.

    det = Detector(adapter=None)          # zero-shot, identical to baseline.py
    boxes = det.detect(pil_image)         # [{"bbox": [x, y, w, h], "score": s}], score > floor

adapter: a directory with adapter_config.json is loaded as PEFT/LoRA weights
(plus head.pt from the same directory, if present); any other path is a torch
state dict (e.g. head weights) loaded with strict=False, and the
missing/unexpected key counts are kept in .adapter_info.
Both formats are what src/train.py exports.
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


def peft_compat():
    """PEFT 0.21 calls get_input_embeddings() on models with dict-style tied weights;
    transformers 5.22 leaves it unimplemented for Grounding DINO. Point it at the BERT word embeddings."""
    if getattr(GroundingDinoForObjectDetection, "_peft_compat", False):
        return
    GroundingDinoForObjectDetection.get_input_embeddings = lambda self: self.model.text_backbone.get_input_embeddings()
    GroundingDinoForObjectDetection._peft_compat = True


def load_state(net, path):
    state = torch.load(path, map_location="cpu")
    missing, unexpected = net.load_state_dict(state, strict=False)
    return net, {"type": "state_dict", "path": str(path), "loaded": len(state) - len(unexpected),
                 "unexpected": len(unexpected), "missing": len(missing)}


def load_adapter(net, path):
    """A directory with adapter_config.json: optional head.pt (fully trained head, loaded first), then PEFT/LoRA.
    Anything else: a head state dict (e.g. head.pt from src/train.py --mode head)."""
    if (path / "adapter_config.json").exists():
        from peft import PeftModel
        peft_compat()
        info = {"type": "peft", "path": str(path)}
        if (path / "head.pt").exists():
            net, info["head"] = load_state(net, path / "head.pt")
        return PeftModel.from_pretrained(net, path), info
    return load_state(net, path)
