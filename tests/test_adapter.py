"""Untrained exports from src/train.py, loaded through the src/detector.py --adapter hook,
must give exactly the zero-shot detections (head.pt = pretrained head; LoRA B = 0).

    python -m unittest discover -s tests -v

Uses 2 val tiles (never calib or test). Loads the model 3 times; about a minute on the M2 Air.
"""
import json, os, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from PIL import Image

from detector import Detector
from train import build, export
from utils import in_repo, load_config, under_root


def val_tiles(n=2):
    cfg = load_config("configs/p5_head.yaml")
    coco = json.loads((in_repo(cfg["manifests"]) / "val.json").read_text())
    ids_with_people = {a["image_id"] for a in coco["annotations"]}
    ims = [im for im in sorted(coco["images"], key=lambda im: im["id"]) if im["id"] in ids_with_people][:n]
    return [Image.open(under_root(cfg["data_root"]) / im["file_name"]).convert("RGB") for im in ims]


class UntrainedAdapterIsIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.images = val_tiles()
        base = Detector()
        cls.reference = [base.detect(im) for im in cls.images]
        del base
        assert any(cls.reference), "reference tiles produced no detections; the test would be vacuous"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, config, perturb=None):
        cfg = load_config(config)
        model, _, head = build(cfg, "cpu")
        if perturb:
            torch.manual_seed(0)
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if perturb in n:
                        p.add_(torch.randn_like(p) * 0.05)
        path = export(model, head, cfg, Path(self.tmp.name) / f"{cfg['mode']}-{perturb}", {"untrained": not perturb})
        det = Detector(adapter=path)
        got = [det.detect(im) for im in self.images]
        (self.assertNotEqual if perturb else self.assertEqual)(got, self.reference)
        return det.adapter_info

    def test_control_perturbed_head_changes_detections(self):   # the hook really loads head.pt
        self.check("configs/p5_head.yaml", perturb="text_projection")

    def test_control_perturbed_lora_changes_detections(self):   # the hook really loads the LoRA weights
        self.check("configs/p5_lora.yaml", perturb="lora_B")

    def test_head_state_dict(self):
        info = self.check("configs/p5_head.yaml")
        self.assertEqual(info["type"], "state_dict")
        self.assertEqual(info["unexpected"], 0)
        self.assertGreater(info["loaded"], 0)

    def test_lora_dir(self):
        info = self.check("configs/p5_lora.yaml")
        self.assertEqual(info["type"], "peft")
        self.assertEqual(info["head"]["unexpected"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
