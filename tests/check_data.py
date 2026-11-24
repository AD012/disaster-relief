"""Check that every manifest's images exist under $DR_ROOT/data before submitting jobs.

    python tests/check_data.py            # on the login node, from $DR_ROOT/repo; seconds, stdlib only
    DR_ROOT=/path/to/root python tests/check_data.py

Same resolution as src/train.py (src/utils.py: dr_root, under_root, in_repo):
  DR_ROOT = env var, default the repo's parent folder;
  image path = $DR_ROOT / <data_root> / <file_name from the manifest>; manifests are read from the repo.
dr_root() is re-implemented here because src/utils.py imports torch.

Exit 1 if anything required for training (SARD train + val) is missing; other manifests are reported only.
"""
import json, os, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("DR_ROOT") or REPO.parent).expanduser().resolve()


def config_data_root(config):
    """data_root from a config, read with a regex (no yaml in the stdlib)."""
    m = re.search(r"^\s*data_root:\s*([^\s#]+)", (REPO / config).read_text(), re.M)
    return m.group(1).strip("'\"")


SARD = config_data_root("configs/p5_head.yaml")                 # data/raw/sard-roboflow-tiles
# (manifest in the repo, data_root under $DR_ROOT, required for training, where the data_root rule comes from)
CHECKS = [
    ("data/manifests/sard/train.json", SARD, True, "configs/p5_head.yaml"),
    ("data/manifests/sard/val.json", SARD, True, "configs/p5_head.yaml"),
    ("data/manifests/sard/calib.json", SARD, False, "configs/baseline_calib.yaml"),
    ("data/manifests/sard/test.json", SARD, False, "configs/baseline_test.yaml"),
    ("data/manifests/sard/roboflow/train.json", SARD, False, "src/convert_sard.py --root"),
    ("data/manifests/sard/roboflow/val.json", SARD, False, "src/convert_sard.py --root"),
    ("data/manifests/sard/roboflow/test.json", SARD, False, "src/convert_sard.py --root"),
    ("data/manifests/heridal/full_test.json", "data/raw/heridal", False, "src/convert_external.py (RAW)"),
    ("data/manifests/heridal/tiles300.json", config_data_root("configs/zeroshot_heridal.yaml"), False,
     "configs/zeroshot_heridal.yaml"),
    ("data/manifests/visdrone/full_val.json", "data/raw/visdrone", False, "src/convert_external.py (RAW)"),
    ("data/manifests/visdrone/tiles300.json", config_data_root("configs/zeroshot_visdrone.yaml"), False,
     "configs/zeroshot_visdrone.yaml"),
]


def main():
    verbose = "-v" in sys.argv
    print(f"DR_ROOT {ROOT}" + ("" if os.environ.get("DR_ROOT") else "  (default: the repo's parent folder)"))
    failed = False
    for manifest, data_root, required, source in CHECKS:
        tag = "REQUIRED" if required else "optional"
        path = REPO / manifest
        if not path.exists():
            print(f"{tag:8}  {manifest}: manifest not found in the repo")
            failed |= required
            continue
        images = json.loads(path.read_text())["images"]
        base = ROOT / data_root
        missing = [im["file_name"] for im in images if not (base / im["file_name"]).is_file()]
        status = "ok" if not missing else "MISSING"
        print(f"{tag:8}  {manifest}: {len(images)} tiles, {len(missing)} missing -> {base}  [{status}]")
        if verbose:
            first = images[0]["file_name"]
            print(f"          rule: {source}; first image {first} -> {base / first} "
                  f"({'exists' if (base / first).is_file() else 'not found'})")
            dirs = sorted({str(Path(im['file_name']).parent) for im in images})
            print(f"          folders needed under {data_root}/: {', '.join(dirs)}")
        for f in missing[:3]:
            print(f"          missing: {base / f}")
        failed |= required and bool(missing)
    print("FAILED: SARD train/val images missing; training cannot start" if failed else
          "OK: everything required for training (SARD train + val) is present")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
