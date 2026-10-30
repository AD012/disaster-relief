"""Seeding and run logging shared by experiment scripts.

Each run lives in results/<run_name>/ with config.yaml, run.json (git commit,
versions, device) and metrics.json. Metrics are also copied to
results/metrics/<run_name>.json, which is tracked by git.
"""
import json, platform, random, subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

RESULTS = Path("results")


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def git_info():
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def start_run(cfg, **extra):
    """Create results/<run_name>/ and record config plus provenance. Safe to re-run."""
    import transformers
    run_dir = RESULTS / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    info = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git": git_info(), "device": get_device(), "python": platform.python_version(),
            "torch": torch.__version__, "transformers": transformers.__version__, **extra}
    (run_dir / "run.json").write_text(json.dumps(info, indent=1))
    if info["git"]["dirty"]:
        print("warning: uncommitted changes; run.json commit hash won't match the code exactly")
    return run_dir


def save_metrics(run_dir, metrics):
    run_dir = Path(run_dir)
    text = json.dumps(metrics, indent=1)
    (run_dir / "metrics.json").write_text(text)
    (RESULTS / "metrics").mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics" / f"{run_dir.name}.json").write_text(text)
