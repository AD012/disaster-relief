#!/bin/bash
# First job on the cluster: environment, GPU, offline model load, data paths, adapter tests and both dry runs.
# Submit FROM $DR_ROOT/repo:   cd $DR_ROOT/repo && sbatch slurm/smoke.sbatch
# Expect "SMOKE PASSED" at the end of $DR_ROOT/logs/p5-smoke-<jobid>.out. Lines marked TODO-HPC: see docs/HPC.md.
#SBATCH --job-name=p5-smoke
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1  # TODO-HPC: fall back to --gres=gpu:nvidia_geforce_rtx_3090:1
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --requeue
#SBATCH --output=../logs/%x-%j.out         # = $DR_ROOT/logs/ when submitted from $DR_ROOT/repo
set -eu

REPO=$SLURM_SUBMIT_DIR
[ -f "$REPO/src/train.py" ] || { echo "submit from the repo folder (\$DR_ROOT/repo), not $REPO"; exit 1; }
export DR_ROOT=${DR_ROOT:-$(cd "$REPO/.." && pwd)}
export HF_HOME=$DR_ROOT/hf_cache
export HF_HUB_OFFLINE=1

set +u   # Lmod and conda activate reference unset variables
module load miniconda/4.12.0
eval "$(conda shell.bash hook)"
conda activate "$DR_ROOT/env"
set -u
export PYTHONNOUSERSITE=1
cd "$REPO"

echo "== host $(hostname), job $SLURM_JOB_ID, DR_ROOT $DR_ROOT"; nvidia-smi
echo "== layout"; for d in data hf_cache runs logs; do [ -d "$DR_ROOT/$d" ] && echo "ok $DR_ROOT/$d" || { echo "missing $DR_ROOT/$d"; exit 1; }; done
echo "== git"; git rev-parse HEAD; git status --porcelain --untracked-files=no
python - <<'EOF'
import torch, transformers, peft
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
print("transformers", transformers.__version__, "peft", peft.__version__)
EOF
echo "== data"; PYTHONPATH=src python - <<'EOF'
import json
from utils import in_repo, under_root, load_config
cfg = load_config("configs/p5_head.yaml")
root = under_root(cfg["data_root"])
for s in ("train", "val"):
    ims = json.loads((in_repo(cfg["manifests"]) / f"{s}.json").read_text())["images"]
    missing = sum(not (root / im["file_name"]).exists() for im in ims)
    print(s, len(ims), "tiles,", missing, "missing under", root)
    assert missing == 0
EOF

echo "== adapter tests"; python -m unittest discover -s tests -v
echo "== dry run (head)"; PY=python bash tests/dry_run.sh configs/p5_head.yaml
echo "== dry run (lora)"; PY=python bash tests/dry_run.sh configs/p5_lora.yaml
echo "SMOKE PASSED"
