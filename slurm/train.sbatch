#!/bin/bash
# Phase 5 training on UT Dallas Ganymede2 / Juno (Slurm). Preemptible: requeues itself and resumes.
# Layout: $DR_ROOT/{repo,data,hf_cache,runs,logs}; HPC DR_ROOT=/mfs/io/groups/zare/Xander/disaster-relief.
# Submit FROM $DR_ROOT/repo (the log path below is relative to the submit dir):
#   cd $DR_ROOT/repo && sbatch slurm/train.sbatch configs/p5_head.yaml
#   cd $DR_ROOT/repo && sbatch slurm/train.sbatch configs/p5_lora.yaml
# Lines marked TODO-HPC still need a value or a decision (docs/HPC.md).
#SBATCH --job-name=p5-train
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1  # TODO-HPC: fall back to --gres=gpu:nvidia_geforce_rtx_3090:1 (24 GB: may need train.batch_size=2 train.grad_accum=2)
#SBATCH --time=TODO-HPC                    # TODO-HPC: wall time within the gpu-preempt limit, e.g. 08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                  # matches train.num_workers in the configs
#SBATCH --mem=32G                          # TODO-HPC: confirm per-job memory limit
#SBATCH --requeue                          # preempted jobs go back in the queue
#SBATCH --signal=B:USR1@300                # warn the batch shell 5 min before the time limit
#SBATCH --output=../logs/%x-%j.out         # = $DR_ROOT/logs/ when submitted from $DR_ROOT/repo
set -u

CONFIG=${1:?usage: sbatch slurm/train.sbatch configs/p5_<head|lora>.yaml}
REPO=$SLURM_SUBMIT_DIR
[ -f "$REPO/src/train.py" ] || { echo "submit from the repo folder (\$DR_ROOT/repo), not $REPO"; exit 1; }
export DR_ROOT=${DR_ROOT:-$(cd "$REPO/.." && pwd)}   # default: the repo's parent folder
export HF_HOME=$DR_ROOT/hf_cache                    # model snapshot downloaded once on a login node
export HF_HUB_OFFLINE=1                             # compute nodes are not assumed to reach the internet
OUT=runs/$(basename "$CONFIG" .yaml)                # -> $DR_ROOT/runs/<config>/ (checkpoints, exports, log.csv, run.json)

module load miniconda/4.12.0
source activate disaster || conda activate disaster       # env created once on a login node (docs/HPC.md)
cd "$REPO" || exit 1

echo "job $SLURM_JOB_ID (restart ${SLURM_RESTART_COUNT:-0}) on $(hostname): $CONFIG -> $DR_ROOT/$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Forward preemption / time-limit signals to train.py, which checkpoints at the next step and exits 75.
python src/train.py --config "$CONFIG" --output-dir "$OUT" --resume auto &
PID=$!
trap 'echo "signal received, asking train.py to checkpoint"; kill -USR1 $PID' USR1 TERM
wait $PID; STATUS=$?
while kill -0 $PID 2>/dev/null; do wait $PID; STATUS=$?; done   # wait again if a trap interrupted wait

if [ "$STATUS" = 75 ]; then
  echo "checkpointed before the end; requeueing"
  scontrol requeue "$SLURM_JOB_ID"
fi
exit $STATUS
