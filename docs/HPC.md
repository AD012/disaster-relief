# HPC readiness (Phase 5)

Training code: `src/train.py`, configs `configs/p5_head.yaml` and `configs/p5_lora.yaml`, jobs in `slurm/`.
Placeholders in the `.sbatch` files are marked `TODO-HPC`; each maps to an open item below.

## Known

- **Clusters:** UT Dallas **Ganymede2** and **Juno**.
- **Scheduler:** Slurm.
- **Python:** Miniconda through `module load miniconda/4.12.0`. No Docker. The env is the conda **prefix** `$DR_ROOT/env`, not a named env (see [Environment](#environment)).
- **Queues:** preempt queues; jobs can be killed and requeued. `src/train.py --resume auto` and `slurm/train.sbatch` (`--requeue`, signal forwarding, exit code 75 → `scontrol requeue`) handle this.
- **Access:** campus network or UTD VPN required.
- **Partition:** `gpu-preempt`, with `--requeue`.
- **GPU request:** the `.sbatch` files currently request `--gres=gpu:nvidia_geforce_rtx_3090:1` (24 GB). `--gres=gpu:nvidia_a100-sxm4-80gb:1` is the alternative.
  - The 3090 may need a smaller batch (`--set train.batch_size=2 --set train.grad_accum=…`); the configs still say batch 4, accum 1.
- **Project root:** `DR_ROOT=/mfs/io/groups/zare/Xander/disaster-relief`. Checkpoints and logs go under `runs/` and `logs/` there, **never under scratch**.

## Paths (`DR_ROOT`)

No absolute paths in code or configs. Everything that is not versioned resolves under `$DR_ROOT`.
- **Where `DR_ROOT` comes from:** the env var. Default: the repo's parent folder. The `.sbatch` files set it from the submit folder's parent.
- **Layout:**

```
$DR_ROOT/
  repo/       this git checkout (manifests, configs and results/metrics resolve here)
  data/       raw/<dataset>/ and processed/<dataset>/ (exact tree: Data layout below)
  env/        conda prefix env (Python 3.11, torch 2.6.0+cu124, pip)
  hf_cache/   HF_HOME (train.py uses it automatically if the folder exists)
  runs/       <run>/: checkpoints/, export/, log.csv, run.json, config.yaml
  logs/       Slurm stdout/stderr (<job-name>-<jobid>.out)
```

- **Same configs on the Mac and the HPC:** relative paths mean `run.json` records the resolved paths (`paths`), and the config hash doesn't change between machines.
- **Mac:** the repo's parent is `~/Documents/Code`, which has no `data/`. Either `export DR_ROOT=~/Documents/Code/disaster-relief` (data/ and a git-ignored runs/ inside the repo), or create a dedicated root with the layout above. `train.py` stops with a clear message if `data_root` is not found.
- **Symlinked repo:** set `DR_ROOT` explicitly. Python sees the physical path, so the default would be the link target's parent.

## Data layout

Image path = `$DR_ROOT / <data_root from the config> / <file_name from the manifest>`.
- `data_root` comes from the configs: `data/raw/sard-roboflow-tiles`, `data/processed/{heridal,visdrone}`.
- `file_name` comes from the manifests in the repo.
- Same rule in `src/train.py`, `tests/test_adapter.py`, `slurm/smoke.sbatch` and `tests/check_data.py`, all through `under_root()` in `src/utils.py`.
- Every manifest needs exactly this tree under `$DR_ROOT/data/`. Folders were listed from the manifests' `file_name` values with `python tests/check_data.py -v`.

```
$DR_ROOT/data/
  raw/                                   <- keep this level
    sard-roboflow-tiles/                 REQUIRED for training (~654 MB)
      train/images/                      all three are needed even for training: the frozen
      valid/images/                      splits mix Roboflow's folders (sard/train.json starts
      test/images/                       in valid/images, sard/val.json in train/images)
    heridal/images/test/                 optional: heridal/full_test.json (~771 MB incl. heridal.ndjson)
    visdrone/unzipped/VisDrone2019-DET-val/images/   optional: visdrone/full_val.json (~159 MB incl. the .zip)
  processed/                             optional: zero-shot tile sets
    heridal/tiles300/                    heridal/tiles300.json   (configs/zeroshot_heridal.yaml)
    visdrone/tiles300/                   visdrone/tiles300.json  (configs/zeroshot_visdrone.yaml)
```

**Levels a copy can easily flatten** (each breaks every path below it):
- `raw/` itself. Copying `data/raw/*` into `data/` gives `data/sard-roboflow-tiles`, the layout that failed the first smoke test with 3676 of 3676 train tiles missing.
- `images/` inside each Roboflow split (`train/images/…`, not `train/…`).
- The `train/`, `valid/` and `test/` split folders. Merging them into one folder breaks the manifests, which keep the Roboflow folder in `file_name`.
- `unzipped/VisDrone2019-DET-val/images/`, which is three levels deep, and `heridal/images/test/`.
- `processed/`, which sits beside `raw/`, not inside it.

**Copy from the Mac** (trailing slashes matter: copy the *contents* of `raw/` into `raw/`):

```
rsync -avP --exclude .DS_Store ~/Documents/Code/disaster-relief/data/raw/ <user>@<login-host>:/mfs/io/groups/zare/Xander/disaster-relief/data/raw/
# optional, zero-shot tile sets only:
rsync -avP --exclude .DS_Store ~/Documents/Code/disaster-relief/data/processed/ <user>@<login-host>:/mfs/io/groups/zare/Xander/disaster-relief/data/processed/
```

Check on the login node: `python tests/check_data.py` (stdlib only, seconds; `-v` also shows the rule, first image and folders for each manifest).
- It prints tiles and missing counts per manifest, plus the first 3 missing paths.
- It exits 1 if SARD train or val is incomplete.

## Environment

- **What the env is:** the conda prefix **`$DR_ROOT/env`**, with Python 3.11 and `torch 2.6.0+cu124`, all installed with **pip**. It is not a named env.
- **Activation in jobs and on login nodes** (as in `slurm/*.sbatch`):

```
set +u                        # Lmod and conda activate reference unset variables
module load miniconda/4.12.0
eval "$(conda shell.bash hook)"
conda activate "$DR_ROOT/env"
set -u
export PYTHONNOUSERSITE=1     # ignore ~/.local site-packages
export HF_HOME=$DR_ROOT/hf_cache
export HF_HUB_OFFLINE=1
```

- **No `module load` after activation.** It can reset paths set by the env.
- **Run git with `LD_LIBRARY_PATH= git …`** while the env is active. The env's OpenSSL breaks `git-remote-https` (fetch, pull, push).
- **Submit from `$DR_ROOT/repo`.** The log path `--output=../logs/%x-%j.out` is relative to the submit folder.

## Pre-submit checklist (login node)

1. Env active as above; `python -c "import torch; print(torch.__version__)"` prints `2.6.0+cu124`.
2. `cd $DR_ROOT/repo` and `LD_LIBRARY_PATH= git status`: clean, on the intended commit.
3. **`python tests/check_data.py`**: exits 0, with SARD train and val showing 0 missing.
4. `$DR_ROOT/hf_cache` holds `IDEA-Research/grounding-dino-tiny` (setup step 3).
5. `$DR_ROOT/runs` and `$DR_ROOT/logs` exist.
6. `sbatch slurm/smoke.sbatch` (first time), then `sbatch slurm/train.sbatch configs/<config>.yaml`.

## Unknowns to confirm

Fill in each answer. Nothing below has been guessed.

1. **Partition names:** `gpu-preempt` (known). Still open: is there a shorter debug partition for `smoke.sbatch`? Is the partition the same on Ganymede2 and Juno?
   Answer: ______________________________

2. **GPU:** A100 SXM4 80 GB first, RTX 3090 as fallback (known; both support bf16). Still open: which one to keep in the `--gres` line (TODO-HPC), and how long the A100 queue waits.
   Answer: ______________________________

3. **CPU RAM and time limits per job** (memory per job; maximum wall time on the preempt partition; grace time before a preempted job is killed) → `--time`, `--mem` (TODO-HPC):
   Answer: ______________________________

4. **Conda and CUDA:** known: prefix env `$DR_ROOT/env`, `torch 2.6.0+cu124` from pip (see Environment). Still open: driver version on the 3090/A100 nodes (cu124 must be supported).
   Answer: ______________________________

5. **Storage quota** for `/mfs/io/groups/zare/Xander/disaster-relief` (size, file-count limit, backups). Runs and logs live there, not on scratch:
   Answer: ______________________________

6. **Internet access** (can compute nodes reach Hugging Face? The jobs assume not: `HF_HUB_OFFLINE=1`, model pre-downloaded on a login node into `$HF_HOME`):
   Answer: ______________________________

7. **Data transfer method** (`scp`, `rsync`, Globus, or another route; size limits; works over VPN?):
   Answer: ______________________________

8. **Job queue limits** (maximum running or queued jobs, GPU-hours per allocation, array-job limits; does `scontrol requeue` work for users on the preempt partition?):
   Answer: ______________________________

## Setup on the cluster (once, on a login node)

1. Create the layout: `mkdir -p $DR_ROOT/{data/raw,data/processed,hf_cache,runs,logs}`, with `DR_ROOT=/mfs/io/groups/zare/Xander/disaster-relief`.
   - Copy the repo **including `.git`** to `$DR_ROOT/repo`. This lets `run.json` record the commit and dirty flag; `train.py` refuses a dirty or git-less checkout unless given `--allow-dirty`.
   - Copy the data with the rsync commands in [Data layout](#data-layout), keeping the `raw/` level. Then run `python tests/check_data.py`.
2. Create the prefix env: `conda create -p $DR_ROOT/env python=3.11`.
   - Activate it as in [Environment](#environment).
   - Then `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124`.
   - Then `pip install transformers==5.22.1 peft==0.21.2 pycocotools scipy pyyaml pillow`.
   - Versions are recorded per run in `run.json`.
3. Download the model once: `HF_HOME=$DR_ROOT/hf_cache python -c "from transformers import AutoProcessor, GroundingDinoForObjectDetection as G; m='IDEA-Research/grounding-dino-tiny'; AutoProcessor.from_pretrained(m); G.from_pretrained(m)"`.
4. Go through the [pre-submit checklist](#pre-submit-checklist-login-node), then `cd $DR_ROOT/repo && sbatch slurm/smoke.sbatch`, then check `$DR_ROOT/logs/p5-smoke-<jobid>.out` for `SMOKE PASSED`. The job checks the layout, GPU, versions, that every train and val tile exists, the adapter tests and both dry runs.
   - Submit from `$DR_ROOT/repo`: `--output=../logs/%x-%j.out` is relative to the submit folder, because `#SBATCH` lines cannot expand env vars.

## Running

- `cd $DR_ROOT/repo && sbatch slurm/train.sbatch configs/p5_head.yaml` first, then `configs/p5_lora.yaml`.
- Output in `$DR_ROOT/runs/<config>/`; the Slurm log goes to `$DR_ROOT/logs/p5-train-<jobid>.out`:
  - `checkpoints/last.pt`: full training state, written every epoch and on preemption.
  - `export/best/`, `export/last/`: weights for the detector.
  - `log.csv`: per-epoch train and val loss and val AP50.
  - `run.json`: commit, dirty flag, config hash, versions and restarts.
- A run with a changed config refuses to resume into the same output directory, because the config hash differs.
- Hyperparameter changes go through `--set key=value` (e.g. `--set train.lr=5e-5`) into a new output directory; compare runs on val AP50 only.
- After training:
  - The summary is written inside the repo, at `$DR_ROOT/repo/results/metrics/<run_name>-train.json`. Commit it.
  - The weights stay in `$DR_ROOT/runs/<config>/export/best/`.

## What training plugs into

- **Adapter path:** `src/detector.py`, `Detector(adapter=...)`, set from `configs/poc.yaml` under `detector.adapter`.
  - **Head mode:** `export/best/head.pt`, a state dict loaded with `strict=False`. **Tested** (`tests/test_adapter.py`): untrained weights give identical detections, and perturbed weights change them.
  - **LoRA mode:** `export/best/`, a PEFT directory (`adapter_config.json` + LoRA weights) with `head.pt` alongside. **Tested** the same way. The hook needs a small compatibility shim for PEFT 0.21 with transformers 5.22 (`peft_compat` in `src/detector.py`).
  - Only the head and LoRA weights are saved. The frozen base model is reloaded from `IDEA-Research/grounding-dino-tiny`.
- **What is trained:**
  - **Head:** the box MLP, one set of weights shared by all 6 decoder levels, plus the text projection. 329,476 parameters. The class head itself has no parameters: it is a dot product between image and text features.
  - **LoRA:** r = 8 on the decoder attention and FFN layers, an extra 466,944 parameters.
  - **Encoder losses:** off, because the checkpoint's encoder class logits are unscaled.
  - **Frozen modules** stay in eval mode, with no dropout.
- **The three config switches** (in `configs/poc.yaml`, under `operating_point`):
  - `method`: `crc-frame` (default). `worst-block` exists only for the no-NMS calibration.
  - `nms`: `0.3` (the variant that passed; default) or `null` (plain baseline). Any other τ is refused.
  - `target`: `0.90` (default) or `0.80`.
  - A fine-tuned model needs its own Phase 3 calibration (re-run `src/conformal.py` and `src/conformal_nms.py` on its cached detections). The current calibration files belong to the zero-shot model.
- **Frozen splits and manifests:** `data/manifests/sard/{train,val,calib,test}.json`. `src/train.py` opens only `train` and `val`, and refuses any other split. No re-splitting.
- **Selection rule:** hyperparameters and stopping points are chosen on **validation only** (early stopping: patience 5 epochs, min gain 0.002 val AP50). Epoch 0 is the untrained model scored the same way, so the full-val epoch 0 should reproduce the Phase 2 AP50 of 0.857. Calib is reserved for conformal calibration. **Test is used once**, with the final settings frozen.

## Mac limits (dry run only)

- A training step on MPS needs about 6.9 GB at batch 1 and 12.2 GB at batch 2; batch 2 swaps on the 16 GB M2 Air.
  - The dry run therefore uses batch 1: 20 train and 20 val tiles, 1 epoch.
  - Measured: head about 2–7 min, LoRA about 2–3.5 min, including the preemption and resume check.
- Real training needs the cluster.
