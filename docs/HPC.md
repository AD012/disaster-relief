# HPC readiness (Phase 5, no training code yet)

## Unknowns to confirm with the university HPC

Fill in each answer. Nothing below has been guessed.

1. **Scheduler:** Slurm or something else? Partition names?
   Answer: ______________________________

2. **GPU type and memory** (for example model and GB per GPU; GPUs per node):
   Answer: ______________________________

3. **CPU RAM and time limits per job** (memory per node or per job; maximum wall time):
   Answer: ______________________________

4. **Module system and Python/conda setup** (`module load …`? Is Miniforge or conda allowed in home or project space? Which CUDA versions?):
   Answer: ______________________________

5. **Storage location and quota** (home, project, scratch; sizes; scratch purge policy):
   Answer: ______________________________

6. **Internet access for model downloads** (can compute nodes reach Hugging Face, or must `IDEA-Research/grounding-dino-tiny` be pre-downloaded on a login node and pointed to with `HF_HOME`?):
   Answer: ______________________________

7. **Data transfer method** (`scp`, `rsync`, Globus, or another route; any size limits):
   Answer: ______________________________

8. **Job queue limits** (maximum running or queued jobs, GPU-hours per allocation, array-job limits):
   Answer: ______________________________

## What training will plug into

- **Adapter path:** `src/detector.py`, `Detector(adapter=...)`, set from `configs/poc.yaml` under `detector.adapter`.
  - **Head-only state dict** (`torch.save` of the trained head tensors), loaded with `strict=False`: **tested**. 40 head tensors were saved and reloaded, giving identical detections and 0 unexpected keys.
  - **LoRA / PEFT directory** (`adapter_config.json` + weights): **untested** until real LoRA weights exist.
  - Save only the LoRA and head weights, not the full model. The frozen base model is reloaded from `IDEA-Research/grounding-dino-tiny`.
- **The three config switches** (in `configs/poc.yaml`, under `operating_point`):
  - `method`: `crc-frame` (default). `worst-block` exists only for the no-NMS calibration.
  - `nms`: `0.3` (the variant that passed; default) or `null` (plain baseline). Any other τ is refused.
  - `target`: `0.90` (default) or `0.80`.
  - A fine-tuned model needs its own Phase 3 calibration (re-run `src/conformal.py` and `src/conformal_nms.py` on its cached detections). The current calibration files belong to the zero-shot model.
- **Frozen splits and manifests:** `data/manifests/sard/{train,val,calib,test}.json` (block splits with buffers; `splits.json` records the seed and blocks). Image paths are relative to `data/raw/sard-roboflow-tiles/`, so staging means copying that folder and the repo and keeping the relative layout. No re-splitting.
- **Selection rule:** hyperparameters and stopping points are chosen on **validation only**; calib is reserved for conformal calibration; **test is used once**, with the final settings frozen.
