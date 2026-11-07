# Plan: Phase 3 (conformal baseline) and Phase 4 (proof-of-concept pipeline)

Deadline: 1 Dec 2026. Frozen detector settings (not changed anywhere below): `IDEA-Research/grounding-dino-tiny`, prompt `"person."`, scale 1.0×, fp32 on MPS. Raw detections are cached at a 0.05 score floor, so Phase 3 is offline analysis unless a lower floor is needed.

## Facts the design rests on

| Split | Blocks | Frames (all contain people) | People (non-ignored) | People with any box ≥ 0.05 at IoU ≥ 0.3 |
|---|---|---|---|---|
| calib | 7 | 218 | 748 | 97.7% |
| val | 6 | 177 | 860 | 97.6% |
| test | 7 | 210 | 619 | 95.2% |

The Phase 2 per-person calibration hit its targets on val but fell 5–9 points short on test, so people are not exchangeable: they cluster in frames, and frames cluster in scene blocks.

## Phase 3 design

### Matching (decision)
- **Found (recall):** a person is found at threshold λ if *any* kept box (score ≥ λ) overlaps their label with IoU ≥ 0.3. Each person therefore has a single "found score" s = the highest score among boxes that qualify, and is found exactly when s ≥ λ. This makes recall monotone in λ, which conformal control needs.
- **Variants reported beside it:** detection centre inside the label box, and strict IoU ≥ 0.5, using the same "any box" rule.
- **Precision and false boxes:** one-to-one greedy matching (`evaluate.match_image`) under the same criterion. A box is false if it matches no person and no ignore region.

### Recall control (decision: frame is the calibration unit)
- **Method:** conformal risk control (Angelopoulos et al., 2022) with **frames** as the exchangeable units.
  - Loss for frame i at threshold λ: L_i(λ) = fraction of that frame's people not found. It lies in [0, 1] and never decreases as λ rises.
  - Rule: choose the largest λ with (n/(n+1))·R̂(λ) + 1/(n+1) ≤ α, where R̂ is the mean loss over the n calib frames and α = 1 − target.
- **Guarantee:** expected per-frame miss rate ≤ α. It holds if frames are exchangeable, which is closer to the truth than people being exchangeable but still optimistic within blocks.
- **Why not blocks as units?** With 7 calib blocks the 1/(n+1) term is 0.125, so no target above 0.875 is reachable.
- **Second rule, `worst-block`:** run conformal risk control separately in each calib block (about 30 frames each) and take the lowest λ. It uses only calib data, and is conservative against scene shift without touching val or test.
- **Targets:** recall 0.70, 0.80, 0.90 and 0.95 (α = 0.30, 0.20, 0.10, 0.05).
- **Both recall numbers reported:** pooled person recall, and frame-averaged recall (the quantity actually guaranteed).

### Block awareness (descriptive only: 6–7 blocks per split)
- **Per-block recall** on val and test at each calibrated λ, plus the worst block.
- **Block bootstrap:** 2,000 resamples of whole blocks, giving a 2.5–97.5% range for achieved recall, precision and false boxes per tile.
- **Leave-one-block-out (LOBO) over calib + val (13 blocks):** calibrate on 12 blocks, apply to the held-out block, and repeat. This shows how much λ moves with scene and how often a block falls short. It is a diagnostic only: final thresholds come from calib alone.

### Box guarantee (decision)
- **Score per matched pair:** for each calib pair (one-to-one, IoU ≥ 0.3, at the calibrated λ), the smallest margin m such that the detection box, grown by m·w on the left and right and m·h on the top and bottom, contains the labelled box.
- **Margin:** the split-conformal quantile of these scores at 90% coverage, giving m90.
- **Reported:** coverage of matched val and test pairs, the change in box area, and the change in false boxes when matching with the inflated boxes. Pairs within a frame are correlated, so coverage ranges come from the block bootstrap.

### Floor
- **Calib and val:** if a target's λ would fall below 0.05, re-run calib and val at a 0.01 floor with the frozen settings, after committing.
- **Test:** re-running test is not done without asking first. Test's ceiling at the 0.05 floor is 95.2%, so a shortfall at 0.95 may be partly a floor effect and will be reported as such.

### Outputs
- **Code:** `src/conformal.py`. Reusable functions: `person_found_scores`, `frame_losses`, `calibrate_crc`, `calibrate_worst_block`, `calibrate_box_margin`, `apply_threshold`, `inflate`. The CLI writes the report.
- **Metrics:** `results/metrics/conformal-baseline.json` holds the calibrated λ and margin per target and method, val and test results with ranges, per-block and LOBO results, and the git commit.
- **Figure:** `results/figures/conformal-recall-vs-fp.png`, recall against false boxes per tile for val and test, with the calibrated points marked.
- **Docs:** `docs/results-phase3.md` (tables) and `docs/log.md` (decisions).
- **Checkpoint:** show the target-vs-achieved table with ranges before starting Phase 4.

## Phase 4 design (thin end-to-end slice on the Mac)
- **`src/detector.py`:** a `Detector` class that wraps `AutoProcessor` and `GroundingDinoForObjectDetection` with the frozen settings, and exposes `detect(image) -> [{bbox, score}]` at the cache floor.
  - Optional `adapter` path: `peft` LoRA weights, or a state dict for the head, loaded with `strict=False`. Without it, the class gives the exact zero-shot model.
  - `baseline.py` keeps its own code path, so cached results stay reproducible.
- **`src/pipeline.py` + `configs/poc.yaml`:**
  - **Input:** an ordered folder of frames, or a manifest plus a frame range.
  - **Per frame:** detect, keep boxes with score ≥ λ(α) taken from `conformal-baseline.json`, inflate them by m90, and flag the frame if anything is kept.
  - **Writes:** per-frame JSON (boxes, scores, inflated boxes, flagged), annotated images, and `summary.json` (frames, detections, flagged count, seconds per tile).
- **Replay:** frames 1321–1360 (tall grass), 481–520 (forest shadow) and one easy block, all from test. Running the frozen pipeline on test blocks is not tuning, but it is a test re-run, so I'll ask first. If you'd rather not, the fallback is to replay easy or hard blocks from val and calib.
- **Consistency test:** on a replayed block, the kept detections must equal the cached detections filtered at λ, within float rounding.
- **Optional:** skip near-duplicate consecutive frames using a cheap thumbnail difference, logging the skip count. Skipped frames reuse the previous frame's detections, so the consistency test runs with skipping off.

## Phase 5 readiness
`docs/HPC.md` covers the environment spec, data staging through relative-path manifests, which weights to save (LoRA and head only), and the questions for you (scheduler, GPU, modules, storage). No training code yet.

## Git protocol
- Branches: `phase3-conformal`, then `phase4-poc`. You make the commits and tags, and I give you the commands at each checkpoint.
- Every inference run needs a clean `git status` first.
- Cached detections and `run.json` files are added with `git add -f`.
