# Run log

Decisions and their reasons, newest last. Results tables live in `docs/results-*.md`; numbers here are only what drove a decision.

## 2026-11-07: Phase 3, conformal baseline (branch `phase3-conformal`)

- **Plan written:** `docs/plan-phase3-4.md`.
- **Checked before designing:**
  - Every frame in calib, val and test contains people.
  - Share of people with a qualifying box (IoU ≥ 0.3) above the 0.05 floor: calib 97.7%, val 97.6%, test 95.2%.
  - Conclusion: test's ceiling at the 0.05 floor is about 0.95, so a shortfall at that target is partly a floor effect.
- **Matching:** "found" means any kept box with IoU ≥ 0.3. Each person therefore has a single found score, which makes recall monotone in λ, as conformal control requires. Centre-inside and IoU ≥ 0.5 are reported alongside. Precision and false boxes use one-to-one greedy matching (`evaluate.match_pairs`).
- **Calibration unit:**
  - *Frames, not blocks.* With 7 calib blocks, the conformal correction term is 1/(n+1) = 0.125, which makes targets above 0.875 impossible. With 218 frames the term is 0.005.
  - *Second rule, `worst-block`.* It calibrates each calib block separately and keeps the lowest λ, so it is conservative against scene shift and uses calib only.
- **Refactor check:** `evaluate.match_image` was split into `qualifies` and `match_pairs` so the box margin can use the matched pairs. Re-scoring val300 and re-drawing a figure gave byte-identical outputs.
- **Finding 1, frame vs pooled recall:** CRC controls *frame-averaged* recall, and calib meets every target under that measure (0.703 / 0.804 / 0.906 / 0.955). Pooled person recall is lower, because crowded frames are harder and frame averaging weights a 1-person frame the same as a 10-person frame. Tables report both.
- **Finding 2, crc-frame misses on val and test:**
  - Test shortfall: −13, −10, −8 and −5 points.
  - Val is also slightly short.
  - Leave-one-block-out (LOBO) shows λ barely moves between folds (for example 0.429–0.453 at the 0.70 target), while per-block recall ranges from about 0.49 to above 0.9.
  - The scene problem is therefore heterogeneous recall *at a fixed λ*, not an unstable λ. That is descriptive, given 6–7 blocks per split.
- **Finding 3, worst-block meets its targets on test (0.70 → 0.730, 0.80 → 0.822) but cannot reach 0.90 or above at the 0.05 floor.** Calib blocks 1–40 and 41–80 have about 9% of people with no box above the floor. A 0.01 floor (calib and val re-run, about 20 minutes) may fix this. Deferred to the checkpoint because a clean commit is required before re-running.
- **Box metric fix:** "box area ×" is now measured over matched pairs only, after clipping. Averaging over all kept boxes understated growth (×1.21 at m = 0.33), because large junk boxes clip at tile edges.
- **Figure:** symmetric-log x-axis, because the low-λ tail reaches 18 false boxes per tile.

## 2026-11-07: Checkpoint decisions (from review)

- **Guarantee:** keep the frame-averaged guarantee, with no people-weighted variant. Frame-averaged recall is now the headline column; pooled person recall is secondary. Both are defined once at the top of `docs/results-phase3.md`.
- **Lower floor:** re-run calib and val at a 0.01 floor (configs `baseline_calib_f001.yaml` and `baseline_val_full_f001.yaml`). Only `cache_threshold` changes. `text_threshold` stays 0.05, because in this processor it only picks label text and doesn't filter boxes.
  - **Test stays at its 0.05 floor.** Any λ below 0.05 is reported as "needs lower-floor run" for test instead of being scored, and a test re-run needs your approval.
  - **Expected check:** detections at or above 0.05 in the new caches should be identical to the old caches (same model and no NMS; only the floor differs).
- **`evaluate.py` refactor verified:** for val, test and calib, all 12 COCO statistics and the P/R/F1/TP/FP/FN counts at threshold 0.3 are identical to the committed Phase 2 metrics (AP50 0.857 / 0.753 / 0.856).
- **LOBO for both methods:** reports the share of blocks meeting each target and the median shortfall among blocks below it. For `worst-block`, a fold's λ is the minimum of the other 12 blocks' own thresholds.
- **Test is confirmatory only.** `worst-block` was introduced *after* the `crc-frame` test shortfall was seen. This is stated in the results doc.
- **False-box types at the 0.90 target:** with IoU ≥ 0.3 as the match rule, boxes overlapping a person at 0.3–0.5 now count as hits. So "loose" now means IoU 0.1–0.3, and most false boxes are duplicates or background. The Phase 2 breakdown used IoU 0.5 and isn't directly comparable.

## 2026-11-07: Floor 0.01 re-run and final Phase 3 outputs

- **Runs:** `baseline-calib-f001` (commit `d7d2c5e`) and `baseline-val-full-f001` (commit `38f4aa5`, started after the second commit). Both are clean, with identical inference code; frozen settings, only `cache_threshold` = 0.01. Speed: 0.87 s per tile for calib.
- **Check 2 (tolerance: same boxes, scores within 1e-4):** passed exactly. All 11,308 calib boxes and 10,613 val boxes above 0.05 match the 0.05-floor caches, with a score difference of 0 and a box difference of 0 px. No MPS nondeterminism was observed, so the old caches stay as they are.
- **Cache sizes:** 23.4 MB (calib) and 21.8 MB (val), over the ~10 MB limit, so **not committed**. Each `run.json` records `cache_committed: false`, the local path and the byte size. The caches regenerate from the committed configs in about 20 minutes.
- **Floor misses (no box with IoU ≥ 0.3 at all):** at 0.05, 17 of 748 calib and 21 of 860 val people. At 0.01, 2 calib (frames 1 and 184) and 2 val (frames 1739–1740). Crops are in `results/figures/floor-misses.jpg`; details in `results/metrics/floor-misses.json`. They look like an edge sliver (24×6 px) and people barely or not visible (dark rocks, tall grass), possibly label issues. Descriptive only.
- **Final outputs:** generated from commit `38f4aa5` with dirty=false. `crc-frame` thresholds are unchanged from the 0.05-floor run, because all of them were above 0.05. `worst-block` now reaches the 0.90 target at λ = 0.031, but at a cost of 51 false boxes per tile on val (precision 0.03), so it isn't usable operationally. Test isn't scored at that λ ("needs lower-floor run"), because test stays at 0.05 until you approve otherwise.
- **Phase 4 default:** your rule was `crc-frame` at 0.90 unless the lower floor let `worst-block` reach 0.90. It does reach it, but at 51 false boxes per tile, so my recommendation is to keep `crc-frame` at 0.90 (0.38 false boxes per tile on val). This needs your decision.
- **NMS:** kept out of the baseline, as instructed. It is proposed as a separate calibrated variant: duplicates are 95 of 211 val false boxes and 109 of 246 test false boxes at the 0.90 target.

## 2026-11-07: Decisions after the Phase 3 checkpoint

- **Phase 4 default:** `crc-frame` at the 0.90 target (λ = 0.250, box margin 0.304). α can be set to give target 0.80 or 0.90.
- **`worst-block` at 0.90:** reported as unusable on val (λ = 0.031, 51 false boxes per tile, precision 0.03). Unscored on test, because there will be no test run at the 0.01 floor.
- **Per-block frame recall at λ = 0.250** (`src/block_recall.py`, `results/metrics/block-recall-lambda0.25.json`):

  | Group | Blocks | Lowest | At or above 0.80 | At or above 0.90 |
  |---|---|---|---|---|
  | calib + val | 13 | 0.731 (val 801–840) | 11 | 7 |
  | test | 7 | 0.703 (481–520) | 4 | 3 |

## NMS variant: rules fixed before running

- **Variant:** class-agnostic greedy NMS on each tile's cached detections, before thresholding. Overlap thresholds τ ∈ {0.3, 0.4, 0.5, 0.6, 0.7}.
  - Lower-scoring boxes never suppress higher ones, so the boxes kept above λ don't depend on the cache floor. Test at its 0.05 floor is therefore valid for any λ ≥ 0.05.
- **For each τ:** recalibrate λ with `crc-frame` on calib with NMS applied, for targets 0.80 and 0.90, and recalibrate the box margin on the same pairs.
- **Selecting τ (val only, at the 0.90 target):**
  - *Eligible* if val frame recall ≥ the no-NMS value − 0.01, **and** val frame recall on frames with 3+ people ≥ the no-NMS value − 0.02.
  - *Chosen:* the eligible τ with the fewest val false boxes per tile; on a tie, the larger τ.
  - *The same τ* is used for the 0.80 target.
- **Pass rule (for plugging into Phase 4):** the chosen τ cuts val false boxes per tile by at least 20% against no NMS at the 0.90 target. If nothing passes, the Phase 4 pipeline runs without NMS.
- **Test:** scored once with the chosen τ, as confirmation only, from the existing 0.05-floor cache (no re-run).

## NMS variant: outcome (`src/conformal_nms.py`, `docs/results-phase3-nms.md`)

- **All five τ were eligible.** At the 0.90 target, val frame recall moves by at most −0.001 and 3+-people frame recall by at most −0.001.
- **Chosen τ = 0.3**, which gives the fewest val false boxes per tile: 0.382 → 0.199 (−48%). The pass rule required ≥ 20%, so it **passed**. λ stays 0.250 at the 0.90 target and 0.351 → 0.349 at 0.80; the box margins are unchanged.
- **Test confirmation, scored once from the existing 0.05-floor cache:**

  | Measure | No NMS | NMS τ = 0.3 |
  |---|---|---|
  | False boxes per tile | 0.456 | 0.219 (−52%) |
  | Precision | 0.672 | 0.808 |
  | Frame recall | 0.861 | 0.856 |
  | Frame recall, 3+-people frames | 0.775 | 0.764 |

- **Caveat on the 3+-people check:** it's not very selective on val, where 168 of 177 frames have 3 or more people. Test has 90 of 210 such frames, and there the cost is about one point of recall.
- **Phase 4:** because the variant passed, it is plugged in, as you instructed. The pipeline default is `crc-frame` at the 0.90 target **with NMS τ = 0.3** (λ and margin from the NMS calibration), labelled as such. A config switch (`nms: null`) gives the no-NMS baseline.

## 2026-11-10: Phase 4, proof-of-concept pipeline (branch `phase4-poc`)

- **Regenerated from clean commit `4626daf`:** NMS variant and block recall, with identical numbers.
- **`src/detector.py`:** wraps `baseline.detect` with the frozen settings, so its output is identical to the cached runs.
  - Optional `adapter`: a PEFT directory, or a state dict loaded with `strict=False`.
  - State-dict path tested: 40 head tensors saved and reloaded gave identical detections, with 0 unexpected keys. The PEFT path is untested until real LoRA weights exist.
- **`src/pipeline.py` + `configs/poc.yaml`:**
  - Each frame goes through: detect per tile → NMS (if set) → keep score ≥ λ → inflate by margin m → flag the frame if anything is kept.
  - λ and m are read from the calibration file for the configured method, target and NMS setting. The pipeline refuses an NMS τ that isn't the passed, chosen variant.
  - The summary records the operating point: method, λ, τ, margin, the calibration file's SHA-256 and its git commit, plus the run's own commit.
- **Default:** `crc-frame` + NMS τ = 0.3 at the 0.90 target (λ = 0.2503, m = 0.3043). Alternatives: `nms: null` for the plain baseline; target 0.80.
- **Replay blocks (all test, explicitly requested):**
  - 1321–1360: tall grass, the main source of false boxes.
  - 481–520: forest shadow, the weakest test block at 0.703.
  - 281–320: easy (0.987 at λ = 0.25 without NMS).
  - Running inference on these blocks is a re-run on test, approved by your instruction to replay them. Nothing is tuned on it.
- **Labels are used only for reporting:** per-block frame and pooled recall against 0.80 and 0.90, and false boxes per tile. The pipeline's own annotated images carry no labels; `src/poc_examples.py` draws a labelled sheet for the report.
- **Consistency test:** kept detections must equal the Phase 3 test cache after the same NMS and threshold (scores within 1e-4, boxes within 0.02 px). It passed on an 11-tile development run.
- **Near-duplicate frame skipping:** not implemented (optional). Recorded in `summary.json`.
- **Split buffers:** frames at block edges were dropped from the splits, so a "40-frame block" replays fewer frames.
