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
