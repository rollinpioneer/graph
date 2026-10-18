# EG-QUALITY-VALIDATION-V1

Status: REGISTERED; execution authorized by user 2026-10-18.
Branch: [codex/eg-quality-validation-v1-20261018](https://github.com/rollinpioneer/graph/tree/codex/eg-quality-validation-v1-20261018)

P0: Validate continuous decoded grasp quality by native geometry/SDF scoring, reproduce saved quality/collision, compare original dense-label cells and width/pose corrections. Never treat missing nearest labels as failed ground truth.
P0: Top-1 failures on frozen stride-4 frames: direct width-only, pose/depth-only (48 cells at predicted width), and joint pose/width controls. These are overlapping counterfactual feasibility categories, not proof of a training mechanism.
P0: Freeze native evaluated candidate lists, rank by best-label, actual-label, actual geometry, and best 48-cell geometry quality. Score AP against actual original candidate quality, including denominator 1..50 and all six friction levels. Also resolve only ties of best geometry quality using actual quality. The difference between oracle targets is conditional ambiguity, not guaranteed trainable gain. Four fixed frames per scene/model for the expensive geometry pool; all 64 stride-4 frames for Top-1. Report per-scene and unique-observation sensitivity; no official split AP claim.
P1: Measure dense-label extraction, write/read time and bytes on two predetermined dev scenes (8,36), comparing one best cell with all 48 cells at the same point/view. Raw-label widths and quality retained; no new supervision algorithm. Report cold source access versus warm cache limits. Sample retained geometric quality generation cost separately.

Reuse old model predictions, original labels, SDFs and scene geometry read-only. Four low-priority single-thread CPU workers, no GPU reservation. Start with one-scene pilot; correctness failures stop affected stages. All substantive source and runtime provenance archived. Full results stay in new root; only lightweight results and code in Git. No new model or training.
