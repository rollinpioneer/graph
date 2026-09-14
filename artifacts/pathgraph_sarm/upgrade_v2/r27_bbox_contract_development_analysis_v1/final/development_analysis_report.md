# R27 Bounding-Box Contract Development Analysis

This is development analysis only. It does not replace the original R27 result, does not rewrite the R27 formal conclusion, and does not announce confirmation passed.

## Contract correction

`visual_separation.detect_frame_boxes()` and the episode adapter now share an explicit `xyxy = [x1, y1, x2, y2]` contract. The adapter validates and preserves detector coordinates; it never applies a second width/height conversion. Tests cover positive gaps, overlap, missing boxes, and detector-to-episode round-trip invariance.

## Frozen thresholds

B1/B2/B3 replay uses the existing locked operating points: B1 separation threshold 0.10; B2/B3 motion window 500 ms, motion threshold 0.35, radial-growth threshold 0.175, and motion-efficiency threshold 0.65. The scoring deadline remains 750 ms. No threshold was lowered and no physical collection was run.

## B1/B2/B3 per-horizon counts

| method | horizon | on time | early | late | missed | right censored | false loss | |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 100 ms | 0 | 0 | 0 | 98 | 18 | 0 |
| B1 | 250 ms | 0 | 0 | 0 | 40 | 76 | 0 |
| B1 | 500 ms | 0 | 0 | 0 | 36 | 80 | 0 |
| B1 | 750 ms | 0 | 0 | 0 | 36 | 80 | 0 |
| B1 | 1000 ms | 0 | 0 | 0 | 24 | 92 | 0 |
| B2 | 100 ms | 38 | 18 | 0 | 42 | 18 | 0 |
| B2 | 250 ms | 38 | 18 | 0 | 24 | 36 | 0 |
| B2 | 500 ms | 38 | 18 | 0 | 20 | 40 | 0 |
| B2 | 750 ms | 38 | 18 | 0 | 20 | 40 | 0 |
| B2 | 1000 ms | 38 | 18 | 0 | 8 | 52 | 0 |
| B3 | 100 ms | 38 | 18 | 0 | 42 | 18 | 0 |
| B3 | 250 ms | 38 | 18 | 0 | 24 | 36 | 0 |
| B3 | 500 ms | 38 | 18 | 0 | 20 | 40 | 0 |
| B3 | 750 ms | 38 | 18 | 0 | 20 | 40 | 0 |
| B3 | 1000 ms | 38 | 18 | 0 | 8 | 52 | 0 |

## Old versus new per-episode differences

Two explicit baselines are retained. `old_vs_review_v1` compares against the prior zero-physics review; its predictions are unchanged (0/624 rows changed). `old_vs_original_r27_v1` compares against the untouched original R27 artifact and records 624/624 rows with state/reason differences because the corrected review uses the replay state output; first evidence and score fields remain directly reported.

Detailed rows: `old_vs_review_v1/old_vs_new_per_episode.csv`, `old_vs_original_r27_v1/old_vs_new_per_episode.csv`; aggregate method/family counts are in each `old_vs_new_change_counts.csv`.

## Six B2 missed replenishment losses

Per-frame raw gate values and simultaneous blockers are in `shadow_replenishment/b2_gate_diagnostics/b2_missed_loss_gate_diagnostics.csv` and `.jsonl`; the summary is `b2_missed_loss_gate_summary.json`. The six episodes are the fast-detachment and contact-occluded true-detachment losses. Across 486 frames, blocker counts are baseline-not-ready 108, visual-missing/unavailable 0, window invalid 0, amplitude below threshold 142, radial growth below threshold 264, and motion efficiency below threshold 378. Blockers may co-occur on a frame.

## Conclusion

The cache-only result remains `R27_INSUFFICIENT_OBSERVATION_HORIZON`; `confirmation_passed=false`. The original R27 result, review report, and 24-rollout raw replenishment data remain preserved.
