# R-TB-E-1 (B1-K+E, seed 1, ELASTIC-01) run record

Generated from files only (no environment, no GPU). Frozen T_B profile, empty R, provider 0, T_P 0, formal test 0.

## Training

actual N = 14309 valid transitions, actual T = 68820.25 s (Ncap 16384, Tcap 68812.80); stop reason Tcap.
complete updates 13 + fragment updates 1; optimizer steps 876; training success episodes 950; NaN_n 0; hard_fail None; first_update_ok True.
wall time (launch to job_summary): 7.827 h.

## Frozen dev10 (deterministic argmax)

| point | success | mean discounted return | checkpoint sha256 |
|---|---|---|---|
| eval_n_000000 | 0/10 | 0.000000 | a83c64bef762fe00... |
| eval_n_004096 | 0/10 | 0.000000 | c606ebeae8fe5d82... |
| eval_n_008192 | 0/10 | 0.000000 | 04238a0b116c6ecd... |
| eval_final | 10/10 | 0.499977 | 62099c437dabd361... |

success_seconds in the eval rows: FINAL_SKILL_DURATION_ONLY (final-skill duration only); episode total success time: UNVERIFIED.

## Checkpoints

selected checkpoint final_n_014309_u13.pt sha256 62099c437dabd361412ae2fe3b5b6fadcf785bc88661e32ac7d35564f79522e7 (rule max_mean3_then_max_worst_then_earlier_N).
18 generations; all file hashes and COMPLETE markers match: True. Fresh-process loads all ok: True (fresh_load_final_n_014309_u13.json, fresh_load_update_complete_1.json).

## Storage guard

events: {"GUARD_START": 1, "SAMPLE": 464, "GUARD_END": 1}.
pause (SIGSTOP) events: 0; resume (SIGCONT) events: 0; HARD_MARGIN_BREACH events: 0.
minimum free over logged samples: 973.929 GiB. the minimum is over the guard's own samples; free space between 5 s samples is not measured; the SAMPLE log is thinned (every 12th tick, or every tick while free < resume threshold); other users share this filesystem; quota was NOT_MEASURED

## Ledger

{"plans": {"R-TB-E-1": "COMPLETE"}, "new_rl_attempts_used": 4, "new_rl_attempts_cap": 4, "elastic_slots": {"ELASTIC-01": "CONSUMED", "ELASTIC-02": "UNALLOCATED", "ELASTIC-03": "UNALLOCATED", "ELASTIC-04": "UNALLOCATED"}}

## Same profile / independent initialisation versus R-TB-E-0

execution-path hashes equal to R-TB-E-0 (raw): False; equal after normalizing only the train-split path in the resolved runtime manifest: True ({"checked": true, "differing_lines": [101], "differs_only_in_train_split_path": true, "normalized_hash_identical_to_E0": true, "note": "raw file hashes differ only because the train-split file lives in a different launch directory"}); frozen H/d_ref/Tcap/Ncap equal: True; final_tb.py hash differs (intended: registration row and cap): True.
initial checkpoint (n_000000) differs from R-TB-E-0's: True (E1 a83c64bef762fe00, E0 1882e819ac5d8899); training seed E1 1, E0 0.

## Why the 0/10 points failed (exit-reason families, normalized)

- eval_n_000000: {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10}
- eval_n_004096: {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10}
- eval_n_008192: {"NO_CANDIDATE_SAFE_TERMINATION": 10}
- eval_final: {"TASK_SUCCESS": 10}

## Descriptive context only

Base run R-TB-E-0 (same method, seed 0): {"run_id": "R-TB-E-0-20261001T115252Z-cac2fa5b", "training_seed": 0, "actual_N_valid_transitions": 14512, "actual_T_interaction_seconds": 68813.30000001457, "complete_updates": 14, "fragment_updates": 1, "optimizer_steps": 872, "train_success_episodes": 981, "NaN_n": 0, "stop_reason": "Tcap", "dev10": [["eval_n_000000", "0/10", 0.0, {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10}], ["eval_n_004096", "0/10", 0.0, {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10}], ["eval_n_008192", "0/10", 0.0, {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 7, "NO_CANDIDATE_SAFE_TERMINATION": 3}], ["eval_final", "10/10", 0.48934047729248054, {"TASK_SUCCESS": 10}]]}
Two seeds are not a reliability estimate; no comparison with B2 and no change to Method 2.1.1 follows from this record.
