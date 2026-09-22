# Plan v1.1 migration summary

Identity: Research Specification v3.1 / Method 2.1.1 / Experimental Plan v1.1.

## 1. Old orchestrator

Stopped. PIDs 2275072/2275069/2688488/2726993/2387622 are not running. Guard file forbids dispatch/resume-to-64/seed1-2. Old Stage 2A remains RUNNING+SUPERSEDED, not PASS.

## 2. Old 8 first-update jobs (actual, not the stale 5/8 snapshot)

See `migration/legacy_run_inventory.csv`. T_A B0/B1/B2/Full seed0: 1 PPO update, 1024 transitions, latest.pt. T_C B0: 1 update/1024/latest.pt. T_C B1: 1 update recorded, optimizer_steps=60, latest.pt. T_C B2: 760 transitions, 0 updates, SIGINT. T_C Full: 609 transitions, 0 updates, SIGINT.

## 3. Old checkpoints not reusable as v2.1.1 results

Old B0/B1 definitions, actor prefix weights, Tcap=2*Ncap*d_ref, T_A in default matrix, weighted checkpoint rule. Historical only.

## 4. T_B binding / T_A exit

T_B goals are Inside(target,container) AND AtBuffer(second_object,buffer). T_A is out of the default matrix. T_A caches were not reused as T_B.

## 5. Production code

B0 two-layer Set Transformer; B1=B1-K phiK(c,0); A_CAT 4-view concat + contract-repeat anchor; actor_episode_discount_weight=false; GRU prefix recompute kept; Gamma=2^(-d/H); Tcap=Ncap*d_ref.

## 6. H and d_ref

{"D0": {"T_ref_median": 15.050000000001983, "attempts": 5, "d_ref_median": 4.5500000000015195, "success_case_ids": ["D0_dev_00", "D0_dev_01", "D0_dev_02", "D0_dev_03", "D0_dev_04"], "successes": 5}, "T_B": {"T_ref_median": 23.09999999999752, "attempts": 5, "d_ref_median": 4.199999999997672, "success_case_ids": ["T_B_dev_00", "T_B_dev_01", "T_B_dev_02", "T_B_dev_03", "T_B_dev_04"], "successes": 5}, "T_C": {"T_ref_median": 23.09999999999752, "attempts": 5, "d_ref_median": 4.399999999997561, "success_case_ids": ["T_C_dev_00", "T_C_dev_01", "T_C_dev_02", "T_C_dev_03", "T_C_dev_04"], "successes": 5}}

H=23.09999999999752 seconds from 5/5 legal scripted successes per family.

## 7. New 0A-0D

- 0A: PASS
- 0B: PASS
- 0C: PASS_WITH_NOTES
- 0D: PASS

## 8. Cache reuse

See `migration/cache_reuse_manifest.csv`. D0/T_C exact reuse; T_B newly captured+requested; T_A not reused.

## 9. First-pause 11 planned runs

All remain NOT_STARTED. No new-profile RL this round (count=0).

## 10. Next authorizable stage

Stage 1A (D0 x B2/Full x seed0) only after explicit authorization

Blocked: []

This round stops here. Do not start Stage 1A/1B/2A/2B/3A without a new explicit authorization.

