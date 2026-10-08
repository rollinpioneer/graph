# C1-PUBLIC-DEPOTS-REL-V1: final summary

Plan: `CP_DISR_C1_Next_Direction_v2_Depots_REL_20261008.md`. Base `e3b1e65b3`, branch `codex/cp-disr-c1-public-depots-rel-v1`. Registration commits pushed before any formal result: `3f30a79` (first launch, stopped, see disclosures) and `28ffb0a` (the run reported here, run root `20261008T063544Z_9898106e`).
Executed: the Blocksworld zero-training stage (gated look-ahead for MG and REL, fixed two-step with an `h_add` leaf, two invariance probes; 384 episodes, 0 trainings); the generic PDDL / relation interface; three Depots trainings (MG-G, DENSE-G, REL-G; 4,100 updates each, common initialisation, identical data); 27 internal evaluations (3 models x one-step+C3 / fixed two-step / gated x struct, joint, IPC); LAMA (first plan and anytime, 300 s) and WL-GOOSE (author code, `classic.toml`, no tuning) on every problem. Not run: AIW-AD / KR 2023 (see claim boundary). 53 DAG tasks, 0 failed.
One training seed per model. Wall-clock times are indicative only: the machine and its GPUs were shared with other users. All numbers below are generated in `results/*.csv` (`final_summary_numbers.md` has the plain tables).

## 1. Data and budgets
- Track B (public generator `depots`, explicit structure rules, all problems pairwise non-isomorphic): train 96 problems (3 to 5 crates, one tower of height 2 or 3, half need transport), dev 24, struct 32 (4 crates 2+2, 5 crates 3+2), joint 128 (8 cells: 6/8 crates x 1/2 towers x height 2/4; every cell 8 problems with and 8 without transport). Track A: the 22 IPC Depots problems, unchanged.
- Reference lengths: exact BFS optimum for train/dev/struct and the 64 six-crate joint problems; A*+LM-cut optimum for all 128 joint problems (so every joint ratio is against a proved optimum) and 7 IPC problems; best known LAMA plan for the other 15 IPC problems (upper bounds, not proofs). Step cap 2 L_ref + 4; per-problem wall budget 300 s for every time-limited method.
- Training: 457 trajectories (canonical optimal plan plus deviations), 1,634,323 decision exposures per model; MG-G 513,984 trainable parameters, DENSE-G / REL-G 580,072 (66,088 in the shared new attention layer). Dev (24 problems): every checkpoint of every model solved 24/24, so the selection rule (most successes, then smaller failure-penalised cost) was decided by ratio differences of 0.001 to 0.05: MG-G update 820, DENSE-G 4100, REL-G 3280.
- REL mask: non-self density at initial states 0.66 (struct), 0.42 (joint), 0.41 (IPC); the contract-threat term is empty in Depots (0 entries on 12 sampled problems).

## 2. Main table (completion / failure-penalised cost ratio, lower is better; ratio = plan length / reference, failure = cap+1)
| method | struct 32 | joint 128 | IPC 22 |
|---|---|---|---|
| MG-G one-step / two-step / gated | 32 / 1.142, 32 / 1.106, 32 / 1.066 | 119 / 1.325, 123 / 1.227, 126 / 1.178 | 4 / 2.03, 9 / 1.72, 8 / 1.74 |
| DENSE-G one-step / two-step / gated | 32 / 1.029, 32 / 1.043, 32 / 1.042 | 125 / 1.164, 125 / 1.145, 127 / 1.119 | 8 / 1.81, 10 / 1.65, 11 / 1.60 |
| REL-G one-step / two-step / gated | 32 / 1.058, 32 / 1.029, 32 / 1.038 | 119 / 1.197, 116 / 1.221, 119 / 1.170 | 7 / 1.90, 11 / 1.63, 12 / 1.61 |
| WL-GOOSE | 32 / 1.174 | 128 / 1.211 | 20 / 1.07 |
| LAMA first plan | 32 / 1.227 | 128 / 1.350 | 20 / 1.42 |
| LAMA anytime (300 s) | 32 / 1.000 | 128 / 1.007 | 21 / 1.05 |

## 3. The registered primary comparison: REL-G vs DENSE-G
| set, execution | solved (REL / DENSE) | only REL / only DENSE (sign p) | common solved: REL cheaper / costlier / same (sign p) | mean cost difference |
|---|---|---|---|---|
| struct, one-step | 32 / 32 | 0 / 0 | 4 / 4 / 24 (1.0) | +0.31 steps |
| struct, two-step | 32 / 32 | 0 / 0 | 4 / 4 / 24 (1.0) | -0.13 |
| joint, one-step | 119 / 125 | 1 / 7 (0.070) | 29 / 27 / 62 (0.89) | +0.20 |
| joint, two-step | 116 / 125 | 2 / 11 (0.022) | 25 / 22 / 67 (0.77) | +0.11 |
(gated, secondary: joint 119 / 127, only 0 / 8 (0.008), cost 22 / 24 / 73 (0.88).) p-values are two-sided, uncorrected, exploratory in the registered sense (only the four rows above are primary).
- REL-G does not complete more problems and does not need fewer actions than DENSE-G. On solved joint problems its mean ratio is slightly lower (1.120 vs 1.140 one-step; 1.115 vs 1.120 two-step; 1.089 vs 1.111 gated), but it solves 6 to 9 fewer problems and the paired cost differences are symmetric, so the failure-penalised ratio favours DENSE-G in every execution.
- REL-G failures on joint are all step-cap exhaustions concentrated in the 8-crate height-4 cells (two-step: n8k1h4 5, n8k2h4 5, n6k1h4 1, n6k2h4 1). With the goal-destruction label known (six-crate joint, 64 problems): TRUE (10 problems) REL-G one-step 10/10, ratio 1.039 vs DENSE-G 10/10, 1.162 vs MG-G 8/10, 1.538; under two-step the order reverses (REL-G 9/10, 1.179; DENSE-G 10/10, 1.055).
- Both attention variants beat MG-G on solved cost: DENSE-G vs MG-G one-step joint 71 cheaper / 16 costlier (p < 1e-6), struct 18 / 2 (p 4e-4); REL-G vs MG-G joint 61 / 22 (p 2e-5), struct 16 / 1 under two-step (p 3e-4). Dense goal mixing already provides this gain here; the relation restriction adds nothing measurable.

## 4. Execution layer
- Fixed two-step vs one-step: helps MG-G (joint 119 -> 123 solved, 57 vs 30 problems cheaper, p 0.005), is neutral for DENSE-G (125 / 125; 25 cheaper, 38 costlier, p 0.13), and hurts REL-G completion (119 -> 116; 27 / 29, p 0.89). On IPC it adds 2 to 5 solved problems per model.
- Selective look-ahead (gate frozen from the Blocksworld stage, tau = 10th percentile of the training top-2 margin of each Depots model: MG-G 1.56, DENSE-G 7.20, REL-G 9.03): on joint it solves at least as many problems as fixed two-step (126 vs 123, 127 vs 125, 119 vs 116) with lower cost (problems cheaper / costlier: MG-G 49 / 22, DENSE-G 39 / 10, REL-G 39 / 9; sign p 0.002, 4e-5, 2e-5) and about 78% fewer evaluated leaf states (joint: 26,149 / 26,310 / 25,438 vs 116,590 / 119,127 / 119,951; one-step logit passes unchanged). On IPC completions differ by 1 (8 vs 9, 11 vs 10, 12 vs 11) and the gate saves less there (10% / 4% / 41% fewer leaf states for MG-G / DENSE-G / REL-G; it expands on 68% to 74% of IPC decisions).
- raw / C3 / final actions (joint, labelled decisions on the six-crate problems): decisions in which an optimal raw action survives C3 and the look-ahead replaces it by a non-optimal one: fixed two-step DENSE-G 57, MG-G 85, REL-G 65; gated DENSE-G 26, MG-G 11, REL-G 32. Full table in `raw_c3_final_actions.csv`.

## 5. Public Depots and external systems (system table, not a structural comparison)
- Track A: WL-GOOSE solves 20/22 (two time-outs, p15 and p20), LAMA first plan 20/22, LAMA anytime 21/22; the best neural row (REL-G gated) solves 12/22 (failures: 6 step-cap, 4 time-outs; the large instances, up to 24,474 template nodes, are beyond the unconditioned scorer within 300 s). Every IPC problem solved by any of the nine neural rows is also solved by WL-GOOSE (it additionally solves p11, p12, p14, p16, p21, p22, none of which any neural row solved).
- Track B: WL-GOOSE solves every struct and joint problem in under 1 s with ratio 1.17 / 1.21; LAMA anytime is near optimal (1.000 / 1.007); the neural rows have failure-penalised ratios of 1.03 to 1.14 (struct) and 1.12 to 1.33 (joint) and complete 116 to 127 of 128 joint problems. On commonly solved joint problems REL-G one-step is cheaper than WL-GOOSE in 72 problems and costlier in 31 (p 7e-5), and costlier than LAMA anytime in 52 of 53 non-tied problems.

## 6. Blocksworld zero-training stage (Confirm128, development material, 384 episodes)
| condition | success / all-steps-optimal | failure-penalised ratio | avoidable destructions | leaf states (unique) |
|---|---|---|---|---|
| LOOK2_MG (reused) | 127 / 96 | 1.106 | 9 | 28,588 (24,430) |
| GATED_MG | 127 / 95 | 1.121 | 11 | 6,279 (6,279) |
| REL_LOOK2 (reused) | 128 / 97 | 1.110 | 12 | 28,440 (24,281) |
| GATED_REL | 127 / 101 | 1.104 | 9 | 7,283 (7,283) |
| LOOK2_HADD (MG roots, `h_add` leaf) | 127 / 99 | 1.077 | 4 | 25,241 (21,297) |
- Gate decision (frozen before the run): pooled gated 254 vs fixed 255 completions, 13,562 vs 57,028 leaf states, 20 vs 21 avoidable destructions, so the gate was carried to Depots. Gate reasons (MG / REL): keep one-step 1,338 / 1,230, low margin 342 / 424, single root 178 / 180, goal next 127 / 127, local dead end 1 / 1.
- `h_add` leaf: same tree, better than the learned leaf on every aggregate (pairs vs LOOK2_MG: 22 cheaper / 14 costlier, p 0.24; vs MG_C3: 37 / 10, p 1e-4); a simple relaxation leaf is a strong control for the learned value, development evidence only.
- Probes (240-state bank): module level, the REL mask makes appended goals invisible to old queries (max change 2.4e-7, float rounding) while DENSE changes by 0.17 on average (max 1.31). End-to-end, appending physically separate satisfied material changes centred scores by about 5% of the score range for MG, DENSE and REL alike (REL 0.054 / 0.053 / 0.053, DENSE 0.052 / 0.047 / 0.053), top-1 unchanged in 97% to 99%: REL is not measurably more invariant than DENSE, so the invariance explanation is put down.

## 7. Outcome by plan section 12
Class C (REL close to or behind DENSE, gate useful) for the structural question: REL-G did not reduce cost against DENSE-G on public Depots and completed fewer hard joint problems; the selective look-ahead is the component that transferred (Blocksworld rule, Depots thresholds from Depots training margins only). Class E for the system question: WL-GOOSE and LAMA complete more problems; our rows produce cheaper plans than WL-GOOSE on commonly solved Track B problems but are not competitive on completion, especially on the large IPC instances. No further module, mask or loss is added.

## 8. Cost
Training wall time (incl. restarts): MG-G 8,789 s, DENSE-G 10,568 s, REL-G 7,702 s (1.3 to 4.9 s per update depending on other users' load on the same GPU); WL-GOOSE fits 54 s (typed encoding) and 42 s (IPC encoding); exact labels 327 s summed over problems (32 processes); Fast Downward references (LAMA first, LAMA anytime, A*+LM-cut) 70 s (struct), 23,995 s (joint), 11,343 s (IPC) summed over 30 parallel workers. Policy wall seconds per set and execution are in `compute_costs.csv` (one-step 119 to 345 s on joint, two-step 146 to 743 s, gated 132 to 235 s; ranking influenced by GPU sharing; leaf and logit-pass counts are the reliable compute measure).
