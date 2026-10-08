# C1-DEPOTS-SEARCH-MATCH-V1: final summary

Plan: `CP_DISR_C1_Depots_Search_Matched_Runbook_v2_20261008.md`. Base `e10e80f56` (previous card, read only), branch `codex/cp-disr-c1-depots-search-match-v1`, registration commit `1243b5c` (pushed before the first search). Run root `runs/final_master/c1_route_b/depots_search_match_v1/20261008T144409Z_a34d9ba1`.
Executed: six frozen scorers in ONE shared search engine on all 182 problems = 1,092 runs (Struct32, Joint128, IPC22). **0 trainings, 0 optimizer steps, 0 new labels, 0 new problems.** All 44 DAG tasks finished, 0 failed, 0 technical failures (no ADAPTER_ERROR / INVALID_PLAN / MODEL_NONFINITE); 1,037 of 1,092 runs solved and every one of the 1,037 plans replays to the goal under the project semantics (`plans/`). Historical results were not touched.
All problems are development / comparison material (previous card, plan section 5.2); one training seed per learned scorer; all numbers are generated in `results/*.csv`.

## 1. What was compared
- Engine: eager greedy best-first search, common adaptation protocol (plan 6.2): one OPEN list keyed (h, serial), complete-state duplicate detection, CLOSED never reopened, shorter paths to OPEN states update in place, canonical action order, goal tested at generation, no beam, no helpful actions, no C3 filter, no step cap from a reference length. Node milestones 1,000 / 10,000 / 100,000 are snapshots of ONE run; every run is also bounded by 300 s wall clock (parsing, grounding / translation, template, conversion, search, scoring) and 8 GiB resident-set growth.
- Scorers: V_DENSE (update 4100), V_MG (820), V_REL (3280): the frozen state values V of the previous card; H_WL: the fitted WL-GOOSE model of the author software (typed model for Struct / Joint, IPC-encoding model for IPC), prediction rounded as the native planner does; H_ADD: additive delete relaxation (C++ helper); H_COUNT: unmet goal atoms.
- Verification before the first search (15 fixtures, all passing): successor index = project legal-action semantics in canonical order; engine semantics on synthetic graphs (dead end + other branch, duplicates, in-place shorter path, FIFO ties, goal at generation, milestones, node limit, +inf last / NaN technical, wall and memory statuses); h_add hand example and C++ vs Python reference on random Depots states; **H_WL equals the native Scorpion planner's heuristic on 48 states of Train96 problems in both encodings (plus 78 + 78 states of six typed and six IPC problems in a pre-registration probe)**; V equals the policy-path value difference; chunking is a pure compute split.

## 2. Coverage and plan length (solved / n (mean plan length of solved problems))
| scorer | Struct32 (100k) | Joint128 1k / 10k / 100k | IPC22 1k / 10k / 100k(300 s) |
|---|---|---|---|
| V_DENSE | 32 (14.2) | 128 (21.8) / 128 / 128 | 13 (37.9) / 14 (37.6) / 14 (37.6) |
| V_MG | 32 (14.8) | 128 (22.4) / 128 / 128 | 11 (40.0) / 14 (50.6) / 14 (50.6) |
| V_REL | 32 (14.3) | 128 (21.5) / 128 / 128 | 13 (37.8) / 15 (40.8) / 15 (40.8) |
| H_WL | 32 (16.2) | 124 (23.3) / 128 (23.7) / 128 | 13 (32.8) / 16 (37.7) / 18 (40.5) |
| H_ADD | 32 (18.2) | 110 (26.9) / 128 (29.6) / 128 | 5 (23.8) / 7 (27.7) / 10 (45.8) |
| H_COUNT | 32 (20.1) | 88 (24.7) / 120 (28.6) / 128 (29.9) | 3 (18.0) / 3 (18.0) / 6 (39.2) |
Mean plan length over proved optimum (Struct / Joint, all solved; optimal plans in brackets): V_DENSE 1.022 (25/32) / 1.095 (72/128); V_MG 1.061 (18) / 1.137 (47); V_REL 1.033 (26) / 1.081 (75); H_WL 1.174 (13) / 1.219 (18); H_ADD 1.296 (7) / 1.485 (8); H_COUNT 1.412 (10) / 1.477 (14). Mean expansions to the first solution (Struct / Joint): 14 / 24 (V_DENSE), 15 / 27 (V_MG), 15 / 26 (V_REL), 37 / 133 (H_WL), 78 / 510 (H_ADD), 162 / 1,852 (H_COUNT).
IPC failures: neural rows 7 to 8 runs each, **all `TIMEOUT_BEFORE_NODE_LIMIT`** (the 300 s wall limit ended the run after about 310 to 15,600 expansions, mean 6,100 to 8,400 per row); H_WL 4 runs reach the 100,000-expansion limit after 38 to 96 s; H_ADD 10 at the node limit and 2 timeouts; H_COUNT 16 at the node limit. Union over the neural rows: 18 problems; H_WL: 18; any scorer: 21 (p20 unsolved by all six).

## 3. Registered primary comparison: V_DENSE vs H_WL (10,000-expansion milestone; sign tests two-sided, uncorrected)
| set | solved (DENSE / WL) | only DENSE / only WL | common: DENSE shorter / longer / same (p) | DENSE fewer / more expansions (p) |
|---|---|---|---|---|
| Struct32 | 32 / 32 | 0 / 0 | 17 / 1 / 14 (1.4e-4) | 31 / 0 (< 1e-8) |
| Joint128 | 128 / 128 | 0 / 0 | 90 / 16 / 22 (< 1e-6) | 125 / 3 (< 1e-6) |
| IPC22 | 14 / 16 (100k: 14 / 18) | 2 / 4 (100k: 2 / 6, p 0.29) | 1 / 8 / 3 (0.039), 12 common | 12 / 0 (4.9e-4) |
- Struct and Joint: no coverage difference, much shorter plans and far fewer expansions for V_DENSE. IPC: V_DENSE needs fewer expansions where both succeed, but its plans are longer in 8 of 12 common problems, it reaches the node budget on fewer problems, and it is behind in coverage once the wall limit binds.
- Scale references (V_DENSE vs H_ADD, 10k): Struct 24 shorter / 0 longer, Joint 112 / 5, IPC 14 vs 7 solved; vs H_COUNT: Joint 128 vs 120 solved, 100 / 5 shorter. The ordering V_DENSE < H_WL < H_ADD < H_COUNT in expansions holds on Struct and Joint.

## 4. DENSE vs REL vs MG under the shared search
- V_DENSE vs V_REL: Joint 29 shorter / 30 longer / 69 same (p 1.0), IPC 14 vs 15 solved; no difference. V_REL has the lowest mean ratio on Joint (1.081 vs 1.095) and, in the 8-crate height-4 cells, 1.109 / 1.184 (REL) against 1.310 / 1.213 (DENSE), but these are cell means of 16 problems.
- Both attention models beat V_MG on Joint (DENSE 59 shorter / 23 longer, p 8.7e-5; REL 55 / 22, p 2.2e-4) and need fewer expansions; the relation restriction does not separate from dense mixing.
- The coverage gap of the previous card (REL-G 116 to 119 of 128 on Joint, DENSE-G 125 to 127, MG-G 119 to 126) disappears under search: all three solve 128 of 128. IPC coverage under the same 300 s: V_DENSE 14 / V_REL 15 / V_MG 14 against 8 to 11 / 7 to 12 / 4 to 9 (one-step, two-step, gated) before. The execution framework therefore explains a large part of the previous Track B coverage differences (H1); it does not explain the IPC gap to WL-GOOSE, which is limited by scoring cost (section 5).

## 5. Quality versus cost (H3)
| | Joint: scoring ms / state | Joint: mean wall s / problem | IPC: scoring ms / state | IPC: total wall s of 22 runs |
|---|---|---|---|---|
| V_DENSE | 3.6 | 0.55 | 3.9 | 2,497 |
| V_MG / V_REL | 3.5 / 3.7 | 0.56 / 0.58 | 4.4 / 4.3 | 2,732 / 2,315 |
| H_WL | 0.046 | 0.23 | 0.080 | 560 |
| H_ADD | 0.047 | 0.14 | 0.169 | 1,674 |
| H_COUNT | 0.008 | 0.12 | 0.006 | 195 |
- Neural scoring is roughly 20 to 80 times more expensive per state than H_WL / H_ADD, and the scoring call is 98% of the neural wall time on IPC (91% on Joint); inference seconds include state conversion on the host, the split inside is not measured here. Peak growth of the host resident set stayed below 1.5 GiB for every run; neural peak GPU memory 2.1 GiB (p22); graph encodings = states scored.
- On Joint the cost is invisible (every run finishes in under 5 s) and the node advantage is a plan-quality advantage. On IPC the per-state cost is the limiting factor: p21 and p22 end after about 1,200 and 310 to 350 expansions (255 to 965 ms per expansion), against 25,000 / 48,600 expansions solved by H_WL in 62 s / 232 s. Time-to-solution curves (IPC solved within 10 s / 60 s / 300 s): V_DENSE 11 / 14 / 14, V_REL 10 / 14 / 15, V_MG 11 / 11 / 14, H_WL 16 / 16 / 18, H_ADD 9 / 10 / 10, H_COUNT 6 / 6 / 6.
- A same-node comparison is therefore not available at 10,000 and 100,000 expansions on IPC for the neural rows: on 8 / 8 / 7 problems they stop before the node limit and are recorded as `TIMEOUT_BEFORE_NODE_LIMIT`, not as lack of guidance. At 1,000 expansions V_DENSE and V_REL (13 each) tie H_WL (13); V_MG solves 11.

## 6. Problems: topology and reference lengths
- Topology (from the problem files): Struct32 and Joint128 all have 1 depot, 1 distributor (2 places), 1 truck and 2 hoists, exactly like Train96. IPC22 has 2 to 6 trucks, 3 to 12 places, 3 to 15 hoists, 3 to 20 pallets, 2 to 20 crates and 84 to 22,852 ground actions (the encoding does not distinguish depots from distributors). Goal height, resources and size change together, so the IPC results do not isolate one cause. Every neural row solved all nine IPC problems with fewer than 1,500 ground actions; its failures are all among the 13 larger ones.
- Reference lengths (derived tables only; registered values and caps untouched): for 8 IPC problems a validated historical plan is shorter than the registered best-known LAMA length: p06 192 -> 58 (counterfactual cap 388 -> 120; one of the three old neural successes exceeds 120), p08 38 -> 34, p09 74 -> 61, p11 60 -> 50, p12 64 -> 63, p18 66 -> 61, p19 40 -> 37, p22 124 -> 89; no plan contradicts a proved optimum. This card additionally finds a 71-step plan for p15 (V_DENSE; historical best 78).

## 7. Decision by plan section 15
- **Joint / Struct: class A.** The frozen DENSE scorer in a shared search has no coverage loss and gives clearly shorter plans and fewer expansions than the fitted WL-GOOSE scorer and the classical references. Caveats: Struct / Joint are project-defined splits; the advantage is not shown on IPC, and per-state cost is 20 to 80 times higher.
- **IPC22: class B with class D components.** Better guidance per expansion where both finish, but the neural rows lose coverage to H_WL once the 300 s wall limit binds and their plans are not shorter there. The IPC comparison between DENSE-G and WL-GOOSE at node budgets above 1,000 is not complete (section 5).
- The relation-restricted scorer (V_REL) neither gains nor loses against V_DENSE under search; the earlier negative REL result is largely an execution effect on completion and not rehabilitated as a structural gain. The written recommendation is in `method_recommendation.md`.
