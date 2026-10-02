# Historical model compatibility (Phase A, zero environment / zero episode / zero optimizer)

Baseline `241159c1ceca3b53f0ba8423879524c6b2668e07`. Nothing was converted, resumed, retrained or re-saved.

| slot | run | method / seed | bytes re-hashed = frozen list | current-tree strict load | archived-tree strict load | class |
|---|---|---|---|---|---|---|
| TB-M1 | R2-B0-s0 | B0 s0 | True | FAIL (14 keys missing) | PASS | **B_ARCHIVED_EXECUTION_CLASS_COMPATIBLE** |
| TB-M2 | R1-B1K-s0 | B1-K s0 | True | FAIL (14 keys missing) | PASS | **B_ARCHIVED_EXECUTION_CLASS_COMPATIBLE** |
| TB-M4 | R1-B2-s0 | B2 s0 | True | FAIL (14 keys missing) | PASS | **B_ARCHIVED_EXECUTION_CLASS_COMPATIBLE** |

## Reading

The three historical final weights do not load strictly into the current 2.1.1 Policy: the current Policy carries 14 `effect_readout.*` parameters that the older state_dict does not have. No key was remapped, filled or dropped. Under the unmodified archived source (the v2_1 worktree whose neural / torch_rl / stage2a_v11 / runtime_factory / collector / persistence hashes equal the run manifests) the same files load strictly with no missing, unexpected or mismatching key. They are therefore class B, **provisionally**: class B is confirmed for the extension only if the archived execution class also reproduces the old dev record on the fixed dev case (Phase C).

B1-K s0, B2 s0 (commit 0eb3776d) and B0 s0 (commit 228517ed) share identical execution-file hashes, so they form one archived execution class.

## Core models (current tree, strict load)

| slot | run | method / seed | sha256 = frozen list | strict load | class |
|---|---|---|---|---|---|
| TB-M3 | R-TB-K-1 | B1-K s1 | True | PASS | A_CURRENT_EVAL_EXACT_LOAD |
| TB-M5 | R-TB-DK-1 | B2 s1 | True | PASS | A_CURRENT_EVAL_EXACT_LOAD |
| TB-M6 | R-TB-E-0 | B1-K+E s0 | True | PASS | A_CURRENT_EVAL_EXACT_LOAD |
| TB-C1 | R-TB-E-1 | B1-K+E s1 | True | PASS | A_CURRENT_EVAL_EXACT_LOAD |
