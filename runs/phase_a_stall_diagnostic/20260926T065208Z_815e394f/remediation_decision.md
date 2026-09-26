# Remediation decision

1. **Task-data defect:** all 84 frozen T_C train/dev cases have initial target/interferer AABB overlap with the production half extent 0.020 m. `T_C_dev_04` has x/y overlaps 0.0081138568/0.0193293959 m; `T_C_train_01` has 0.0095378463/0.0172831078 m. Contact audit reports negative target/interferer distances.
2. **Do not rerun Phase A.** Produce a new non-overlapping T_C split/cache generation. Preserve intermediate-relocation semantics, enforce a positive minimum surface gap, recapture RGB/depth, invalidate/regenerate affected cache, repeat semantic reset and decision-structure audits, and freeze new source/split/cache hashes.
3. **Resource finding:** the failed pair used a shared, heavily loaded GPU and high thread count. Isolated bounded runs with one process and OMP/MKL/OpenBLAS=1 progressed normally; no two-process test was run.
4. No reward, H, d_ref, Method 2.1.1 or VLM changes are authorized by this diagnosis.
