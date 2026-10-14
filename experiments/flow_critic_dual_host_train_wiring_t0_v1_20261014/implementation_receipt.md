# T0 training wiring and admission receipt

**T0_ADMITTED.** Both combinations now have controlled real optimizer updates, Target write-back and native-main creation/logging/save/restore routes. This is engineering admission, not performance reproduction or stability admission.

GFP+floq: HumanoidMaze-medium task1, seed0, 2 real updates. DFQL+floq: Cube-double task2, seed0, 2 real updates. Total **4 actual optimizer updates**, zero evaluation episodes. Each used its official host alpha/discount/policy loss and the frozen floq value core. No Baseline run, short stability run, online learning or 1M experiment was started.

Model step advanced 1→2→3 and Adam count 0→1→2. Required trainable groups changed and target gradients/moments remained zero. EMA used pre-optimizer online parameters: first-step target movement was rounding only (~7.45e-9); second-step movement was ~1.55e-6 and distinguishable from an incorrect post-optimizer update. Input Agents remained immutable. Full-state checkpoints were restored between real steps; fresh-template restoration and model/Adam/JAX/NumPy/Python RNG round-trips subsequently passed without extra updates.

Native main defaults reject before setup without an execution gate. T0 caps reject all four public update APIs after the two-step allowance. Original scalar baseline class/code behavior remains intact. Batch/fused optimizer programs compile; this card did not execute additional batch/fused updates and does not claim their long-run stability. CPU dataset use is specific to this two-step native check; no throughput inference is made.

Engineering repairs: the isolated copied core contained a read-only file (copy permissions repaired only in the new workspace); DFQL's source logger forced online WandB despite the disabled environment (new explicit logger mode defaults to disabled; first attempt stopped at initialization with 0 updates); state fingerprints based on raw serialized dict ordering falsely rejected fresh templates (verified all saved/restored leaves identical, switched to canonical ordered/numeric fingerprints while retaining exact checkpoint byte SHA). Old raw fingerprints stay in historical receipts; old native weight files were not altered. CPU dataset/Python RNG sidecars are added to the final IO API and verified independently on the post-T0 state.

DFQL's supervised corrected launch exited 0. GFP's initial launcher did not preserve a process exit code; its two native validated updates, two native log/checkpoint rows, completed 2/2 output, absence of traceback and ended process were verified. This receipt does not invent an exit code; the retained supervisor handles future launches.

Old preparation branch/files, 44 old checkpoints and all source dataset files remain unchanged. The two missing research reports are not used as a training blocker. Existing metadata/loss/math remain reproducible from pinned sources.

**T1, T2 and T3 remain NOT_AUTHORIZED.** Next: freeze new-task G0/D0 Baseline protocols and obtain T1 approval; separately approve T2 stability runs after T0; only inspect T1/T2 results before approving 1M/seeds0/1 T3. Freeze evaluation episode counts, total updates, seeds, log/checkpoint cadence and GPU resources before formal runs. No score, superiority, convergence or novelty conclusion follows from this four-update engineering test.

Root: `/home/xushijie3/work/flow_critic_dual_host_train_wiring_t0_v1_20261014`.
Branch: `codex/flow-critic-dual-host-train-wiring-t0-v1-20261014`.
Branch URL: https://github.com/rollinpioneer/graph/tree/codex/flow-critic-dual-host-train-wiring-t0-v1-20261014
Large model/RNG checkpoints and full logs stay on the server; the archive contains code, configurations, patches, per-step results and identity receipts only.
