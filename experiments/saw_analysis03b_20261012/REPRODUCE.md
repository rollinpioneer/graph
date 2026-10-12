# SAW-ANALYSIS-03B reproduction record

This records the authorized round; it does not authorize another run. Branch: codex/saw-analysis-03b-20261012. All outputs belong to its experiments/saw_analysis03b_20261012 directory. Keep existing results immutable. Read REQUESTED_CARD.md, FROZEN_RUN_APPENDIX.md, USER_STEERING.md and E3_SELFTEST_APPENDIX.md first.

Use the verified RD01 Python environment and fixed official SAW commit3535a16c901e5de1233ac5a586401b3ddfa5e30d. common.py resolves OLD to the immutable RD01 experiment for official imports and four raw datasets. Adapt input paths only after verifying identical data/source/checkpoint hashes. Requirements lock is inherited from RD01; DEPENDENCIES.json records actual runtime. EVAL02 complete-state raw traces are the immutable E2 inputs.

The original run used a single keepalive SSH shell, with four CPU threads per process, OpenBLAS/MKL one thread, CUDA preallocation disabled and CPU-only diagnostics. CPU environment:

```bash
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false'
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl
PY=/home/xushijie3/work/graph_saw_rd01_20261012/experiments/saw_rd01_20261012/vendor/venv/bin/python
```

Execution order in a new separately authorized clean directory:

1. Audit input hashes/config/sampler with qualify.py audit. Generate exclusive-create scenes.py files before outcomes and freeze their identities in Git.
2. Original round ran exactly1000-update pilots on GPUs2/6; user then explicitly instructed immediate formal execution, removing the estimated-throughput admission gate. No additional benchmark was run. Each formal guard starts a fresh seed0 child, never pilot/A weights:

```bash
CUDA_VISIBLE_DEVICES=3 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false "$PY" resource_guard.py --label formal_navigate --seconds 9000 -- "$PY" train.py --kind navigate
CUDA_VISIBLE_DEVICES=4 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false "$PY" resource_guard.py --label formal_stitch --seconds 9000 -- "$PY" train.py --kind stitch
```

3. Run each guard in an independent monitored background process. Immediately run A E1 evaluation (evaluate.py --kind KIND --variant A --phase E1), local_branches.py, support_value.py prepare; then support_value.py A and verified_candidates.py --variant A. CLOCKS.json must record a fresh actual4-hour CPU analysis deadline for a new authorized round. Existing completed CLOCKS is not reusable authorization.
4. After both training exits/checkpoint identity checks, evaluate B E1 on available registered stages. Run local_branches_b.py first, if Navigate-B has a complete1M, using only the four previously fixed A gate-qualified failures (E2_B_SUPPLEMENT.md); then support_value.py B / verified_candidates.py --variant B, with gradient triggers based on B's own physical branches. Missing1M stays missing. No complete endpoint for undertrained B. Both complete1M weights use final phase scenes and the same evaluation command with --phase final for A/B independently, plus a recorded final evaluation clock. Never select best stage.
5. replay_initial_audit.py includes initial state error in data-chain join gate using already saved measurements. analyze.py, plots.py, report.py and verify_assets.py synthesize the deliverables and verify original assets. archive_light.py validates/stages its explicit own-directory lightweight list; inspect the staged diff, commit/push only this branch, verify ls-remote equals HEAD and sync local handoff. Exit SSH only after own jobs have stopped naturally or safely saved via guard and exit receipts were read.

Checkpoint files contain full serialized agent/target/optimizer/JAX RNG, Python/NumPy/environment/action RNG and environment fields. Full weights, scene lists, complete trajectories, feasibility/selftest evidence and logs remain server-only, hashed in EVIDENCE_INDEX.json. Git holds source, frozen designs, score/transition/signal tables, necessary receipts and publication figures. Missing/failed/stopped conditions are retained. Native A/B1M final scores are current single-seed recipe sensitivity, not cross-training-seed evidence or recovered author Stitch protocol.
