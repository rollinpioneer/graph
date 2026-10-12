# SAW-RD-01 reproduction

Archive branch: `codex/saw-rd-01-20261012` in `rollinpioneer/graph`.
All execution scripts live under `experiments/saw_rd01_20261012`. Server working directory:
`/home/xushijie3/work/graph_saw_rd01_20261012/experiments/saw_rd01_20261012`.

Read `EXPERIMENT_CARD.md`, `REQUESTED_PLAN.md`, `FROZEN_RUN_APPENDIX.md`, and the resource/report results before launching any work. This document describes the archived run and does not authorize another training run. Do not overwrite existing `runs/formal_*` or `runs/final_*` directories.

Official source setup in an isolated location:

```bash
git clone https://github.com/johnlyzhou/saw.git vendor/saw
git -C vendor/saw checkout --detach 3535a16c901e5de1233ac5a586401b3ddfa5e30d
uv venv --python 3.10 vendor/venv
# Replace the editable absolute server path in the lock with the current official checkout.
sed '/^-e /d' requirements.lock.txt > vendor/requirements.portable.txt
uv pip install --python vendor/venv/bin/python -r vendor/requirements.portable.txt -e 'vendor/saw[train]'
```

The four original NPZ files are downloaded from
`https://rail.eecs.berkeley.edu/datasets/ogbench/<dataset-name>.npz`:
`pointmaze-large-navigate-v0`, `pointmaze-large-navigate-v0-val`,
`pointmaze-large-stitch-v0`, `pointmaze-large-stitch-v0-val`.
Store them in `vendor/data`, verify SHA256 against `results/DATA_AUDIT.json`.
Source paths, file sizes and hashes for raw evidence and checkpoints are indexed in the final `EVIDENCE_INDEX.json`.

CPU environment for audit/diagnosis:

```bash
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu MUJOCO_GL=egl
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
vendor/venv/bin/python qualify.py --mode audit
vendor/venv/bin/python data_analysis.py
vendor/venv/bin/python failure_map.py --mode calibrate
```

Authorized GPU job invocation, from the experiment directory (example Navigate):

```bash
export CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export MUJOCO_GL=egl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python3 resource_guard.py --label pilot_navigate --seconds 1800 -- vendor/venv/bin/python qualify.py --mode pilot --kind navigate --updates 1000
# Only after both data/config/resource gates pass:
python3 resource_guard.py --label formal_navigate --seconds 14400 -- vendor/venv/bin/python train.py --kind navigate
```

Stitch uses `--kind stitch`, distinct run/receipt labels, and GPU 3 in this archived run.
Both formal jobs independently start from seed 0, rather than the pilot state.
All job time, including CPU evaluation while the GPU model is resident, is charged to the GPU budget.

After both 1M checkpoints freeze, CPU-only final evaluation and discovery:

```bash
vendor/venv/bin/python final_eval.py --kind navigate
vendor/venv/bin/python final_eval.py --kind stitch
vendor/venv/bin/python diagnose.py --kind navigate --phase discovery
vendor/venv/bin/python diagnose.py --kind stitch --phase discovery
vendor/venv/bin/python failure_map.py --mode build
```

Do not run confirmation until the discovery evidence and the conditional intervention decision are recorded. Confirmation uses the preregistered fresh seeds and separate state-goal samples. At most 64 matched state-waypoint pairs (128 policy segments) across discovery and confirmation. Any extra segments or actor training require a separate budget decision.

Raw trajectories, model/optimizer/RNG checkpoints, NPZ datasets and complete logs remain on the server. Git contains scripts, protocol, dependency lock, light CSV/JSON metrics, diagnostics, reports, figures and hashes. This is a single-training-seed implementation evaluation; Stitch protocol gaps limit direct comparison to the authors' Appendix J.
