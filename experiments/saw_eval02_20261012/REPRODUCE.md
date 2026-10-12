# SAW-EVAL-02 reproduction

Branch codex/saw-eval-02-20261012, parent RD01 6462e06c540594acca6bc8630c37a803fe36dd4a. Read EXPERIMENT_CARD.md and FROZEN_PROTOCOL.md first. This document records the authorized completed audit, not authorization for another run. No training command exists in this card.

Run from an isolated output directory. eval_core.py resolves immutable inputs from /home/xushijie3/work/graph_saw_rd01_20261012/experiments/saw_rd01_20261012; adapt OLD only when moving verified identical inputs. Official checkout must remain 3535a16c901e5de1233ac5a586401b3ddfa5e30d. Check two 1M checkpoint SHA256 values against FROZEN_PROTOCOL.md/EVIDENCE_INDEX.json. Reuse verified RD01 dependency lock or install it separately; never modify the original dependency environment for this audit. DEPENDENCIES.json records the actually loaded versions.

```bash
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
export MUJOCO_GL=egl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TQDM_DISABLE=1
export PYTHONDONTWRITEBYTECODE=1
PY=/home/xushijie3/work/graph_saw_rd01_20261012/experiments/saw_rd01_20261012/vendor/venv/bin/python
# Archive measured trial and pre-outcome SAMPLE_SIZE before formal evaluation:
"$PY" probe.py
# Only if timing-qualified; A must pass before B:
"$PY" run_a.py
"$PY" run_b.py --kind navigate
"$PY" run_b.py --kind stitch
"$PY" technical.py
"$PY" analyze.py
"$PY" report.py
```

The original round wall deadline is fixed in eval_core.py; a separately authorized repeat needs its own preregistered start/deadline, branch, output paths and identities. Existing raw files use exclusive creation and must not be overwritten. A uses actual official and project evaluator functions with replay-reset injection and a one-task presentation view; native official environment task list is intact. Matched task IDs/scenes/complete state/RNG are recorded. B only changes continuous vs per-episode reset flow; BOTH conditions omit official per-task NumPy action-key draws. technical.py evaluates that RNG boundary and extra video-designated reset/actions independently; render outputs are placeholders, no actual graphics context.

Four requested deliverables: EVALUATOR_AUDIT.md, PROTOCOL_SCORES.md, SCENE_DISTRIBUTIONS.md, PROTOCOL_VERDICT.md. All per-step A evidence and per-episode B rows are lightweight CSVs. All model weights, matched scenes/full trajectories and full logs remain server-only, identified in EVIDENCE_INDEX.json. archive.py checks parent evidence/weights/source identities and explicit staged paths, then writes LIGHT_MANIFEST.json. Post-push local receipt records verified branch hash, transport hash and SSH closure; it is excluded from the committed manifest to avoid self-reference.

Supplement: after report.py, run uncertainty_note.py to retain boundary-binomial caveats and supplementary score intervals; it changes no primary decision or sampled case.
