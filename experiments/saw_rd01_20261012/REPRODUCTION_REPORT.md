# REPRODUCTION_REPORT

Official SAW + local OGBench source: `3535a16c901e5de1233ac5a586401b3ddfa5e30d`. Full default network/batch/config, independent fresh training seed 0 per dataset, 1,000,000 joint updates each. Complete 100k/300k/500k/1M checkpoints include optimizer and all stochastic training state. Qualification pilot checkpoints are never reused for formal training.

| Baseline | Status | Fresh final success | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|---|---|---:|---:|---:|---:|---:|---:|
| navigate | SINGLE_SEED_BASELINE | 62.8% | 100.0% | 26.0% | 88.0% | 100.0% | 0.0% |
| stitch | SINGLE_SEED_BASELINE + PROTOCOL_APPROXIMATION | 49.6% | 100.0% | 0.0% | 88.0% | 60.0% | 0.0% |

Final confirmation uses 5 tasks × 50 episodes per model, temperature 0, CPU evaluation, horizon 1000 and unchanged official success tolerance. Final seeds Navigate 91001 / Stitch 91002 are independent of development evaluation. Raw initial/goal/full trajectories are retained. This is a single-training-seed evaluation, not a cross-seed paper reproduction.

Stitch uses preregistered Large Navigate defaults. No separate official SAW Stitch command or Appendix G hyperparameter row was found in the finite protocol audit. Inferred items: actor future-only goal distribution, 25-step subgoal horizon, temperatures 3/3/3, expectile .7, model/config defaults. The local OGBench source is explicitly fixed rather than silently upgraded through PyPI. Independent evaluator RNG isolation, deterministic zero-temperature mode, local logging, extra checkpoints and video omission are recorded instrumentation choices.

Tests passed: dataset original identities and finite fields, valid transitions/trajectory bounds, exact original sampler rewards/masks/subgoal indices, independent sampler/validation RNG, exact pilot parameter/optimizer/JAX RNG equality with logging on/off, true gradient/optimizer updates, synchronized cost timings, unchanged evaluator success/non-success/short-positive controls, and hard aggregate memory/budget gates.

Development evaluations are retained in `results/*_development_eval.json`; raw development trajectories are not recycled as final confirmation. All losses are in `results/*_training.csv`. Source code versions, data hashes and raw checkpoint/trajectory paths are in `EVIDENCE_INDEX.json`.
