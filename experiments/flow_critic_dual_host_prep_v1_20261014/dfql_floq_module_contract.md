# DFQL + floq Critic module contract

Frozen host: DFQL `8b6b15803cc1e5605ad1524553a2c95e103b0bad`.
Frozen value source: floq `3d60e638b42bbef018c56cb1199ff37c3470520d`.

`agents/floq_adapter.create_agent('dfql', ...)` returns the unchanged official constructor for `critic_type=scalar`. For `critic_type=floq`, it creates `floq`, `target_floq`, `critic`, and the original `actor_bc_flow` definition with `tanh_output=True`. The unused original one-step Actor and scalar target are absent from the combined path; they remain present in the exact original scalar path. No host utilities are overwritten.

| Key | Role | Parameter gradients |
|---|---|---|
| `floq` | Online expected-return velocity | Flow matching loss only |
| `target_floq` | Next-state Euler value backup | None |
| `critic` | Online scalar distilled readout | Scalar distillation only |
| `actor_bc_flow` | Original iterative DFQL policy | BC and terminal Q guidance only |

The original actor loss and policy functions are inherited unchanged: grid-time velocity BC, stop-gradient after every rollout step, step weights `1/(K-k)`, guidance coefficient `-2*alpha/K`, mean over step/batch/action dimensions, Q aggregation, normalization switch and action sampling RNG remain fixed.

The final rollout action is clipped first; Q is then differentiated with respect to that already clipped action. The derivative of a composite `Q(clip(raw_action))` is not substituted. Critic parameters remain closed-over frozen values while action-input derivatives supply the policy guidance. Synthetic non-flat Q tests distinguish these two paths.

TD actions come from the current DFQL multi-step policy. No floq/FQL one-step policy is called. Target Flow integration aggregates ensembles before the sample average; reward and mask are applied before averaging, matching the pinned source's reduction order. Scalar supervision comes from online Flow integration and is stopped before distillation.

DFQL alpha is a host terminal-gradient coefficient and remains at its official task value (Cube 2, Humanoid 10, Puzzle 3.3). Source floq Actor alpha is not inherited. There is no Q-Flow intermediate-value network and no unused EMA scalar readout.

The Flow EMA definition uses pre-optimizer parameters, with `floq→target_floq` only. Agent update calls are blocked for this preparation card and target math is tested on independent arrays. DFQL trained weights are `NOT_YET_TRAINED`; this does not block random initialization checks and does not establish Baseline reproduction.
