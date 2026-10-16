# TD Target Drift Decomposition: fixed-checkpoint evidence

Analysis of seeds 0/3, arms R2/R3/R4/R5, existing fork/+50k/+100k checkpoints. No training, no optimizer updates, no environment construction/reset/step. All numbers average the two pre-fixed noise schedules; each schedule is separately retained in decomposition.json.

## Sampling and interface

2048 original processed training transitions from current-state start cells and 2048 outside, sampled without replacement with the same indices for all models. Outside is a comparison region, not a verified normal region or a posture-matched control. Target action is generated at next state; VaBC at current state and scored at next state, preserving target_use_next=false. Discount .995, mean ensemble and mean actor/flow aggregation preserved. All component swaps use the same state and random input. No trajectory-cluster significance claim.

## Reference drift versus accepted-increment drift: fork to +100k

MAE below means mean absolute CHANGE on the fixed inputs, not a prediction error against true value. Positive signed change means target increases (less negative in this reward convention). These changes span checkpoints and are not per-update jumps.

| seed | arm | region | reference signed drift | reference mean abs drift | accepted mean abs drift | total mean abs drift |
|---|---|---|---|---|---|---|
| 0 | R2 | start | 59.9818 | 60.0448 | 0.0000 | 60.0448 |
| 0 | R2 | outside | 1.2902 | 4.5012 | 0.0000 | 4.5012 |
| 0 | R3 | start | -0.0069 | 0.9002 | 0.0000 | 0.9002 |
| 0 | R3 | outside | -1.4312 | 1.8751 | 0.0000 | 1.8751 |
| 0 | R4 | start | 58.9249 | 59.1245 | 0.0650 | 59.1284 |
| 0 | R4 | outside | -1.0624 | 3.0666 | 0.0338 | 3.0717 |
| 0 | R5 | start | 63.7310 | 63.8630 | 0.1095 | 63.8368 |
| 0 | R5 | outside | -0.4934 | 4.5689 | 0.0506 | 4.5781 |
| 3 | R2 | start | -0.1140 | 11.8138 | 0.0000 | 11.8138 |
| 3 | R2 | outside | 7.7985 | 12.8437 | 0.0000 | 12.8437 |
| 3 | R3 | start | 9.4947 | 23.4910 | 0.0000 | 23.4910 |
| 3 | R3 | outside | -7.7926 | 7.9507 | 0.0000 | 7.9507 |
| 3 | R4 | start | 3.9679 | 13.3670 | 0.0297 | 13.3614 |
| 3 | R4 | outside | 5.7691 | 12.4440 | 0.0383 | 12.4441 |
| 3 | R5 | start | -7.9750 | 15.0119 | 0.0285 | 15.0148 |
| 3 | R5 | outside | 3.8519 | 11.6381 | 0.0632 | 11.6395 |

## Old-anchored factorial signed decomposition: fork to +100k

A=replace reference Actor only; V=replace VaBC only; C=replace Target Critic only. Main effects hold other factors at the earlier checkpoint. AC/VC are interactions; AV and AVC should vanish under the additive target formula. Values sum exactly to total reference drift per state; averaging preserves this identity. Main effects depend on the chosen anchor. Shapley values distribute interactions symmetrically as a secondary accounting convention, not a causal explanation.

| seed | arm | region | A | V | C | AC | VC | interaction mean abs sum |
|---|---|---|---|---|---|---|---|---|
| 0 | R2 | start | -0.0209 | -0.0069 | 59.0947 | 0.8174 | 0.0975 | 1.0149 |
| 0 | R2 | outside | -0.0220 | -0.0156 | 1.2059 | 0.1039 | 0.0180 | 0.1858 |
| 0 | R3 | start | 0.0006 | -0.0002 | -0.0175 | 0.0094 | 0.0008 | 0.0348 |
| 0 | R3 | outside | -0.0073 | -0.0080 | -1.4296 | 0.0100 | 0.0037 | 0.0610 |
| 0 | R4 | start | -0.0236 | -0.0063 | 58.0754 | 0.7857 | 0.0937 | 0.9793 |
| 0 | R4 | outside | -0.0173 | -0.0136 | -1.0934 | 0.0520 | 0.0099 | 0.1193 |
| 0 | R5 | start | -0.0325 | -0.0080 | 63.1561 | 0.5292 | 0.0861 | 0.6868 |
| 0 | R5 | outside | -0.0237 | -0.0165 | -0.5530 | 0.0846 | 0.0152 | 0.1594 |
| 3 | R2 | start | -0.0808 | -0.0165 | -0.1747 | 0.1308 | 0.0271 | 0.2120 |
| 3 | R2 | outside | -0.0632 | -0.0176 | 7.7413 | 0.1095 | 0.0285 | 0.1975 |
| 3 | R3 | start | -0.0172 | -0.0070 | 9.3080 | 0.1916 | 0.0193 | 0.4102 |
| 3 | R3 | outside | -0.0108 | -0.0017 | -7.8199 | 0.0269 | 0.0129 | 0.0937 |
| 3 | R4 | start | -0.0844 | -0.0143 | 3.8688 | 0.1657 | 0.0322 | 0.2601 |
| 3 | R4 | outside | -0.0561 | -0.0170 | 5.7057 | 0.1096 | 0.0269 | 0.1975 |
| 3 | R5 | start | -0.0741 | -0.0156 | -8.0426 | 0.1348 | 0.0225 | 0.2149 |
| 3 | R5 | outside | -0.0441 | -0.0126 | 3.7324 | 0.1446 | 0.0316 | 0.2437 |

## Slow EMA R3 across intervals

| seed | region | interval | reference signed drift | reference mean abs drift | A signed | V signed | C signed | AC signed | VC signed |
|---|---|---|---|---|---|---|---|---|---|
| 0 | start | 0->50000 | -0.3211 | 0.5364 | 0.0010 | 0.0028 | -0.3255 | 0.0012 | -0.0006 |
| 0 | outside | 0->50000 | -0.8609 | 1.6236 | -0.0041 | -0.0079 | -0.8544 | 0.0074 | -0.0019 |
| 0 | start | 50000->100000 | 0.3141 | 0.6352 | 0.0020 | -0.0029 | 0.3096 | 0.0051 | 0.0003 |
| 0 | outside | 50000->100000 | -0.5703 | 1.0678 | -0.0011 | 0.0015 | -0.5737 | 0.0036 | -0.0005 |
| 0 | start | 0->100000 | -0.0069 | 0.9002 | 0.0006 | -0.0002 | -0.0175 | 0.0094 | 0.0008 |
| 0 | outside | 0->100000 | -1.4312 | 1.8751 | -0.0073 | -0.0080 | -1.4296 | 0.0100 | 0.0037 |
| 3 | start | 0->50000 | -19.3523 | 19.6665 | -0.0505 | -0.0127 | -19.3859 | 0.0829 | 0.0138 |
| 3 | outside | 0->50000 | -5.1117 | 5.3367 | -0.0042 | 0.0024 | -5.1285 | 0.0135 | 0.0051 |
| 3 | start | 50000->100000 | 28.8470 | 30.0451 | 0.0145 | -0.0057 | 28.5136 | 0.2670 | 0.0576 |
| 3 | outside | 50000->100000 | -2.6810 | 3.1656 | -0.0035 | -0.0030 | -2.6913 | 0.0136 | 0.0032 |
| 3 | start | 0->100000 | 9.4947 | 23.4910 | -0.0172 | -0.0070 | 9.3080 | 0.1916 | 0.0193 |
| 3 | outside | 0->100000 | -7.7926 | 7.9507 | -0.0108 | -0.0017 | -7.8199 | 0.0269 | 0.0129 |

## Necessary validation

{
  "passed": true,
  "branches": [
    {
      "seed": 0,
      "arm": "R2",
      "unchanged_full_state": true,
      "unchanged_checkpoint_files": true,
      "optimizer_updates": 0,
      "environment_steps": 0
    },
    {
      "seed": 0,
      "arm": "R3",
      "unchanged_full_state": true,
      "unchanged_checkpoint_files": true,
      "optimizer_updates": 0,
      "environment_steps": 0
    },
    {
      "seed": 0,
      "arm": "R4",
      "unchanged_full_state": true,
      "unchanged_checkpoint_files": true,
      "optimizer_updates": 0,
      "environment_steps": 0
    },
    {
      "seed": 0,
      "arm": "R5",
      "unchanged_full_state": true,
      "unchanged_checkpoint_files": true,
      "optimizer_updates": 0,
      "environment_steps": 0
    },
    {
      "seed": 3,
      "arm": "R2",
      "unchanged_full_state": true,
      "unchanged_checkpoint_files": true,
      "optimizer_updates": 0,
      "environment_steps": 0
    },
    {
      "seed": 3,
      "arm": "R3",
      "unchanged_full_state": true,
      "unchanged_checkpoint_files": true,
      "optimizer_updates": 0,
      "environment_steps": 0
    },
    {
      "seed": 3,
      "arm": "R4",
      "unchanged_full_state": true,
      "unchanged_checkpoint_files": true,
      "optimizer_updates": 0,
      "environment_steps": 0
    },
    {
      "seed": 3,
      "arm": "R5",
      "unchanged_full_state": true,
      "unchanged_checkpoint_files": true,
      "optimizer_updates": 0,
      "environment_steps": 0
    }
  ],
  "max_native_target_abs_error": 2.3572027203044854e-05,
  "max_reconstruction_abs_error": 0.0,
  "additive_actor_vabc_and_triple_interactions_zero": true,
  "synthetic_terminal_pass": true,
  "paired_noise_repeats": 2,
  "total_sampled_transitions": 4096,
  "region_counts": [
    2048,
    2048
  ],
  "optimizer_updates": 0,
  "environment_steps": 0
}

## Interpretation limits

Existing seed-3 BFC anomalies occurred around +30k; no +30k checkpoint is available. At +50k the system is already after that event. Endpoints cannot reconstruct an unrecorded transient peak, establish temporal precedence, or prove that the largest changing component was wrong. Seed-0 and seed-3 forks differ in training age (600k vs 300k) as well as parameters. A changing Critic may reflect policy/bootstrapping feedback or legitimate learning; no true long-horizon action values are measured here. Small direct policy effects under a frozen Critic do not rule out policy-driven effects mediated by Critic training.

For the two BFC arms, accepted increment is always bounded by epsilon at an endpoint, so its between-endpoint change is at most 2 epsilon. Reference drift is unconstrained by this formula. A large reference/accepted ratio alone is partly structural; localization and the component-swapping results provide the additional evidence.

Interaction magnitudes, per-state quantiles, signed/absolute/RMS statistics and both noise repeats are retained in decomposition.json and compact_summary.json. Raw Q cross-products and exact fixed inputs remain on the server under arrays/. No model/data/checkpoint was overwritten.
