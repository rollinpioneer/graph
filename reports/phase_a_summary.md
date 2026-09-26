# CP-DISR v1.2 Phase A First-Evidence

Status: NEEDS_RERUN

- phase_a_e16_ready: false
- phase_a_status: NEEDS_RERUN
- authorized jobs: v12_E16_T_C_Full_s0, v12_E16_T_C_B2_s0
- training source commit: 8c40dcd8ea424a959e05c2e898a4e9d8efeb0470
- stamp/configsha8: 20260926T050457Z/f0828f34
- physical GPU: 2 (shared; existing workload retained)

## Hard stop

Both workers initialized and published only `n_000000`, then remained for approximately 44 minutes in the MuJoCo collision/forward path (`mj_filterSphere` / `mj_collideTree` / `mj_forwardSkip`) without producing a transition, first-update self-check, or PPO update. They were terminated at the safe pre-update boundary. The pair gate recorded missing first-update evidence for both methods.

No formal checkpoint beyond n_000000 is claimed. No automatic restart was performed.

## Cost ledger

- formal RL jobs started: 2 (both stopped before update)
- real-environment PPO updates: 0
- diagnostic optimizer steps: 0
- Full/B2 no-learning probes: 0
- VLM requests: 0
- final test episodes: 0

A rerun requires explicit remediation/authorization for the MuJoCo initial-rollout stall and should use a safe resource allocation; this run is not a first-evidence result.
