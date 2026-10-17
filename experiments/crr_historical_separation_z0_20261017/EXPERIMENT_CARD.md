# CRR-HISTORICAL-SEPARATION-Z0 — execution freeze

Authorized by user: execute the supplied historical necessary-condition screen.

Branch: [codex/crr-historical-separation-z0-20261017](https://github.com/rollinpioneer/graph/tree/codex/crr-historical-separation-z0-20261017).

Only read existing task1 s0 600k→700k and s3 300k→400k B0/B1a weights. All 10k checkpoints exist: 42 distinct files, 44 logical arm/time models. No training, initialization, optimizer use, environment construction/reset/step, new episodes or missing-timepoint reconstruction.

B0 uses the current online Actor in TD; B1a holds the fork Actor parameters fixed only for TD. Both continue online Actor/VaBC and Critic learning historically. This is not the prior EMA R3 experiment. Saved plain GFP payloads contain the Critics needed here; frozen B1a generator sidecar is not present and complete B1a resumability is not claimed.

4096 fixed actual-data transitions: 2048 start and2048 nonstart from the prior FACB fixed pool. Two fixed Gaussian arrays; fork Actor(next state) and VaBC(current state) generate actions once. Target scores average Critic heads and then the two fixed action sources. Online raw action-gradient uses ensemble mean at the clipped fork Actor action, not guidance_agg=min. Region is determined by next-state xy; no changed target_use_next semantics. Two preselected balanced halves provide sensitivity without extra states.

Deliver three plot groups: DeltaC, fixed-action Q differences/drift, online gradient ratio. Preserve per-arm, per-region and per-latent numbers and paired trajectory-cluster descriptive intervals. No retrospectively fitted risk threshold. Shared-data/latent repeats do not provide independent training confirmation.

Freeze original historical alarm markers separately: s0+80k(start onlineV rise20), s3+40k(start-minus-all rise15). Also display finer existing5k onset, especially s3 joint alarm+35k. A+30k difference is only5k before+35k; do not claim10k warning by rounding. Historical probe actions and start-minus-all contrasts differ from the new fixed-action start-minus-nonstart score.

Behavioral evaluation is reused only. s3 was already weak at fork(9%); sparse later evaluations cannot locate the onset of success decline or establish prospective prediction. NO new behavior evaluation to fill gaps.

GO would permit only recommending a separately scoped Critic-only response screen; no gate training is authorized. Late/one-trajectory separation lowers priority. Temporal insufficiency remains INCONCLUSIVE rather than changing the window.

Model/file identity, counts, target mapping, finite values, fork equality and zero-operation counters must pass. Record actual CPU/GPU wall cost and true exit. Full arrays/weights stay server-side; explicit lightweight archive, ordinary commit/push and SHA verification, local handoff, close this round's sole SSH session.
