# Minimal geometry prototype proposal (NOT AUTHORIZED; Part B is a proposal only)

Status: PROPOSED_NOT_AUTHORIZED. approved_branch_attempts = 0. execute_now = false.

## V1 close-out (unchanged)

T_P_SOFT_RELOCATION_V1: qualification not established; provider/representation NOT_RUN (not observed zero).
Direct route succeeded in 46/48 physical branches; relocation in 26/48. The bins are centre-distance bins, camera
projected overlap was zero in nearly all configs, and no recorded field observes real gripper interference. V1 evidence is retained as-is.

## Single mechanism for V2 candidate

A movable interferer intrudes into the target's real gripper sweep region (approach, descend, close, lift). The interferer's own
grasp point remains reachable. Moving it to the buffer frees the sweep, so relocation can reduce direct-route interference.
The mechanism is defined on real gripper/object collision geometry and the control path, not on camera overlap or centre distance.
pool_33 is not a V2 positive example (its failure evidence was never recorded).

## Proposed new configs (4 only)

- 2 configs with expected path interference (interferer placed inside the finger/palm sweep of the target grasp).
- 2 matched offset controls (same geometry, interferer shifted outside the sweep).
- Asymmetric interferer shape/orientation may be used only if it matches gripper opening and grasp height;
  where the current controller cannot support it, mark "not implemented" instead of forcing it.

## Constraints

- Target publicly localizable at start; both first actions pass safety checks; both nominal routes completable.
- No new contract dead ends; no initial penetration, no disabled collisions, no teleport.
- No per-route timeout tweaks, random failures or safety removal; controller fixes are shared by both routes and all future methods.
- State differences from V1 are listed explicitly; V1 assets, runtime and observations are not overwritten.
- V1 scenes where direct succeeds and relocation costs may serve as low-opportunity control candidates; they are not verified harmful-prior data.

## Staged request (proposal)

20 branch attempts: 4 diagnostic (sensor/recording check) + 16 prototype. Requires separate approval. Provider, RL, optimizer,
elastic and standalone capture resets: 0. S2/S3/formal test/method upgrade not requested.
