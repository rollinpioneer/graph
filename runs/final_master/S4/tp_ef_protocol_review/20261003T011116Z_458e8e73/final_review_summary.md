# CP-DISR-TP-EF-PROTOCOL-REVIEW-1 — final review summary

Zero environment, zero provider, zero RL, zero skill executions in this card. DISCOVERY-1 raw results accepted read-only.

## Pair types over the 20 saved T_B dev states

- **OPEN_vs_PICK_second**: {'CONTRACT_TIED_SCRIPTABLE': 20}; route A reachable in 20/20, route B in 20/20; offline gate PASS; first illegal step A=[''] B=['']; skills never observed succeeding: []
- **OPEN_vs_PICK_target**: {'CONTRACT_REVEALED_PAIR': 20}; route A reachable in 20/20, route B in 0/20; offline gate FAIL; first illegal step A=[''] B=['a:PLACE:target:container:v1']; skills never observed succeeding: []
- **PICK_target_vs_PICK_second**: {'CONTRACT_REVEALED_PAIR': 20}; route A reachable in 0/20, route B in 20/20; offline gate FAIL; first illegal step A=['a:PLACE:target:container:v1'] B=['a:PLACE:target:container:v1']; skills never observed succeeding: []

## Requested next protocol (PICK(target) vs PICK(second_object), no stepwise B_PLAN)

Status: **NOT_FROZEN_OFFLINE_GATE_FAILED**. requested continuation A starts PICK(target) then PLACE(target, container); PLACE requires Open(container) which is FALSE in all saved dev states, OPEN requires GripperEmpty, and PLACE_BUFFER does not restore OnTable, so no legal continuation of candidate A reaches the goal under the registered contracts

## Budget reconciliation (as recorded, no refund)

- Attempt 1 (STOPPED_ENGINEERING): env constructions 37, explicit resets 37, restores logged 8, skill executions 0, ledger 'used' 8.
- Attempt 2 (formal): env constructions 37, explicit resets 37, restores logged 8, skill executions 19, ledger 'used' 8.
- Cumulative: {'environment_constructions': 74, 'explicit_env_reset_calls': 74, 'skill_executions': 19, 'worker_processes': 16}

## Interpretation

See derived_interpretation.json: only the concrete DISCOVERY-1 protocol is rejected; GLOBAL_T_P is UNRESOLVED.
