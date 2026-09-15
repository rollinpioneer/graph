# Independent constructed-state holdout

independent_state_holdout_passed: true
representation_confirmation: SUPPORTED_ON_INDEPENDENT_STATE_HOLDOUT
reward_accounting_confirmation: SUPPORTED_ON_INDEPENDENT_STATE_HOLDOUT
confirmation_passed: false

This holdout is STATE_CONDITIONED_HOLDOUT. It is not physics, vision, policy, or robot confirmation.
Generator did not import V6 reward. Scoring used frozen methods after generator commit/hash lock.

## Gates
- A_legal_order: PASS
- B_aliasing: PASS
- C_credit: PASS
- D_exact_state_cycles: PASS
- E_potential_identity: PASS

## Gate A legal order
- family 990100: H1=0.75 H2=0.75 |diff|=0 passed=True
  A-first/B-first on H1: 1.0 / 1.0; on H2: 1.0 / 1.0
- family 990101: H1=0.75 H2=0.75 |diff|=0 passed=True
  A-first/B-first on H1: 1.0 / 1.0; on H2: 1.0 / 1.0
- family 990102: H1=0.75 H2=0.75 |diff|=0 passed=True
  A-first/B-first on H1: 1.0 / 1.0; on H2: 1.0 / 1.0
- family 990103: H1=0.75 H2=0.75 |diff|=0 passed=True
  A-first/B-first on H1: 1.0 / 1.0; on H2: 1.0 / 1.0

## Gate B aliasing
- P1HOLD_F990100__H3: count_same=True graph_diff=True events_diff=True passed=True
- P1HOLD_F990101__H3: count_same=True graph_diff=True events_diff=True passed=True
- P1HOLD_F990102__H3: count_same=True graph_diff=True events_diff=True passed=True
- P1HOLD_F990103__H3: count_same=True graph_diff=True events_diff=True passed=True
- P1HOLD_F990100__H4: count_same=True graph_diff=True events_diff=True passed=True
- P1HOLD_F990101__H4: count_same=True graph_diff=True events_diff=True passed=True
- P1HOLD_F990102__H4: count_same=True graph_diff=True events_diff=True passed=True
- P1HOLD_F990103__H4: count_same=True graph_diff=True events_diff=True passed=True

## Gate C credit
- H5 P1HOLD_F990100__H5: -0.3333333333333333 passed=True
- H5 P1HOLD_F990101__H5: -0.3333333333333333 passed=True
- H5 P1HOLD_F990102__H5: -0.3333333333333333 passed=True
- H5 P1HOLD_F990103__H5: -0.3333333333333333 passed=True
- H6 P1HOLD_F990100__H6: 0.0 passed=True
- H6 P1HOLD_F990101__H6: 0.0 passed=True
- H6 P1HOLD_F990102__H6: 0.0 passed=True
- H6 P1HOLD_F990103__H6: 0.0 passed=True
- H8 P1HOLD_F990100__H8: 0.0 passed=True
- H8 P1HOLD_F990101__H8: 0.0 passed=True
- H8 P1HOLD_F990102__H8: 0.0 passed=True
- H8 P1HOLD_F990103__H8: 0.0 passed=True

## Gate D exact state cycles
n_cycles=12
all passed

## Gate E potential identity
n=32 max_abs_error=1.6653345369377348e-16

## Incremental value (observed, not preset win)
- current_valid_subgoals: count=True events=True graph=True (valid flags are visible to all three)
- loss_recovery_events: count=False events=True graph=True (H3/H4 event keys differ; count keys do not)
- active_object_identity: count=False events=True graph=True (product-state keeps object-phase pairs)
- phase_structure: count=False events=True graph=True (PathGraph node includes phase; events are token-based)
- multiple_legal_orders: count=False events=False graph=True (H1/H2 both legal; count does not encode order)
- remaining_graph_cost: count=False events=False graph=True (V6 capability units / GRAPH_COST_ONLY if legacy bound; this holdout uses constructed legacy=0)
- unified_state_reward: count=False events=False graph=True (Gate E identity is V6 potential, not count/event)

## Out of scope
- physical closed loop
- vision
- policy training
- real robot
