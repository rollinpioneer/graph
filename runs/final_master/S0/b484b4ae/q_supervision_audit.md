# Q supervision audit

- `Policy.forward` computes Q values for the candidate set.
- `PPO.update` indexes `out.q[selected]` using `Transition.selected_candidate_id`; no target is fabricated for unexecuted candidates.
- `q_targets()` and `value_targets()` detach their targets. `ppo_losses()` detaches Q and V targets again at the loss boundary.
- Existing `test_q_executed_only.py` and `test_q_target_detach.py` passed.
- No relation truth, oracle relation, evaluator result, R*, or Q_ref sidecar is imported by this production objective path.

`MODEL_Q_PROXY` remains an analysis label only; this audit does not promote it to true all-candidate utility.
