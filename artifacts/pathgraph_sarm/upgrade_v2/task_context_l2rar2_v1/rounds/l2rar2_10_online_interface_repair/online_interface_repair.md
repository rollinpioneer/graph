# L2RAR2 online interface repair

- Version: `l2rar2_online_interface_repair_v1`.
- Scope: `32` frozen attach-relpose repair rollouts; no new sampling, training, API calls, or confirmation.
- The interface now records three causal states: `data_missing_or_invalid`, `valid_no_current_hold_evidence`, and `historical_hold_established`.
- Retry is permitted only for a valid, complete `HOLD_OBJECT` attempt ending without current hold evidence.
- Active touch/acquisition does not retry. A non-release contact loss recovers only after a historical hold was established and the loss is actually observed online.
- In this cache, K4/K5/K6 contain offline contact-loss events but no corresponding online contact-loss transition; the interface therefore does not claim recovery for them.
- Timing audit definitions: `t_physical` is the first offline `contact_lost` event; `t_observable` is the earliest permitted online evidence sufficient to indicate that event, operationalized in this cache as the first online contact true-to-false transition; `t_decision` is the first online emergency decision in the frozen window.
- The recorded latency identity is `t_decision - t_physical = (t_observable - t_physical) + (t_decision - t_observable)`; offline events are used for timing audit only and never as online input.
- Timing audit rows: `24` current attach-relpose rows plus `4` historical context rows; historical rows are excluded from current metrics.
- This is a diagnostic interface repair, not a passing candidate or confirmation result. `G1` remains retained and L3 remains closed.

## Metrics

- `B_count2`: K1=1.0, K4=0.0, K5=0.0, K6=0.0, K2/K3/K7/K8 false-emergency rates=[0.0, 0.0, 0.0, 0.0]; candidate_pass=`false`.
- `C3_vector_rho035`: K1=1.0, K4=0.0, K5=0.0, K6=0.0, K2/K3/K7/K8 false-emergency rates=[0.0, 0.0, 0.0, 0.0]; candidate_pass=`false`.
