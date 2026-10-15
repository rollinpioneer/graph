# transport_dual_order semantics

- Initial state: `start`; terminal success: `success`; terminal failure: `terminal_failure`.
- Complete history is required for recovery and completed-subgoal-set disambiguation.
- Legal paths: [['start', 'A_done', 'B_done', 'success'], ['start', 'B_done', 'A_done', 'success']]
- Failure is a state-supported invalidation, never an episode midpoint heuristic.
