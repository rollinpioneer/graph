# transport_recovery semantics

- Initial state: `start`; terminal success: `success`; terminal failure: `terminal_failure`.
- Complete history is required for recovery and completed-subgoal-set disambiguation.
- Legal paths: [['start', 'grasped', 'in_transit', 'placed', 'success'], ['start', 'grasped', 'in_transit', 'dropped_or_misaligned', 'recovery', 'grasped', 'in_transit', 'placed', 'success']]
- Failure is a state-supported invalidation, never an episode midpoint heuristic.
