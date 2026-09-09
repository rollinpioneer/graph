# L2RA-R1.7 Controller Attempt Lifecycle Contract

This interface adds controller-owned attempt context without changing the
hold-evidence candidate set or entering L3.

## Online fields

- `attempt_id`: monotonically increasing integer for each controller attempt.
- `attempt_phase`: `inactive`, `acquiring`, `settling`,
  `post_contact_motion`, or `ended`.
- `attempt_active`: whether the controller currently owns an active attempt.
- `attempt_end`: one-cycle post-update edge emitted by the controller.
- `attempt_end_reason`: `segment_complete`, `cancelled`, or `release`.

The end reason is lifecycle-only. It must not encode success, missed grasp,
attachment, or any other physical result. The controller emits the fields from
its current skill schedule; the event module does not inspect action names,
future actions, oracle state, or final outcomes.

## Event semantics

- Stable hold followed by non-release contact loss remains
  `held_object_loss` / `recover_object`.
- No stable hold plus contact loss with `attempt_end=false` or unknown remains
  `hold_unknown` / `needs_observation`.
- No stable hold plus closed gripper, no contact, complete history, and
  `attempt_end=true` becomes `retryable_missed_grasp` / `retry_grasp`.
- Missing history or missing lifecycle context remains three-valued unknown.

The cached R1 rows are regression inputs only. They do not establish the new
interface's family-level confirmation gates, and this contract does not
authorize L3, R4, training, or threshold expansion.
