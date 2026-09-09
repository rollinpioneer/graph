# K8 lifecycle correction

The development data in `data/dev_fit` and `data/dev_select` was regenerated with
`K8_acquisition_touch_then_continue.end_action=verify`. The earlier generation
lock still recorded `lift`; it is preserved as
`locks/generation_lock_pre_correction_k8_lifecycle_20260909.json` and is not a
current input. The canonical `locks/generation_lock.json` was regenerated from
the current collector and retains the same family IDs, seeds, physical specs,
and 160-rollout plan. No new rollout was created by this lock repair.
