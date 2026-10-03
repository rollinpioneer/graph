# Failure analysis (offline, written after the run; raw results untouched)

Stop: `STOPPED_RESTORE_FAIL:T_B_dev_03` - fresh-process restore of the first snapshot failed the frozen gating check `forced_render_obs`. T_B_dev_05 was not attempted (stop rule).

What matched exactly after loading the saved bytes into a new process and a new environment: simulator time/qpos/qvel/act/ctrl/mocap/qacc_warmstart (max abs 0.0, before and after forward), controller and gripper state (314 python-state entries, 0 differences, named goal_pos/goal_ori/new_update/torques/current_action all equal), numpy and python RNG hashes, episode clock, facts (cached and forced-render), goal, candidate ids and mask, Open/GripperEmpty/OnTable(target)/OnTable(second) = TRUE, both PICK masks TRUE, cached public observation (rgb exact, depth and proprio 0.0), and the depth and proprioception of the forced re-render (max abs 0.0).

What failed: the RGB of the forced re-render was not bit-identical (`rgb_exact=false`; the frozen criterion for RGB was exact equality). The restored RGB array was not stored by the validation, so the size and location of the difference is NOT measured. Depth being exactly equal and every fact being equal indicates the simulator state was restored; whether the RGB difference is rendering non-determinism across processes or a residual state difference is not established by this card.

Offline context from the saved setup arrays (same process): cached vs forced RGB differ in 1700 pixels (max 83), depth in 1139 pixels (max 0.016), proprioception in 7 entries (max 5e-4): the cached observation is a stale sampled frame, so cached and forced observations are intentionally distinct objects.

Per the approval terms the criterion is not relaxed after the fact; this card is a FAIL and nothing further is started.
