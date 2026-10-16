# User-approved all-parallel amendment

The user explicitly requested all twelve branches start without waiting. All 12 now have real fused4 update and complete-restore admission PASS. Completed branches are not restarted. Existing training processes were not interrupted.

The prior coordinator alone was suspended for a controlled adoption. code/all_parallel_manager.py adopts its children, reads their original Linux zombie wait status before retiring the old parent, and launches remaining branches once. Full process identities, dispatch and actual exit receipts are retained. Maximum 12 training + 2 ordinary evaluations, no fixed per-GPU job cap; >=10000 MiB free at launch, with a 9000 MiB reservation for each just-launched job until its allocation settles. Measured existing training allocation was about 6748 MiB. No batch size, precision, random seed, update count or algorithm parameter changed.

## Evaluation reporting failure and authorized repair

Nine evaluations completed all 100 episodes, but a checkpoint-path variable was overwritten by a Pipe Connection variable and the final SHA check raised TypeError before full results were persisted. This was an implementation bug. Their last progress receipts have completed=100 and exact control-call counts. Success/return trajectories were lost and are not reconstructed or invented. Original checkpoints and training are unaffected.

The path now has an unambiguous name; new evaluations have already written validated results. A pre-verification episode dump further protects future diagnostics. The user explicitly authorized exactly these nine evaluations to be repeated, 900 extra episodes, total physical control-call cap 6750000. code/repair_evaluation_manager.py manages these once, at most two concurrently, separately from ordinary evaluations. New directories end in _repair1; failed directories and logs are preserved. There is no new retry authorization if a repair fails. Training updates remain capped at 1200000.

Summaries use original successful results or the authorized repaired result, never count a condition twice, and add the nine failed executions' exact control costs. Both managers must complete before final reporting. Old B1a results remain historical context.
