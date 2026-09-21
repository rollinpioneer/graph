# B2 abort: cumulative empty-episode counter

Time: 2026-09-21T14:04:51Z
Error: BindingError: Runtime repeatedly exposes no executable skill

The smoke runner incremented empty_episodes for every NO_CANDIDATE_SAFE_TERMINATION
and never reset it. Eight scattered empty episodes (ep-4,6,11,15,17,21,26,32)
triggered the abort even though later episodes still produced real transitions
and TASK_SUCCESS.

This is not 8 consecutive empty episodes. MAX_EMPTY is now a consecutive streak.
NO_CANDIDATE_SAFE_TERMINATION itself is a legal safety end and remains logged.
