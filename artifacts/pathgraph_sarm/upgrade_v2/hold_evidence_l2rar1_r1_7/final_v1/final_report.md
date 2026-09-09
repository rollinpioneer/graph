# L2RA-R1 final report

- Scientific status: **L2RAR1_PARTIAL_KEEP_G1**
- Retained graph: `G1_predicate_bound`; L3 entry remains closed.
- Valid development execution: 16 fit families / 64 rollouts and 16 select families / 64 rollouts, with 128 distinct online content groups.
- Invalid earlier batch: rejected for duplicate control programs, retained only as generator-contract audit evidence, and counted as 0 scientific rollouts.
- Development route: `DEVELOPMENT_NOT_READY`; 0/8 candidates eligible; selected candidate: `None`.
- Best dense-stream result by preregistered ordering: `B_count1`: miss 1.0 (8/8 correct), regular/brief/long-gap loss 1.0/1.0/1.0, wrong-or-unknown 0.0.
- Confirmation: `NOT_RUN_DEVELOPMENT_FAILED`; standard `NOT_RUN`, challenge `NOT_RUN`. R4 consumed no family because development failed.
- Failed development gate fields observed across candidates: `hold_evidence_and_retention, per_negative_stratum`. Legacy compatibility was not evaluated because every candidate had already failed primary event gates.
- Callback equivalence and dense-stream timestamp normalization passed; normalization reexecuted 0 physical rollouts.
- API calls: `0`; training jobs: `0`; API keys read: `false`.

The old magnitude-only co-motion score is direction-blind, but the preregistered direction-aware C3 candidate tied B_count2 on the primary dense stream rather than establishing feature gain. Dense and action-end sampling exchanged missed-grasp and loss recall, so sampling gain was not established either. The unresolved boundary is causal: transient contact before a physically recorded missed grasp prevents the common online interface from asserting an empty grasp, while treating such contact as decisive would recreate the short-touch false-positive problem.

All conclusions are limited to the fixed single-view MuJoCo proxy families. They are not physical-robot guarantees or evidence that the allowed observation stream is fundamentally insufficient.
