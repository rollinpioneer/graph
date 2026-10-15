# Experimental Setup

The locked benchmark uses the transport dual-order and recovery tasks with frozen train/validation/test partitions. The manual graph supplies nodes and typed legal, failure, and recovery edges. We independently ran 3 frozen reward-model checkpoints with ensemble seeds [20260906, 20260907, 20260908] in evaluation/inference mode, with history length 32.

`content_group_id` is the statistics unit. Core reward comparisons include linear and sequential baselines and the predeclared structural ablations. We use 10000 stratified, paired content-group bootstrap resamples. Controlled symbolic stress tests demonstrate fixed graph semantics and are not described as real-robot generalization. Policy evidence remains secondary/mixed.
