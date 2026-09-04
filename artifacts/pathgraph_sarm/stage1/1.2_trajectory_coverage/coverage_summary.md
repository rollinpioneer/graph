# Coverage summary

Structural flags are conservative: supplied rollout manifests expose outcome and complete-history boundaries but no semantic subgoal/recovery events.

- `square`: 100 usable; success=71, failure=29, distinct_success_paths=1, recovery=0; action-chunk coverage=1.000.
- `transport`: 100 usable; success=44, failure=56, distinct_success_paths=1, recovery=0; action-chunk coverage=1.000.

Key gap: alternative-order and recovery evidence must be collected or manually annotated before PathGraph training.
