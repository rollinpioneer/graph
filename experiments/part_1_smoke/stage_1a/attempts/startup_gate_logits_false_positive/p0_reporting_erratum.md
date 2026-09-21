# P0 reporting erratum

Historical Stage 1A-P0 dry-run reports recorded `transition.action`.
The Transition dataclass has no `action` field. The executed skill identifier is `selected_candidate_id`.
Values of `action: null` in experiments/part_1_smoke/stage_1a_p0/b2_full_dry_run.json are a report-field bug,
not evidence that the collector executed an empty action.
The original P0 files are not modified. Corrected records are in this directory.

P0 B2 prior_edge_count=1 described the source cache, not B2 effective prior. See gate B.
