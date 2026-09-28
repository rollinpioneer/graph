# R3 duration audit

The run did not record a complete task-duration interval from episode start to independent first success. `success_seconds` in the dev JSON is retained as the measured first-success clock value where present, but no full-task duration is inferred. Terminal-skill duration is not substituted for full-task duration; unreconstructable fields are `NA`.
