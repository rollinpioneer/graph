# Budget, checkpoint and counter accounting

- S0 itself consumed 0 RL attempts, 0 elastic attempts, 0 provider requests and no formal training slot.
- The selected S0 tests perform production forward/backward and one synthetic PPO update for logging-path acceptance; this is not a research RL run and is labeled `synthetic_unit_fixture`.
- No checkpoint, historical raw run, envelope, or shared editable import was overwritten.
- `PPO.update` logs actual optimizer-step identity; persistence remains hash-checked and refuses overwrite of an existing checkpoint.
- Historical pause/resume counters and old Recovery accounting were not rewritten. Unknown counters remain `NOT_MEASURED`; filenames are not treated as checkpoint truth.
- B_PLAN development used exactly 10 authorized T_B episodes; S0-TB RL training, S1, S2, S3, all RL training, elastic attempts and method upgrades remain `authorized=false / PLANNED`.
