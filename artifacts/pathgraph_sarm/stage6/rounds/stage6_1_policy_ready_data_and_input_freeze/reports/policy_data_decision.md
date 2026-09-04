# Stage 6.1 policy-data gate

Decision: `POLICY_DATA_REAL_ACTION_READY`

All BC actions are `action_applied` values recorded from the actual `env.step(action)` call; Stage 2 synthetic zero-action traces were not read.
