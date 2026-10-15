# Stage 3.1 runtime input patch

M1 was verified unchanged using its checksum manifest. Runtime GraphSpec v1.0.1 resolves duplicate IDs, adds accepted GT-only transitions, and normalizes failure/recovery labels. Runtime GT v1.0.1 maps the erroneous initial recovery edge to `start_to_grasped` and applies GraphSpec-compatible labels.

- episode observations before grouping: 108
- content groups after grouping: 56
- duplicate scripted-oracle observations excluded from independent evidence: 52
- GT interval corrections: 68
- statistical unit for Stage 3: `content_group_id`
- validation: PASS
