# Stage 1.1 summary

- Source: existing CUPID rollout manifests (`square` and `transport`), read-only.
- Episodes: 200 total (100 per task).
- Full-history source files: present for all 200 episodes; first timestep is 0.
- Known labels: 115 success and 85 failure from manifest `success`.
- Action chunks: present for all episodes (`action_shape` records horizon 16 and action dimension 10).
- Semantic subgoal, timestamp, SARM annotation, and explicit recovery fields are absent and remain unknown/false rather than inferred.
