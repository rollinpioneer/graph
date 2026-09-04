# Stage 3.3 summary

- Required oracle baselines: completed.
- Required learned jobs: 9 completed with CUDA-visible devices 3-6.
- Training budget: fixed 80 epochs; checkpoints selected from canonical training data only.
- Checkpoints: retained by manifest path, omitted from ZIP.
- Predictions: per-episode/per-step JSONL under `predictions/`.
- Provenance: scripted_oracle diagnostics; no policy-generalization claim.
