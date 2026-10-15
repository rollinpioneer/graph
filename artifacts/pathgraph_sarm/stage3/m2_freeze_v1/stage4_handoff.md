# Stage 4 handoff

- Decision: GO_STAGE4 after corrected content-group aggregation.
- Runtime graph specs: `stage3/input_adapter_v1/runtime_graph_specs_v1.0.1/`.
- Diagnostic suite and checksum: `stage3/diagnostic_suite_v1/`.
- Runtime patch and episode index: `stage3/input_adapter_v1/`.
- Baseline predictions and corrected analysis: `stage3/rounds/stage3_3_baseline_runs/` and `stage3/rounds/stage3_4_misscoring_analysis/`.
- Confirmed structural signatures: alternative-order negative reward, recovery/cycle mis-scoring, and terminal time-fraction reward.
- Stage 4 outputs must include node belief, edge belief, within-node progress, and remaining cost.
- Start with history window 32; select checkpoints only on validation canonical controls, never Stage 3 test diagnostics.
- Preserve `scripted_oracle` provenance and content-group statistics; checkpoint paths remain in manifests and are omitted from the ZIP.
