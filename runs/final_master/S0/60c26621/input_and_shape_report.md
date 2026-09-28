# Input and shape report

The production fixture probe used `tests/fixtures/*`, explicitly labeled synthetic and not paper-performance eligible.

- Graph node representation: 128 dimensions.
- RGCN: four layers, 4 relation bases, relation vocabulary includes forward and reverse relations.
- GoalReadout: one nonlinear global row plus one 128-dimensional row per signed goal.
- Candidate projection: candidate feature vector to 128 dimensions.
- Full/A_STAT context: recurrent 128 + candidate 128 + K global 128 = 384.
- Prior readout: 512-dimensional context (`context` + `u_K`) and 128-dimensional row input; bounded scalar residual with B=0.5.
- A_CAT: four encoded views concatenated to 512 then projected to 128; its capacity/null behavior is reported separately.
- B1-K+E: contract/effect tokens are each projected to 128 and pooled by context attention before a 256-to-128 fusion with the original B1-K candidate representation.
- Value head input: 512; Q head input: 640.

Full/A_STAT parameter shape hash and active parameter counts match in `parameter_capacity_report.json`.
