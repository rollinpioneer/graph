# Task binding

D0, T_A, and T_C have independent resolved definitions under `configs/tasks/resolved/` and independent scene reset configurations. They use project-owned procedural MuJoCo geometry: target, second object or interferer, openable container geometry, and buffer region. The Stage 0C contract registry freezes OPEN, PICK, PLACE, and PLACE_BUFFER schemas, grounded action IDs, proposition IDs, nominal ADD/DEL effects, and registry hash. It marks execution as `UNVERIFIED_FOR_STAGE_1A`.

D0 uses two movable objects, a container, buffer, and multi-step goal. T_A has the shared OPEN prerequisite for two PLACE actions. T_C has an interferer and a buffer relocation represented by PICK+PLACE_BUFFER. No MOVE alias was introduced.

The resolved task files separately declare `stage_0c_capture_ready: true` and `stage_1a_runtime_ready: false`. Controller, verifier, evaluator, safety, and timing fields remain Stage 1A requirements.
