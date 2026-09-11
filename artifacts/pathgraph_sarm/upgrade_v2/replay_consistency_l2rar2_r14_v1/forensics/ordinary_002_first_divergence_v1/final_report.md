# R14 ordinary_002 first-divergence forensic review

## Scope

This review is static-only. No replay was rerun, no MuJoCo model/data/renderer was constructed by the review tools, and no new RGB was generated. `$EXEC` and `$CACHE` were treated as immutable inputs.

## Fixed execution identity

- runner commit: `ab342f8a5a04871b7d535c7b9190574cfce2cb94`
- protocol SHA256: `effc4137447803c3b684e5d60373b1faddf0d2d8de03428008a97ed62c9c134a`
- nonce: `e8027aed85c2233b173af332524b288339ddcb2866d0626d`
- execution: `ordinary_002_e8027aed85c2233b`
- original result: `STOP_AFTER_EXECUTION_1`, 7/8 main gates

## Findings

- Actions: exact sequence match.
- Low-level controls: exact match.
- Events: exact match.
- Callback/capture order: exact 37-row mapping.
- Renderer callback contract: 37 total, 29 control callbacks, 8 action-end callbacks; pass.
- Numeric health: qpos/qvel/mocap/eq_data/qacc_warmstart shapes and finite checks pass.
- Internal close-return/lift boundaries: pass.
- Attach geometry: cache and ordinary close-gripper world-relative geometry match exactly; ordinary local relative position equals `model.eq_data` relpose.
- First divergence: capture 12, `lift`, `control_tick`, action index 4, time `0.5000000000000002 s`; object/relative L2 `1.2249276643646983e-05 m` versus locked tolerance `1e-12 m`; gripper delta is zero and weld state matches.
- Pattern: `PERSISTENT_VARIABLE_OFFSET`; 25 post-first rows remain above tolerance.

## Provenance assessment

The Round-9 generation-lock commit is `f5e16da903bb7d64a26eac6163d41030a2fbcee7`. It does not contain `repair_collection.py` or `repaired_simulator.py`; later existence does not prove the historical uncommitted content. The R14 result records MuJoCo 3.4.0, NumPy 2.2.6, OpenCV 4.13.0, Python 3.10.19, but the historical collection runtime, `MUJOCO_GL`, platform, renderer backend, warmstart source, RNG source, and model XML hash are not proven.

## Decision

`OLD_CACHE_EXACT_RECONSTRUCTION_NOT_RECOVERABLE` (Route B). The evidence excludes sampling-point mapping and attach-relpose mismatch as the immediate explanation, but it cannot identify one repairable causal source with the required certainty. A new reproducible baseline would be a separate experiment and requires explicit authorization.

## Frozen gates

`R14 instrumented replay = 0`; `R16 calibration = 0`; `R16 development = 0`; new physical authorization = 0; new physical execution = 0; scientific status remains `L2RAR2_PARTIAL_KEEP_G1`; selected candidate remains `null`; confirmation remains `false`; L3 remains `false`.
