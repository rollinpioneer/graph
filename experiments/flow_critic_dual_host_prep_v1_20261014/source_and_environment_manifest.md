# Sources, environments and differences

Card: FLOW-CRITIC-DUAL-HOST-PREP-V1. Authorized endpoint: asset preparation, two implementations, zero-training acceptance. Training, optimizer updates, online learning, Git push, PR and merge are not authorized.

Server root: `/home/xushijie3/work/flow_critic_dual_host_prep_v1_20261014`.
Archive branch: `codex/flow-critic-dual-host-prep-v1-20261014`, initially local to the server repository; subsequent user authorization on 2026-10-14 permits publication at https://github.com/rollinpioneer/graph/tree/codex/flow-critic-dual-host-prep-v1-20261014.

| Asset | Fixed commit | Acquisition |
|---|---|---|
| GFP | `1abbd5f0e287caf9bda70e7bc655d0a499ac8401` | Copy of previously verified official source archive |
| floq | `3d60e638b42bbef018c56cb1199ff37c3470520d` | Copy of existing pinned checkout, excluding Git metadata |
| DFQL | `8b6b15803cc1e5605ad1524553a2c95e103b0bad` | Official exact-commit codeload archive; SHA resolved via GitHub connector |
| FQL reference | `e8cd16eb490332924dfa2492097219f181765933` | Existing verified source and assets; no new FQL implementation |

DFQL `agents/dfql.py` Git blob is `e266620dd015e24a8f322fe3eece386d06588b41`, matching the GitHub connector response. Source files, original dependency files, READMEs, license files if supplied, agents, networks and main/evaluation entry files are retained under `upstream_snapshots`. File SHA-256 identities are in `asset_manifest.json`. Snapshot files are read-only; host changes appear only as new adapter/package files. The shared namespaced network file is byte-identical to pinned floq `utils/networks.py`.

| Environment | Python | JAX / jaxlib | Flax | Optax | OGBench | Gymnasium | NumPy |
|---|---|---|---|---|---|---|---|
| New GFP host | 3.12.14 | 0.6.2 / 0.6.2 | 0.10.7 | 0.2.5 | 1.1.0 | 1.1.1 | 1.26.4 |
| New DFQL host | 3.10.19 | 0.6.2 / 0.6.2 | 0.10.7 | 0.2.5 | 1.1.0 | 0.29.1 | 1.26.4 |
| Original floq asset reader | 3.8.19 | 0.4.13 / 0.4.12 CUDA11 | 0.7.2 | 0.1.3 | 1.1.2 | 1.0.0 | 1.22.4 |

New environments are private filesystem copies of the validated GFP and FQL environments; script paths are relocated. Original environments are not installed into or modified. All distribution versions/locations are recorded in `{gfp,dfql}_environment.json` and `{gfp,dfql}_requirements.lock`. These environments have no pip module: attempted pip inspection returned “No module named pip”; metadata-based locks are the actual inventory. No package upgrade was used. DFQL's required state-based versions and minimum JAX/Flax requirements are satisfied. Optional D4RL execution is outside this card.

Tests use NVIDIA A100-SXM4-40GB GPUs 3 and 7 for the two host suites, float32 parameters, JAX x64 disabled and unchanged default matrix precision. CPU core affinity is 52–55 and 56–59 respectively, BLAS/OpenMP threads 1, preallocation disabled. Original asset restore and value mapping run on CPU with their original environments. CPU-only CUDA plugin discovery warnings and XLA autotuning hint warnings are retained in raw receipts; finite successful calculations determine acceptance.

## Recorded differences and resolutions

1. The pinned floq configuration comments describe Fourier-64 time embedding, but `CriticVectorField` and `CriticResVectorField` implement only `cos(times)` with one channel and do not use `time_embed_dim`. The transplant preserves that code and retains the inert field. It does not add Fourier features.
2. GFP HumanoidMaze-medium `eta_temperature=0.001` is confirmed by the official OmegaConf/CSV matcher and agrees with the card. An initial terminal-wrap reading as 0.0001 was incorrect and is explicitly superseded. No η change or tuning was made.
3. An early preparation helper used generic YAML parsing, turning scientific-notation defaults into strings. GFP Z0 stopped before parameter updates. The failure JSON/log are preserved. `resolve_gfp_configs.py` now uses OmegaConf and the actual pinned CSV matcher, with no command-line agent overrides. These final configurations are the accepted ones.
4. Actual dataset rewards are Humanoid/AntMaze `[-1,0]`, Cube-double `[-2,0]`, Puzzle `[-9,0]`. Encoding supports and initial-noise intervals derive from these processed ranges and each host discount; AntMaze bounds are not copied to other tasks.
5. Pinned GFP generates VaBC target actions at current observations when `target_use_next=False` and evaluates their value at next observations. That behavior is preserved and explicitly tested.
6. Source floq/GFP EMA uses pre-optimizer online parameters, not just-updated parameters. This one-update lag is retained in the pure target specification. It is exercised only with synthetic arrays.
7. Pinned DFQL allocates an unused one-step Actor even for `actor_type=DFQL`. Its original scalar path is untouched; the combined path includes only the used multi-step Actor. The source's clipped-endpoint gradient calculation and tanh velocity output remain unchanged.
8. `critic_type`, execution guard fields, namespaced core, and GFP EMA scalar-readout connection are new card designs. They are not official host flags or original floq features. Source floq Actor alpha is not inherited.

The two named prior reports were not found in the approved local/remote workspace searches and remain `MISSING_USER_ACCESS`. The card and fixed sources provide the implemented contract; no conclusions are attributed to unavailable reports. No DFQL trained weights or new-task Baseline reproduction are claimed.

The branch was created from the repository's existing `origin/main` without altering the primary checkout or its untracked artifacts. The preparation phase prohibited pushing. The subsequent user instruction “推送” authorizes ordinary publication of this registered branch, while training, PR and merge remain unauthorized.
