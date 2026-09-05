# PathGraph-SARM Research Snapshot

This repository is the lightweight, structure-preserving GitHub snapshot of the PathGraph-SARM research workspace. The source workspace was `/home/xushijie/CUPID` (resolved locally to `/home/__compress_data/xushijie/CUPID`) and the original code base is kept under `repo/`.

## Contents

- `repo/`: CUPID/diffusion-policy source code, configuration, tests, and notebooks.
- `artifacts/pathgraph_sarm/`: Stage 1 through Stage 8 research artifacts, reports, manifests, metrics, and round summaries.
- `tools/`, `configs/`: research-stage scripts and configuration files.
- `docs/agent_instructions/`: the complete PathGraph-SARM stage instructions and fixed execution specification.
- `docs/project_notes/`: project-level research notes and runbooks.
- `downloads/`: lightweight checksums and delivery metadata.
- `LARGE_FILES_OMITTED.tsv`: machine-readable index of every source file omitted from this snapshot.

The simulator-scoped U2 stochastic-boundary prototype is under
`artifacts/pathgraph_sarm/upgrade_v2/u2_stochastic_boundary/`. Its seven round
directories contain `run_manifest.md`, GPU visibility records, lightweight
checksums, reports, and omission manifests. The U2 decision is
`GO_U3_WITH_BOUNDARY_FALLBACK`; claims remain restricted to the explicit
stochastic simulator family.

## Omitted files and placeholders

Files that are datasets, raw episodes, videos, checkpoints, model weights, memory maps, environment binaries, run caches, or delivery archives are not uploaded. For each omitted source file, the original relative path contains a sibling placeholder named `original-name.placeholder.md`. Each placeholder records the original filename, path, type, intended use, byte size, omission reason, and restore location. Empty source directories contain `DIRECTORY_PLACEHOLDER.md` when needed to keep the layout visible.

The upload threshold is 10 MiB per file. This prevents accidental large-file commits while retaining lightweight tables, logs, figures, code, and manifests. Do not replace placeholders with binaries in this repository; restore heavy artifacts from external storage when reproducing an experiment.

## Research status

The final Stage 8 status is `RESEARCH_COMPLETE_CORE_REWARD_ONLY`. The final lightweight release archive was created outside this GitHub snapshot with SHA256:

`65b24b5e4a089d86922e24371889fbf3af7926560382415d3137079a18b6ecbb`

Per the user's delivery policy, the current workspace uses one cumulative ZIP
only (U0/U1 plus U2 lightweight evidence):
`downloads/upgrade_v2/U0_U1_complete.zip`.

Checkpoint and raw-prediction artifacts remain external by design. The stage documents and manifests in this repository describe the exact source paths and the evidence boundary for the final claims.

## Reproduction

Read the stage instructions in `docs/agent_instructions/` in order. Install the dependencies described by `repo/conda_environment.yaml`, restore only the required external artifacts at the paths recorded in `LARGE_FILES_OMITTED.tsv`, and use the stage manifests to verify checksums and run provenance. GPU experiments must follow the privileged GPU-query, multi-GPU, and per-round ZIP rules in `实验Agent固定执行规范_GPU并行与每轮ZIP交付_V1.2.md`.
