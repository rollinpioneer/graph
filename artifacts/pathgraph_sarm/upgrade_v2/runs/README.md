# Omitted checkpoint binaries

The binary checkpoint named `best.pt` in each `u1_pilot/*/` and
`u1_formal/*/` run directory is intentionally local-only.  Each is a PyTorch
`ValueModel` state dictionary selected from the named run.  The original
filename is `best.pt`; its full local path, SHA-256 digest, model variant,
task, seed, and selection metric are recorded in that run's sibling
`job_result.json`.

This omission keeps the repository lightweight while preserving the experiment
directory structure and a verifiable record of every produced checkpoint.
