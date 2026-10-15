# PathGraph-SARM Stage 8 Reproducibility Release

## Final research boundary

This release is limited to the frozen, manual, core PathGraph reward representation: multiple legal paths, failure/recovery semantics, remaining cost plus within-node progress, and recovery-debt accounting. It does not claim stable policy improvement, coverage scaling, unseen-order generalization, or automatic graph discovery as a main contribution.

## Layout

- `configs/`: frozen scope, input, reward, inference, and statistical locks.
- `code/`: minimal Stage 4 model, Stage 5 reward engine, and Stage 8 execution/analysis code—not the whole repository.
- `results/`: compact final CSVs, paper tables/figures, and manuscript evidence material.
- `scripts/`: checksum verification and reproduction entry points.
- `manifests/`: environment, code, and external-artifact records.

## Quick verification

```bash
bash scripts/verify_release.sh
```

## Cached-prediction validation

```bash
bash scripts/reproduce_from_cached_predictions.sh \
  --ensemble-test /path/ensemble_test_predictions.jsonl.gz \
  --ensemble-diagnostic /path/ensemble_stage3_diagnostic_predictions.jsonl.gz \
  --output /path/output --max-content-groups 10 --dry-run
```

The complete cached-prediction reconstruction additionally requires frozen oracle traces and seed prediction inputs listed in `manifests/external_artifact_manifest.tsv`.

## Checkpoint reproduction

```bash
bash scripts/reproduce_from_checkpoints.sh \
  --model-bundle /path/model_bundle_persistent.json \
  --supervision /path/supervision --diagnostic /path/diagnostic_suite \
  --output /path/output --gpus auto
```

Use the external manifest to locate the three SHA-verified checkpoints. The copied Stage 8 tools implement the frozen nine GPU inference jobs, ensemble merge, metrics, reward table, ablations, and grouped statistics. Query GPUs with `nvidia-smi`; the original execution used deterministic CUDA settings with one job per selected GPU.

## Figure reproduction

```bash
bash scripts/reproduce_figures.sh --results-dir /path/results --output /path/figures
```

All paper figures are CSV-driven. The auxiliary/negative R1 source tables required for appendix figures are externalized in the manifest.

## What is not included

No checkpoints, source supervision archives, raw per-step predictions, bootstrap distributions, or full repository snapshot is packaged. Their paths, size and available SHA256 values are listed in `manifests/external_artifact_manifest.tsv`.

## Result locations and limitations

`results/tables/` and `results/figures/` map directly to the paper artifacts; `docs/manuscript/` contains the claim-to-evidence map. Statistical inference uses `content_group_id`; controlled symbolic stress remains explicitly limited and is not real-robot generalization. Policy evidence is secondary/mixed and the manual graph remains the main method.
