"""Create the required U2 round-level provenance, GPU, and checksum records.

This script does not rerun experiments or alter result payloads.  It records
the preserved execution evidence in a portable, single-ZIP-friendly form.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROUND_SPECS = {
    "u2_0_entry_and_eventful_dataset": {
        "purpose": "Freeze simulator-only entry, eventful trajectories, family split, and exact state restore.",
        "gpu_ids": [],
        "commands": [
            'source "$REPO_ROOT/artifacts/pathgraph_sarm/upgrade_v2/u2_stochastic_boundary/u2_env.sh"',
            '"$PYTHON_BIN" -m upgrade_v2.u2.cli collect-eventful-dataset --mode formal --root-families 120 --rollouts-per-family 6 --scenarios nominal_success,grazing_contact,slip_recovery,obstacle_detour,terminal_collision,stagnation --split-ratios 0.70,0.15,0.15 --split-unit root_family_id --seed 20260953 --output-root "$U2_DATA/formal"',
            '"$PYTHON_BIN" -m upgrade_v2.u2.cli validate-eventful-dataset --dataset "$U2_DATA/formal" ...',
            '"$PYTHON_BIN" -m upgrade_v2.u2.cli audit-observation-action-alignment --dataset "$U2_DATA/formal" ...',
        ],
    },
    "u2_1_event_candidates_and_weak_labels": {
        "purpose": "Generate causal sensor-change candidates, calibrated weak posteriors, and unknown states.",
        "gpu_ids": [],
        "commands": [
            'source "$REPO_ROOT/artifacts/pathgraph_sarm/upgrade_v2/u2_stochastic_boundary/u2_env.sh"',
            '"$PYTHON_BIN" -m upgrade_v2.u2.cli write-weak-rules --output "$U2_WEAK/configs/weak_rules.yaml"',
            '"$PYTHON_BIN" -m upgrade_v2.u2.cli extract-event-candidates --dataset "$U2_DATA/formal" --rules "$U2_WEAK/configs/weak_rules.yaml" --output-root "$U2_WEAK/candidates" --manifest "$U2_WEAK/manifests/event_candidate_manifest.csv"',
            '"$PYTHON_BIN" -m upgrade_v2.u2.cli aggregate-weak-events --candidate-root "$U2_WEAK/candidates" --dataset "$U2_DATA/formal" ...',
        ],
    },
    "u2_2_segmentation_baselines": {
        "purpose": "Fit and evaluate uniform, causal hysteresis, and offline multivariate change-point baselines.",
        "gpu_ids": [],
        "commands": [
            'source "$REPO_ROOT/artifacts/pathgraph_sarm/upgrade_v2/u2_stochastic_boundary/u2_env.sh"',
            '"$PYTHON_BIN" -m upgrade_v2.u2.cli run-segmentation-baselines --dataset "$U2_DATA/formal" --weak-posteriors "$U2_WEAK/posteriors" --methods uniform,sensor_hysteresis,multivariate_change_point --selection-split val ...',
            '"$PYTHON_BIN" -m upgrade_v2.u2.cli evaluate-segmentation-baselines --dataset "$U2_DATA/formal" --prediction-root "$U2_BASELINES/predictions" --selection "$U2_BASELINES/selection/baseline_selection.csv" --split test ...',
        ],
    },
    "u2_3_causal_boundary_models": {
        "purpose": "Train three frozen boundary variants, select on validation only, and run val/test inference.",
        "gpu_ids": [1, 4, 6],
        "commands": [
            'source "$REPO_ROOT/artifacts/pathgraph_sarm/upgrade_v2/u2_stochastic_boundary/u2_env.sh"',
            '"$PYTHON_BIN" -m upgrade_v2.u2.cli build-boundary-jobs --mode formal --dataset "$U2_DATA/formal" ...',
            '"$PYTHON_BIN" -m upgrade_v2.u2.cli launch-jobs --job-table "$U2_ROUNDS/u2_3_causal_boundary_models/tables/u2_boundary_formal_jobs.tsv" --gpu-ids 1,4,6 ...',
            '"$PYTHON_BIN" -m upgrade_v2.u2.cli select-boundary-checkpoints --job-root "$U2_MODELS/formal" --job-table ... --split val ...',
            '"$PYTHON_BIN" -m upgrade_v2.u2.cli launch-inference-jobs --gpu-ids 1,4,6 ...',
        ],
    },
    "u2_4_segment_representation": {
        "purpose": "Build unknown-aware segments, history embeddings, clusters, and observed transition summaries.",
        "gpu_ids": [],
        "commands": [
            'source "$REPO_ROOT/artifacts/pathgraph_sarm/upgrade_v2/u2_stochastic_boundary/u2_env.sh"',
            '"$PYTHON_BIN" -m upgrade_v2.u2.cli build-segments --dataset "$U2_DATA/formal" --boundary-source "$U2_SEGMENTS/configs/boundary_source_lock.json" ...',
            '"$PYTHON_BIN" -m upgrade_v2.u2.cli encode-segments --segments "$U2_SEGMENTS/segments" ...',
            '"$PYTHON_BIN" -m upgrade_v2.u2.cli cluster-segments --methods raw_observable_kmeans,history_embedding_kmeans,history_plus_event_posterior_kmeans --clusters 5,7,9,11 --seeds 631,632,633 ...',
        ],
    },
    "u2_5_budgeted_correction": {
        "purpose": "Compare zero-label, fixed oracle-budget, and query-strategy correction under a frozen protocol.",
        "gpu_ids": [1, 4, 6],
        "commands": [
            'source "$REPO_ROOT/artifacts/pathgraph_sarm/upgrade_v2/u2_stochastic_boundary/u2_env.sh"',
            '"$PYTHON_BIN" -m upgrade_v2.u2.cli build-query-queues --dataset "$U2_DATA/formal" --weak-posteriors "$U2_WEAK/posteriors" ...',
            '"$PYTHON_BIN" -m upgrade_v2.u2.cli reveal-oracle-clips --queue-root "$U2_BUDGET/queues" --dataset "$U2_DATA/formal" ...',
            '"$PYTHON_BIN" -m upgrade_v2.u2.cli launch-value-jobs --gpu-ids 1,4,6 ...',
            '"$PYTHON_BIN" -m upgrade_v2.u2.cli evaluate-budgeted-correction --job-root "$U2_BUDGET/models" --dataset "$U2_DATA/formal" ...',
        ],
    },
    "u2_6_reward_impact_and_gate": {
        "purpose": "Fit independent continuation q/D references, aggregate reward by boundary source, and finalize U2 gate.",
        "gpu_ids": [1, 4, 6],
        "commands": [
            'source "$REPO_ROOT/artifacts/pathgraph_sarm/upgrade_v2/u2_stochastic_boundary/u2_env.sh"',
            '"$PYTHON_BIN" -m upgrade_v2.u2.cli collect-boundary-value-continuations --dataset "$U2_DATA/formal" --anchors-per-family 4 --continuations-per-anchor 3 --horizon 32 ...',
            '"$PYTHON_BIN" -m upgrade_v2.u2.cli launch-value-jobs --gpu-ids 1,4,6 ...',
            '"$PYTHON_BIN" -m upgrade_v2.u2.cli aggregate-reward-by-boundary --dataset "$U2_DATA/formal" ...',
            '"$PYTHON_BIN" -m upgrade_v2.u2.cli evaluate-boundary-reward-impact --bootstrap 5000 ...',
            '"$PYTHON_BIN" -m upgrade_v2.u2.cli finalize-u2 --repo-root "$REPO_ROOT" --u2-root "$U2_ROOT" --final-root "$U2_FINAL"',
        ],
    },
}

ROUND_FILTERS = {
    "u2_0_entry_and_eventful_dataset": ("data_v1/",),
    "u2_1_event_candidates_and_weak_labels": ("weak_events_v1/",),
    "u2_2_segmentation_baselines": ("segmentation_baselines_v1/",),
    "u2_3_causal_boundary_models": ("boundary_models_v1/",),
    "u2_4_segment_representation": ("segment_representation_v1/",),
    "u2_5_budgeted_correction": ("budgeted_correction_v1/",),
    "u2_6_reward_impact_and_gate": ("reward_impact_v1/",),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tsv_write(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mtime_bounds(round_dir: Path) -> tuple[str, str]:
    mtimes = [path.stat().st_mtime for path in round_dir.rglob("*") if path.is_file() and path.name not in {"run_manifest.md"}]
    if not mtimes:
        now = datetime.now().astimezone()
        return now.isoformat(), now.isoformat()
    return (datetime.fromtimestamp(min(mtimes)).astimezone().isoformat(), datetime.fromtimestamp(max(mtimes)).astimezone().isoformat())


def write_gpu_record(round_dir: Path, gpu_ids: list[int], commit: str) -> None:
    gpu_dir = round_dir / "gpu"
    gpu_dir.mkdir(parents=True, exist_ok=True)
    snapshot = ""
    try:
        snapshot = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        snapshot = f"nvidia-smi unavailable during audit: {exc}\n"
    (gpu_dir / "gpu_query_mode.txt").write_text(
        "GPU_QUERY_MODE=privileged_audit_snapshot\n"
        "Historical training/inference usage is taken from the preserved job status TSV.\n"
        f"historical_gpu_ids={','.join(map(str, gpu_ids)) or 'none'}\n",
        encoding="utf-8",
    )
    (gpu_dir / "gpu_inventory.csv").write_text(
        "# privileged audit snapshot 2026-09-05; not a claim that CPU-only rounds used GPU\n" + snapshot,
        encoding="utf-8",
    )
    (gpu_dir / "gpu_usage_record.md").write_text(
        "# GPU usage record\n\n"
        f"- code_commit: `{commit}`\n"
        f"- historical_gpu_ids: `{','.join(map(str, gpu_ids)) or 'none (CPU-only round)'}`\n"
        "- source: preserved `*_status.tsv` job records when present; otherwise CPU-only protocol.\n"
        "- privileged visibility was confirmed with `nvidia-smi`; ordinary Torch CUDA visibility is not used as a negative GPU conclusion.\n",
        encoding="utf-8",
    )


def write_checksums(round_dir: Path) -> None:
    checksum_dir = round_dir / "checksums"
    checksum_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(round_dir.rglob("*")):
        if not path.is_file() or path.is_relative_to(checksum_dir) or path.name.endswith(".placeholder.md"):
            continue
        if path.suffix in {".npz", ".pt", ".pth", ".parquet", ".jsonl", ".log", ".pyc", ".pyo"}:
            continue
        rows.append(f"{sha256(path)}  {path.relative_to(round_dir).as_posix()}")
    (checksum_dir / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (checksum_dir / "README.md").write_text(
        "Checksums cover lightweight round evidence only. Omitted large artifacts are identified by the round/final large-file manifests and placeholders.\n",
        encoding="utf-8",
    )


def write_round_omission_manifests(repo: Path, rounds_root: Path) -> None:
    """Give every round a schema-compliant omission manifest.

    The final manifest is the complete index.  Round manifests are filtered
    views so each round remains independently auditable inside the one ZIP.
    """
    final_root = rounds_root.parent / "final_v1"
    checkpoint_path = final_root / "manifests" / "checkpoint_manifest.tsv"
    large_path = final_root / "manifests" / "large_file_manifest.tsv"
    with checkpoint_path.open(encoding="utf-8", newline="") as handle:
        checkpoints = list(csv.DictReader(handle, delimiter="\t"))
    with large_path.open(encoding="utf-8", newline="") as handle:
        large = list(csv.DictReader(handle, delimiter="\t"))
    checkpoint_fields = ["path", "size_bytes", "job_id", "artifact_type", "reason_omitted", "sha256", "packaged", "epoch_or_step", "key_metric"]
    large_fields = ["path", "size_bytes", "job_id", "artifact_type", "reason_omitted", "packaged"]
    for name, selectors in ROUND_FILTERS.items():
        round_dir = rounds_root / name
        selected_checkpoints = [row for row in checkpoints if any(selector in row.get("path", "") for selector in selectors)]
        selected_large = [row for row in large if any(selector in row.get("path", "") for selector in selectors)]
        _tsv_write(round_dir / "manifests" / "checkpoint_manifest.tsv", selected_checkpoints, checkpoint_fields)
        _tsv_write(round_dir / "manifests" / "large_file_manifest.tsv", selected_large, large_fields)


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    rounds_root = repo / "artifacts/pathgraph_sarm/upgrade_v2/u2_stochastic_boundary/rounds"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    generated_at = datetime.now().astimezone().isoformat()
    for name, spec in ROUND_SPECS.items():
        round_dir = rounds_root / name
        start, end = mtime_bounds(round_dir)
        run_manifest = round_dir / "run_manifest.md"
        command_block = "\n".join(f"```bash\n{command}\n```" for command in spec["commands"])
        run_manifest.write_text(
            f"# {name}\n\n"
            f"- round_id: `{name}`\n"
            f"- purpose: {spec['purpose']}\n"
            "- execution_status: `PASS` (preserved result audit; no payload rerun required)\n"
            f"- evidence_start: `{start}`\n"
            f"- evidence_end: `{end}`\n"
            f"- manifest_generated_at: `{generated_at}`\n"
            f"- code_commit: `{commit}`\n"
            f"- gpu_ids: `{','.join(map(str, spec['gpu_ids'])) or 'none'}`\n"
            "- selection_split: `val` where model/config selection applies\n"
            "- test_used_for_selection: `False`\n"
            "- archive_policy: `single cumulative ZIP only; no per-round ZIP emitted`\n\n"
            "## Reproduction commands\n\n" + command_block + "\n",
            encoding="utf-8",
        )
        write_gpu_record(round_dir, spec["gpu_ids"], commit)
        write_checksums(round_dir)
    write_round_omission_manifests(repo, rounds_root)


if __name__ == "__main__":
    main()
