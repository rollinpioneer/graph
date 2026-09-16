"""Resume D7-D9 from a fully committed D6 corpus without repeating input scans."""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from .core import sha256_file, write_json
from .runner import (
    METHODS, SEEDS, aliasing, claims_and_report, counterfactual, graph_decision,
    iter_jsonl, load_runtime, manifest, mechanism_decisions,
    select_counterfactual_states,
)


def csv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resume(repo, p2b_root, external_root, artifact):
    repo, p2b_root, external_root, artifact = map(Path, (repo, p2b_root, external_root, artifact))
    runtime = load_runtime(repo)
    db_path = external_root / "potential_corpus/state_index.sqlite"
    corpus_path = external_root / "potential_corpus/common_state_potential_corpus.jsonl"
    db = sqlite3.connect(db_path)
    groups = aliasing(db, artifact)
    selected = select_counterfactual_states(db, runtime[0]["conditions"])
    selected_file = external_root / "counterfactual_actions/selected_states.json"
    write_json(selected_file, {"count": len(selected), "state_hashes": selected})
    episode_specs = {}
    for method in METHODS:
        for seed in SEEDS:
            episode_specs[(method, seed)] = {
                row["episode_id"]: {k: row[k] for k in ("family", "profile", "condition", "repeat", "seed", "episode_id")}
                for row in iter_jsonl(p2b_root / "test" / method / f"seed_{seed}" / "episodes.jsonl")
                if row["repeat"] == 0
            }
    counter = counterfactual(db, artifact, runtime, selected, episode_specs)
    validation = csv_rows(artifact / "validation_curve_classification.csv")
    training = csv_rows(artifact / "training_episode_summary.csv")
    failures = csv_rows(artifact / "failure_episode_taxonomy.csv")
    for row in training:
        row["policy_seed"] = int(row["policy_seed"]); row["successes"] = int(row["successes"])
    for row in validation: row["policy_seed"] = int(row["policy_seed"])
    for row in failures: row["policy_seed"] = int(row["policy_seed"])
    sign_rows = csv_rows(artifact / "shaping_sign_matrix.csv")
    condition_rows = csv_rows(artifact / "condition_potential_disagreement.csv")
    for row in sign_rows: row["rate"] = float(row["rate"])
    for row in condition_rows: row["graph_unique_rate"] = float(row["graph_unique_rate"])
    mechanisms = mechanism_decisions(training, validation, failures)
    graph_result = graph_decision(sign_rows, condition_rows, counter, groups)
    full_training = external_root / "seed_forensics/training_episode_summary.full.csv"
    external = [
        {"kind": "corpus", "absolute_path": str(corpus_path), "bytes": corpus_path.stat().st_size, "sha256": sha256_file(corpus_path), "role": "complete raw-transition causal replay corpus"},
        {"kind": "state_index", "absolute_path": str(db_path), "bytes": db_path.stat().st_size, "sha256": sha256_file(db_path), "role": "unique-state and replay locator index"},
        {"kind": "training_episode_table", "absolute_path": str(full_training), "bytes": full_training.stat().st_size, "sha256": sha256_file(full_training), "role": "full per-episode training forensics"},
        {"kind": "counterfactual_registry", "absolute_path": str(selected_file), "bytes": selected_file.stat().st_size, "sha256": sha256_file(selected_file), "role": "deterministic counterfactual state registry"},
    ]
    import json
    pre = json.loads((artifact / "preflight.json").read_text(encoding="utf-8"))
    claims_and_report(repo, p2b_root, external_root, artifact, pre, mechanisms, graph_result, counter, external)
    result = manifest(artifact)
    db.close()
    return {"status": "COMPLETE", "manifest_files": result["file_count"], "counterfactual_states": counter["states"], "graph_result": graph_result["status"], "zero_seed_mechanisms": mechanisms}
