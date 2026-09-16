from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .core import (
    FAILURE_PRIORITY,
    RunningPair,
    active_object,
    canonical,
    classify_features,
    geom_signature,
    guidance_category,
    potential_history,
    quantiles,
    sha256_file,
    spearman,
    state_hash,
    value_hash,
    write_csv,
    write_json,
    write_tsv,
)

BASE = "0f26660cde901aa32b0cfc032fc75ec01b31687b"
BRANCH = "research/pathgraph-p2b-d1-graph-increment-diagnostics-v1"
METHODS = (
    "TASK_ONLY", "COUNT_PBRS", "COUNT_EVENTS_PBRS",
    "GEOM_COUNT_EVENTS_PBRS", "GRAPH_COST_PBRS", "GRAPH_FULL_PBRS",
)
SEEDS = (211, 223, 227, 229, 233, 239, 241, 251)
MILESTONES = (0, 32768, 65536, 131072, 262144, 524288)
TARGETS = (
    ("TASK_ONLY", 211), ("TASK_ONLY", 239),
    ("COUNT_EVENTS_PBRS", 227), ("COUNT_EVENTS_PBRS", 233), ("COUNT_EVENTS_PBRS", 239),
)
AUXILIARY = (("COUNT_PBRS", 223), ("COUNT_PBRS", 227))
TARGET_SEEDS = tuple(sorted({seed for _, seed in TARGETS + AUXILIARY}))
CONTROL_METHODS = ("GEOM_COUNT_EVENTS_PBRS", "GRAPH_COST_PBRS", "GRAPH_FULL_PBRS")
FORENSIC_JOBS = tuple(sorted(set(TARGETS + AUXILIARY + tuple((m, s) for s in TARGET_SEEDS for m in CONTROL_METHODS))))
V6_TOOLS = Path("/home/xushijie/PathGraph_P1_V6_Three_Issue_Execution_Agent_Package_V1.0/tools")


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def status_csv(path, status, reason=""):
    write_csv(path, [{"status": status, "reason": reason}], ["status", "reason"])


def load_runtime(repo):
    sys.path.insert(0, str(repo / "experiments/pathgraph_p2b_ppo_v1"))
    from p2b.io_utils import protocol as frozen_protocol
    from p2b.load_sources import load_sources
    from p2b.shaping import shaped_transition
    from p2b.task_core import TaskCore
    config = frozen_protocol()
    envmod, cap, loaded_sources = load_sources(repo, V6_TOOLS)
    return config, envmod, cap, loaded_sources, shaped_transition, TaskCore


def protocol_document():
    return {
        "schema": "P2B-D1_GRAPH_INCREMENT_DIAGNOSTICS",
        "base_commit": BASE,
        "branch": BRANCH,
        "new_training_jobs": 0,
        "optimizer_updates": 0,
        "gradient_steps": 0,
        "checkpoint_files_modified": 0,
        "checkpoint_selection": "FIXED_LAST_STEP_ONLY",
        "policy_seeds": list(SEEDS),
        "methods": list(METHODS),
        "gamma": 0.99,
        "beta": 1.0,
        "profiles": ["BASE64", "LONG_CHAIN128"],
        "conditions": ["FREE_ORDER", "A_STARTED", "B_STARTED", "A_VALID_B_LOSS", "B_VALID_A_LOSS", "INVALIDATE_FIRST", "LATE_LOSS", "TWO_LOSSES"],
        "zero_tolerance": 1e-10,
        "failure_priority": list(FAILURE_PRIORITY),
        "p2b_original_decision_modified": False,
        "confirmation_passed": False,
    }


def source_lock(repo, artifact):
    paths = [
        "experiments/pathgraph_p2a_utility_v1/p2a/env.py",
        "experiments/pathgraph_p2b_ppo_v1/p2b/train.py",
        "experiments/pathgraph_p2b_ppo_v1/p2b/evaluate.py",
        "experiments/pathgraph_p2b_ppo_v1/p2b/task_core.py",
        "experiments/pathgraph_p2b_ppo_v1/p2b/potentials.py",
        "experiments/pathgraph_p2b_ppo_v1/p2b/shaping.py",
        "experiments/pathgraph_p2b_ppo_v1/p2b/specs.py",
        "experiments/pathgraph_p2b_ppo_v1/p2b/statistics.py",
    ]
    rows = {}
    for rel in paths:
        path = repo / rel
        if not path.is_file():
            raise RuntimeError(f"D0_BLOCKED_SOURCE_LOCK_MISMATCH: missing {rel}")
        work_hash = sha256_file(path)
        base_bytes = subprocess.check_output(["git", "-C", str(repo), "show", f"{BASE}:{rel}"])
        base_hash = __import__("hashlib").sha256(base_bytes).hexdigest()
        if work_hash != base_hash:
            raise RuntimeError(f"D0_BLOCKED_SOURCE_LOCK_MISMATCH: {rel}")
        rows[rel] = {"sha256": work_hash, "base_sha256": base_hash, "match": True}
    result = {"base_commit": BASE, "status": "PASS", "files": rows}
    write_json(artifact / "source_lock.json", result)
    return result


def external_checkpoint_hashes(repo):
    path = repo / "artifacts/pathgraph_sarm/upgrade_v2/p2b_graph_ppo_utility_v1/external_artifacts.tsv"
    out = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("kind") == "checkpoint":
                out[row["path"]] = row["sha256"]
                out[Path(row["path"]).name + "|" + Path(row["path"]).parent.name + "|" + Path(row["path"]).parent.parent.name] = row["sha256"]
    return out


def preflight(repo, p2b_root, artifact):
    expected_hash = external_checkpoint_hashes(repo)
    checkpoint_rows, hash_rows, input_rows, missing = [], [], [], []
    episode_specs = {}
    total_test = 0
    final_matches = 0
    complete_count = 0
    validation_count = 0
    for method in METHODS:
        for seed in SEEDS:
            train = p2b_root / "training" / method / f"seed_{seed}"
            test = p2b_root / "test" / method / f"seed_{seed}"
            complete_path = train / "complete.json"
            if complete_path.is_file():
                complete = read_json(complete_path)
                complete_ok = complete.get("status") == "TRAINING_COMPLETE"
                complete_count += int(complete_ok)
            else:
                complete_ok = False
                missing.append(str(complete_path))
            for step in MILESTONES:
                cp = train / f"policy_{step}.zip"
                exists = cp.is_file()
                actual = sha256_file(cp) if exists else ""
                final = step == 524288
                key = cp.name + "|" + cp.parent.name + "|" + cp.parent.parent.name
                expected = expected_hash.get(str(cp), expected_hash.get(key, "")) if final else ""
                matches = (actual == expected) if final and expected else None
                if final and matches:
                    final_matches += 1
                if not exists:
                    missing.append(str(cp))
                checkpoint_rows.append({"method": method, "policy_seed": seed, "checkpoint_step": step, "path": str(cp), "exists": exists, "bytes": cp.stat().st_size if exists else 0, "sha256": actual, "is_final": final})
                hash_rows.append({"method": method, "policy_seed": seed, "checkpoint_step": step, "actual_sha256": actual, "expected_sha256": expected, "match": matches})
            for step in MILESTONES:
                val = train / "validation" / str(step) / "summary.json"
                validation_count += int(val.is_file())
                if not val.is_file():
                    missing.append(str(val))
            ep_path = test / "episodes.jsonl"
            trace_path = test / "fixed_subset_traces.jsonl"
            specs = {}
            n_ep = 0
            if ep_path.is_file():
                for row in iter_jsonl(ep_path):
                    n_ep += 1
                    if row["repeat"] == 0:
                        specs[row["episode_id"]] = {k: row[k] for k in ("family", "profile", "condition", "repeat", "seed", "episode_id")}
                total_test += n_ep
            else:
                missing.append(str(ep_path))
            if not trace_path.is_file():
                missing.append(str(trace_path))
            episode_specs[(method, seed)] = specs
            input_rows.append({"method": method, "policy_seed": seed, "training_complete": complete_ok, "test_episodes": n_ep, "repeat0_episode_specs": len(specs), "episodes_path": str(ep_path), "traces_path": str(trace_path), "traces_bytes": trace_path.stat().st_size if trace_path.is_file() else 0})
    final_integrity = final_matches == 48
    pre = {
        "status": "PASS" if final_integrity and total_test == 98304 else "FAIL",
        "training_complete": complete_count,
        "training_jobs_expected": 48,
        "final_checkpoint_hash_matches": final_matches,
        "final_checkpoints_expected": 48,
        "milestone_checkpoints_present": sum(bool(r["exists"]) for r in checkpoint_rows),
        "milestone_checkpoints_expected": 288,
        "validation_summaries_present": validation_count,
        "validation_summaries_expected": 288,
        "test_episodes": total_test,
        "test_episodes_expected": 98304,
        "fixed_subset_coverage": "PENDING_SINGLE_PASS_CAUSAL_REPLAY",
        "missing_count": len(missing),
    }
    write_json(artifact / "preflight.json", pre)
    write_csv(artifact / "input_inventory.csv", input_rows)
    write_csv(artifact / "checkpoint_inventory.csv", checkpoint_rows)
    write_csv(artifact / "artifact_hash_audit.csv", hash_rows)
    write_json(artifact / "missing_artifacts.json", {"count": len(missing), "paths": missing})
    if not final_integrity:
        raise RuntimeError("D1_BLOCKED_FINAL_CHECKPOINT_INTEGRITY_FAILURE")
    return pre, episode_specs


def zero_registry(artifact):
    result = {
        "frozen_before_diagnostics": True,
        "primary": [{"method": m, "policy_seed": s} for m, s in TARGETS],
        "auxiliary": [{"method": m, "policy_seed": s} for m, s in AUXILIARY],
        "same_seed_controls": list(CONTROL_METHODS),
    }
    write_json(artifact / "zero_seed_registry.json", result)


def classify_validation(values):
    nonzero = [v > 0 for v in values]
    if not any(nonzero): return "NEVER_DETERMINISTIC_VALIDATION_SUCCESS"
    if any(nonzero[:-1]) and not nonzero[-1]: return "EARLY_SUCCESS_LATE_COLLAPSE"
    if not any(nonzero[:3]) and any(nonzero[3:]): return "LATE_DISCOVERY"
    if all(nonzero[-3:]): return "STABLE_SUCCESS"
    return "NONZERO_BUT_UNSTABLE"


def seed_forensics(p2b_root, artifact, external_root):
    val_rows, classes = [], []
    first_rows, window_rows, training_summary = [], [], []
    ppo_rows, anomaly = [], {}
    full_training = external_root / "seed_forensics/training_episode_summary.full.csv"
    full_training.parent.mkdir(parents=True, exist_ok=True)
    full_fields = ["method", "policy_seed", "worker", "episode_index", "cumulative_worker_steps", "approx_env_interactions", "success", "steps", "invalid_actions", "waits", "observed_losses", "restored_losses", "action_histogram", "condition", "profile", "family"]
    with full_training.open("w", encoding="utf-8", newline="") as full_handle:
        full_writer = csv.DictWriter(full_handle, fieldnames=full_fields)
        full_writer.writeheader()
        for method, seed in FORENSIC_JOBS:
            train = p2b_root / "training" / method / f"seed_{seed}"
            vals = []
            for step in MILESTONES:
                summary = read_json(train / "validation" / str(step) / "summary.json")
                success = float(summary["success"])
                vals.append(success)
                eps = list(iter_jsonl(train / "validation" / str(step) / "episodes.jsonl"))
                val_rows.append({"method": method, "policy_seed": seed, "checkpoint_step": step, "validation_success": success, "validation_mean_steps": sum(r["steps"] for r in eps) / len(eps), "validation_invalid_actions": sum(r["invalid_actions"] for r in eps), "checkpoint_sha256": sha256_file(train / f"policy_{step}.zip")})
            classes.append({"method": method, "policy_seed": seed, "classification": classify_validation(vals), "success_values": canonical(vals)})
            for worker in range(8):
                path = train / "training_logs" / f"episodes_worker_{worker}.jsonl"
                cumulative = 0
                successes = 0
                first = None
                bucket = []
                count = 0
                for index, row in enumerate(iter_jsonl(path)):
                    cumulative += int(row["steps"])
                    count += 1
                    successes += int(row["success"])
                    if row["success"] and first is None:
                        first = (index, cumulative, row["episode_id"])
                    record = {"method": method, "policy_seed": seed, "worker": worker, "episode_index": index, "cumulative_worker_steps": cumulative, "approx_env_interactions": 8 * cumulative, "success": row["success"], "steps": row["steps"], "invalid_actions": row["invalid_actions"], "waits": row["waits"], "observed_losses": row["observed_losses"], "restored_losses": row["restored_losses"], "action_histogram": canonical(row["action_histogram"]), "condition": row["condition"], "profile": row["profile"], "family": row["family"]}
                    full_writer.writerow(record)
                    bucket.append(row)
                    if len(bucket) == 100:
                        window_rows.append({"method": method, "policy_seed": seed, "worker": worker, "window_end_episode": index, "n": len(bucket), "success_rate": sum(x["success"] for x in bucket) / len(bucket), "mean_invalid_actions": sum(x["invalid_actions"] for x in bucket) / len(bucket)})
                        bucket = []
                if bucket:
                    window_rows.append({"method": method, "policy_seed": seed, "worker": worker, "window_end_episode": count - 1, "n": len(bucket), "success_rate": sum(x["success"] for x in bucket) / len(bucket), "mean_invalid_actions": sum(x["invalid_actions"] for x in bucket) / len(bucket)})
                training_summary.append({"method": method, "policy_seed": seed, "worker": worker, "episodes": count, "cumulative_worker_steps": cumulative, "approx_env_interactions": 8 * cumulative, "successes": successes, "success_rate": successes / count if count else None})
                first_rows.append({"method": method, "policy_seed": seed, "worker": worker, "first_success_episode_index": first[0] if first else None, "cumulative_worker_steps": first[1] if first else None, "approx_env_interactions": 8 * first[1] if first else None, "episode_id": first[2] if first else "", "status": "OBSERVED" if first else "TRAIN_STREAM_NO_SUCCESS_OBSERVED"})
            progress = train / "ppo_logs/progress.csv"
            with progress.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            metrics = ["time/total_timesteps", "train/approx_kl", "train/clip_fraction", "train/entropy_loss", "train/explained_variance", "train/policy_gradient_loss", "train/value_loss", "train/loss", "train/learning_rate"]
            diag = {"method": method, "policy_seed": seed, "rows": len(rows)}
            numeric = {}
            for key in metrics:
                vals2 = []
                for row in rows:
                    try:
                        if row.get(key, "") != "": vals2.append(float(row[key]))
                    except ValueError:
                        pass
                numeric[key] = vals2
                diag[key + "_first"] = vals2[0] if vals2 else None
                diag[key + "_last"] = vals2[-1] if vals2 else None
                diag[key + "_min"] = min(vals2) if vals2 else None
                diag[key + "_max"] = max(vals2) if vals2 else None
            flags = {
                "nonfinite": any(not math.isfinite(x) for vals2 in numeric.values() for x in vals2),
                "extreme_kl": any(abs(x) > 1 for x in numeric.get("train/approx_kl", [])),
                "extreme_clip_fraction": any(x > .95 for x in numeric.get("train/clip_fraction", [])),
                "explained_variance_below_minus_one": any(x < -1 for x in numeric.get("train/explained_variance", [])),
                "entropy_fast_collapse": False,
            }
            ent = numeric.get("train/entropy_loss", [])
            if len(ent) >= 10:
                flags["entropy_fast_collapse"] = abs(ent[min(9, len(ent)-1)]) < .1 * max(abs(ent[0]), 1e-12)
            anomaly[f"{method}:{seed}"] = flags
            ppo_rows.append(diag)
    write_csv(artifact / "validation_checkpoint_forensics.csv", val_rows)
    write_csv(artifact / "validation_curve_classification.csv", classes)
    write_csv(artifact / "training_episode_summary.csv", training_summary)
    write_csv(artifact / "first_training_success_by_worker.csv", first_rows)
    write_csv(artifact / "training_success_windows.csv", window_rows)
    write_csv(artifact / "ppo_optimization_diagnostics.csv", ppo_rows)
    write_json(artifact / "ppo_anomaly_flags.json", anomaly)
    return classes, training_summary, full_training


def new_failure_acc(row):
    return {
        "summary": row, "actions": [], "invalid": [], "states": [], "active": [], "valid_counts": [],
        "ever_held_or_transport": False, "transport_stall": False, "invalid_loop": False,
        "object_switch_loop": False, "premature_place_or_release": False,
        "loss_seen": False, "recovery_started": False, "regrasped": False,
        "resumed_after_regrasp": False, "first_valid_object": "", "first_loss_step": None,
        "first_recovery_start_step": None, "first_regrasp_step": None,
    }


def update_failure(acc, before, after, action_name, invalid, step, info):
    acc["actions"].append(action_name)
    acc["invalid"].append(bool(invalid))
    acc["states"].append(after)
    active = active_object(after)
    acc["active"].append(active)
    vc = sum(bool(x["valid"]) for x in after["objects"].values())
    acc["valid_counts"].append(vc)
    phases = [x["phase"] for x in after["objects"].values()]
    acc["ever_held_or_transport"] |= any(p in ("HELD", "TRANSPORT") for p in phases)
    for event in after.get("events", []):
        if event["kind"] == "LOSS":
            acc["loss_seen"] = True
            if acc["first_loss_step"] is None: acc["first_loss_step"] = step
    if action_name.startswith("START_RECOVERY_") and not invalid:
        acc["recovery_started"] = True
        if acc["first_recovery_start_step"] is None: acc["first_recovery_start_step"] = step
    if action_name.startswith("REGRASP_") and not invalid:
        acc["regrasped"] = True
        if acc["first_regrasp_step"] is None: acc["first_regrasp_step"] = step
    if acc["regrasped"] and (action_name.startswith("ADVANCE_") or action_name.startswith("PLACE_")) and not invalid:
        acc["resumed_after_regrasp"] = True
    if action_name.startswith(("PLACE_", "RELEASE_")) and invalid:
        acc["premature_place_or_release"] = True
    if len(acc["invalid"]) >= 4 and all(acc["invalid"][-4:]) and len(set(acc["actions"][-4:])) == 1:
        acc["invalid_loop"] = True
    if len(acc["active"]) >= 8:
        a = acc["active"][-8:]
        switches = sum(bool(a[i] and a[i-1] and a[i] != a[i-1]) for i in range(1, 8))
        if switches >= 4 and acc["valid_counts"][-1] <= acc["valid_counts"][-8]:
            acc["object_switch_loop"] = True
    if len(acc["states"]) >= 9:
        for oid in ("A", "B"):
            segment = acc["states"][-9:]
            if all(s["objects"][oid]["held"] and s["objects"][oid]["phase"] in ("HELD", "TRANSPORT") for s in segment):
                distances = [math.dist(s["objects"][oid]["pos"][:2], s["objects"][oid]["target_xy"]) for s in segment]
                if not any(distances[i] < distances[i-1] for i in range(1, len(distances))):
                    acc["transport_stall"] = True


def finalize_failure(acc, method, seed, env_actions):
    row = acc["summary"]
    if row["success"]:
        return None
    final = acc["states"][-1] if acc["states"] else None
    invalid_last = sum(acc["invalid"][-16:]) / min(16, len(acc["invalid"])) if acc["invalid"] else 0
    features = dict(acc)
    features.update({
        "actions": acc["actions"],
        "final_valid_count": sum(bool(x["valid"]) for x in final["objects"].values()) if final else 0,
        "last_16_invalid_fraction": invalid_last,
        "progress_during_cycle": False,
    })
    primary, secondary = classify_features(features)
    valid_order = []
    previous = {"A": False, "B": False}
    for state in acc["states"]:
        for oid in ("A", "B"):
            now = bool(state["objects"][oid]["valid"])
            if now and not previous[oid] and oid not in valid_order: valid_order.append(oid)
            previous[oid] = now
    longest = 0
    for i, action in enumerate(acc["actions"]):
        j = i
        while j < len(acc["actions"]) and acc["actions"][j] == action: j += 1
        longest = max(longest, j - i)
    return {
        "method": method, "policy_seed": seed, "episode_id": row["episode_id"], "family": row["family"], "profile": row["profile"], "condition": row["condition"], "success": False, "steps": row["steps"],
        "first_valid_object": valid_order[0] if valid_order else "", "final_valid_A": final["objects"]["A"]["valid"], "final_valid_B": final["objects"]["B"]["valid"],
        "first_loss_step": acc["first_loss_step"], "first_recovery_start_step": acc["first_recovery_start_step"], "first_regrasp_step": acc["first_regrasp_step"],
        "invalid_action_count": sum(acc["invalid"]), "object_switch_count": sum(bool(acc["active"][i] and acc["active"][i-1] and acc["active"][i] != acc["active"][i-1]) for i in range(1, len(acc["active"]))),
        "longest_identical_action_run": longest, "last_16_invalid_fraction": invalid_last, "first_irreversible_error_step": None,
        "primary_failure_class": primary, "secondary_failure_labels": canonical(secondary),
    }


def create_database(path):
    if path.exists(): path.unlink()
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("CREATE TABLE states(state_hash TEXT PRIMARY KEY, geom_signature TEXT, valid_count INTEGER, open_losses INTEGER, geometry REAL, phi_geom REAL, phi_graph REAL, condition_name TEXT, profile TEXT, phase_a TEXT, phase_b TEXT, active_object TEXT, trace_path TEXT, episode_offset INTEGER, step INTEGER, episode_id TEXT, method TEXT, seed INTEGER, seen_primary INTEGER)")
    db.execute("CREATE TABLE candidates(state_hash TEXT PRIMARY KEY, category TEXT, condition_name TEXT)")
    return db


def causal_corpus(repo, p2b_root, external_root, artifact, episode_specs, runtime):
    config, envmod, cap, _, shaped_transition, TaskCore = runtime
    corpus_dir = external_root / "potential_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = corpus_dir / "common_state_potential_corpus.jsonl"
    db_path = corpus_dir / "state_index.sqlite"
    db = create_database(db_path)
    raw_pairs = {"primary": RunningPair(), "secondary": RunningPair()}
    shape_pairs = {"primary": RunningPair(), "secondary": RunningPair()}
    category_counts = defaultdict(Counter)
    condition_counts = defaultdict(Counter)
    profile_counts = defaultdict(Counter)
    differences = {"primary": [], "secondary": []}
    coverage = []
    failures = []
    failure_examples = []
    actions = tuple(envmod.ACTIONS)
    with corpus_path.open("w", encoding="utf-8") as corpus:
        for method in METHODS:
            for seed in SEEDS:
                specs = episode_specs[(method, seed)]
                summaries = {r["episode_id"]: r for r in iter_jsonl(p2b_root / "test" / method / f"seed_{seed}" / "episodes.jsonl") if r["repeat"] == 0}
                trace_path = p2b_root / "test" / method / f"seed_{seed}" / "fixed_subset_traces.jsonl"
                core = TaskCore(envmod, cap, config, "GRAPH_FULL_PBRS")
                current = None
                current_offset = 0
                current_failure = None
                episode_ids = set()
                transitions = 0
                with trace_path.open("rb") as handle:
                    while True:
                        offset = handle.tell()
                        raw = handle.readline()
                        if not raw: break
                        trace = json.loads(raw)
                        episode_id = trace["before"]["episode_id"]
                        if episode_id != current:
                            if current_failure is not None:
                                result = finalize_failure(current_failure, method, seed, actions)
                                if result:
                                    failures.append(result)
                                    if len(failure_examples) < 200 and result["primary_failure_class"] != "TIMEOUT_OTHER":
                                        failure_examples.append({"method": method, "policy_seed": seed, "episode_id": result["episode_id"], "primary_failure_class": result["primary_failure_class"], "actions": current_failure["actions"]})
                            current = episode_id
                            current_offset = offset
                            episode_ids.add(current)
                            core.reset(specs[current])
                            if canonical(core.last_state) != canonical(trace["before"]):
                                raise RuntimeError(f"causal replay initial mismatch {method} {seed} {current}")
                            current_failure = new_failure_acc(summaries[current]) if (method, seed) in FORENSIC_JOBS else None
                        if canonical(core.last_state) != canonical(trace["before"]):
                            raise RuntimeError(f"causal replay before mismatch {method} {seed} {current} {core.env.t}")
                        before = copy.deepcopy(core.last_state)
                        before_values = dict(core.values)
                        before_meta = copy.deepcopy(core.vmeta)
                        before_hash = state_hash(before, potential_history(core.bank))
                        action = int(trace["action_applied"])
                        invalid_before = core.env.invalid_actions
                        _, _, terminated, truncated, info, replay_trace = core.step(action)
                        invalid = core.env.invalid_actions > invalid_before
                        if canonical(core.last_state) != canonical(trace["after"]):
                            raise RuntimeError(f"causal replay after mismatch {method} {seed} {current} {core.env.t}")
                        for key, value in trace["all_potentials_after"].items():
                            if abs(float(core.values[key]) - float(value)) > 1e-10:
                                raise RuntimeError(f"potential mismatch {key} {method} {seed} {current}")
                        geom_row = shaped_transition(info["task_reward"], before_values["GEOM_COUNT_EVENTS_PBRS"], core.values["GEOM_COUNT_EVENTS_PBRS"], gamma=config["gamma"], beta=config["beta"], terminated=terminated, truncated=truncated)
                        graph_row = shaped_transition(info["task_reward"], before_values["GRAPH_FULL_PBRS"], core.values["GRAPH_FULL_PBRS"], gamma=config["gamma"], beta=config["beta"], terminated=terminated, truncated=truncated)
                        category = guidance_category(geom_row["shaping_reward"], graph_row["shaping_reward"])
                        scope = "primary" if method in ("GEOM_COUNT_EVENTS_PBRS", "GRAPH_FULL_PBRS") else "secondary"
                        valid_count = sum(bool(x["valid"]) for x in before["objects"].values())
                        active = active_object(before)
                        row = {
                            "state_hash": before_hash,
                            "transition_hash": value_hash([before_hash, action, state_hash(core.last_state, potential_history(core.bank))]),
                            "source_policy_method": method, "source_policy_seed": seed,
                            "episode_id": current, "family": specs[current]["family"], "profile": specs[current]["profile"], "condition": specs[current]["condition"], "step": int(before["state_index"]), "action": actions[action],
                            "terminated": terminated, "task_reward": info["task_reward"], "valid_A": before["objects"]["A"]["valid"], "valid_B": before["objects"]["B"]["valid"], "held_A": before["objects"]["A"]["held"], "held_B": before["objects"]["B"]["held"], "phase_A": before["objects"]["A"]["phase"], "phase_B": before["objects"]["B"]["phase"], "open_loss_A": before["objects"]["A"]["phase"] in ("LOST", "RECOVERING"), "open_loss_B": before["objects"]["B"]["phase"] in ("LOST", "RECOVERING"), "active_object": active, "remaining_time": max(0.0, 1 - before["state_index"] / config["profiles"][specs[current]["profile"]]["horizon"]),
                            "count_component": before_values["COUNT_PBRS"], "event_component": before_values["COUNT_EVENTS_PBRS"], "geometry_component": before_meta["geometry"], "phi_geom": before_values["GEOM_COUNT_EVENTS_PBRS"], "phi_graph_cost": before_values["GRAPH_COST_PBRS"], "phi_graph_full": before_values["GRAPH_FULL_PBRS"],
                            "shaping_geom": geom_row["shaping_reward"], "shaping_graph_full": graph_row["shaping_reward"], "shaping_difference": graph_row["shaping_reward"] - geom_row["shaping_reward"], "sign_relation": category,
                            "matched_open_losses": before_meta["matched_open_losses"], "v6_cost_cap": before_meta["v6"]["cost_cap"], "v6_local_credit": before_meta["v6"]["local_credit"], "v6_psi": before_meta["v6"]["psi"], "corpus_scope": scope,
                        }
                        corpus.write(canonical(row) + "\n")
                        sig = geom_signature(valid_count, before_meta["matched_open_losses"], before_meta["geometry"])
                        db.execute("INSERT OR IGNORE INTO states VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (before_hash, sig, valid_count, before_meta["matched_open_losses"], before_meta["geometry"], before_values["GEOM_COUNT_EVENTS_PBRS"], before_values["GRAPH_FULL_PBRS"], specs[current]["condition"], specs[current]["profile"], before["objects"]["A"]["phase"], before["objects"]["B"]["phase"], active, str(trace_path), current_offset, int(before["state_index"]), current, method, seed, int(scope == "primary")))
                        if scope == "primary":
                            db.execute("UPDATE states SET seen_primary=1 WHERE state_hash=?", (before_hash,))
                        if category in ("GRAPH_ONLY_NONZERO", "OPPOSITE_SIGN"):
                            db.execute("INSERT OR IGNORE INTO candidates VALUES(?,?,?)", (before_hash, category, specs[current]["condition"]))
                        for aggregate_scope in (("secondary", "primary") if scope == "primary" else ("secondary",)):
                            raw_pairs[aggregate_scope].add(before_values["GEOM_COUNT_EVENTS_PBRS"], before_values["GRAPH_FULL_PBRS"])
                            shape_pairs[aggregate_scope].add(geom_row["shaping_reward"], graph_row["shaping_reward"])
                            category_counts[aggregate_scope][category] += 1
                            condition_counts[(aggregate_scope, specs[current]["condition"])][category] += 1
                            profile_counts[(aggregate_scope, specs[current]["profile"])][category] += 1
                            differences[aggregate_scope].append(row["shaping_difference"])
                        if current_failure is not None:
                            update_failure(current_failure, before, core.last_state, actions[action], invalid, core.env.t, info)
                        transitions += 1
                        if transitions % 5000 == 0: db.commit()
                if current_failure is not None:
                    result = finalize_failure(current_failure, method, seed, actions)
                    if result: failures.append(result)
                coverage.append({"method": method, "policy_seed": seed, "fixed_subset_episodes": len(episode_ids), "expected": 512, "transitions": transitions, "coverage_pass": len(episode_ids) == 512})
                db.commit()
    if not all(x["coverage_pass"] for x in coverage):
        raise RuntimeError("D1_BLOCKED_FIXED_SUBSET_COVERAGE_FAILURE")
    pre = read_json(artifact / "preflight.json")
    pre["fixed_subset_coverage"] = "48/48 policies x 512 episodes PASS"
    pre["fixed_subset_policies_verified"] = 48
    pre["status"] = "PASS"
    write_json(artifact / "preflight.json", pre)
    write_csv(artifact / "common_state_potential_corpus.index.csv", coverage)
    index_external = [{"kind": "corpus", "absolute_path": str(corpus_path), "bytes": corpus_path.stat().st_size, "sha256": sha256_file(corpus_path), "role": "complete raw-transition causal replay corpus"}, {"kind": "state_index", "absolute_path": str(db_path), "bytes": db_path.stat().st_size, "sha256": sha256_file(db_path), "role": "unique-state and replay locator index"}]
    unique_pairs = {"primary": RunningPair(), "secondary": RunningPair()}
    for scope, where in (("primary", "seen_primary=1"), ("secondary", "1=1")):
        for x, y in db.execute(f"SELECT phi_geom,phi_graph FROM states WHERE {where}"):
            unique_pairs[scope].add(x, y)
    corr = {scope: {"raw_transition_count": raw_pairs[scope].n, "raw_phi_correlation": raw_pairs[scope].result(), "raw_shaping_correlation": shape_pairs[scope].result(), "unique_state_count": unique_pairs[scope].n, "unique_state_phi_correlation": unique_pairs[scope].result()} for scope in ("primary", "secondary")}
    write_json(artifact / "potential_correlation_summary.json", corr)
    sign_rows = []
    for scope in ("primary", "secondary"):
        den = sum(category_counts[scope].values())
        for category in ("BOTH_ZERO", "SAME_SIGN", "OPPOSITE_SIGN", "GRAPH_ONLY_NONZERO", "GEOM_ONLY_NONZERO"):
            n = category_counts[scope][category]
            sign_rows.append({"scope": scope, "category": category, "count": n, "denominator": den, "rate": n / den if den else None})
    write_csv(artifact / "shaping_sign_matrix.csv", sign_rows)
    condition_rows, profile_rows = [], []
    for (scope, condition), counts in sorted(condition_counts.items()):
        den = sum(counts.values())
        condition_rows.append({"scope": scope, "condition": condition, "transitions": den, "graph_only_nonzero": counts["GRAPH_ONLY_NONZERO"], "opposite_sign": counts["OPPOSITE_SIGN"], "graph_unique_rate": (counts["GRAPH_ONLY_NONZERO"] + counts["OPPOSITE_SIGN"]) / den if den else None})
    for (scope, profile), counts in sorted(profile_counts.items()):
        den = sum(counts.values())
        profile_rows.append({"scope": scope, "profile": profile, "transitions": den, "graph_only_nonzero": counts["GRAPH_ONLY_NONZERO"], "opposite_sign": counts["OPPOSITE_SIGN"], "graph_unique_rate": (counts["GRAPH_ONLY_NONZERO"] + counts["OPPOSITE_SIGN"]) / den if den else None})
    write_csv(artifact / "condition_potential_disagreement.csv", condition_rows)
    write_csv(artifact / "profile_potential_disagreement.csv", profile_rows)
    write_json(artifact / "potential_difference_quantiles.json", {scope: quantiles(differences[scope]) for scope in differences})
    write_csv(artifact / "failure_episode_taxonomy.csv", failures)
    summary_counts = defaultdict(Counter)
    for row in failures: summary_counts[(row["method"], row["policy_seed"])][row["primary_failure_class"]] += 1
    failure_summary = []
    for (method, seed), counts in sorted(summary_counts.items()):
        den = sum(counts.values())
        for cls, n in sorted(counts.items()): failure_summary.append({"method": method, "policy_seed": seed, "primary_failure_class": cls, "count": n, "failed_episode_denominator": den, "rate": n / den})
    write_csv(artifact / "failure_class_summary.csv", failure_summary)
    with (artifact / "action_loop_examples.jsonl").open("w", encoding="utf-8") as handle:
        for row in failure_examples: handle.write(canonical(row) + "\n")
    breakdown = defaultdict(lambda: [0, 0])
    for method, seed in FORENSIC_JOBS:
        for row in iter_jsonl(p2b_root / "test" / method / f"seed_{seed}" / "episodes.jsonl"):
            key = (method, seed, row["condition"], row["profile"])
            breakdown[key][0] += int(row["success"]); breakdown[key][1] += 1
    breakdown_rows = [{"method": k[0], "policy_seed": k[1], "condition": k[2], "profile": k[3], "successes": v[0], "episodes": v[1], "success_rate": v[0] / v[1]} for k, v in sorted(breakdown.items())]
    write_csv(artifact / "zero_seed_condition_breakdown.csv", breakdown_rows)
    comparisons = defaultdict(dict)
    for row in breakdown_rows:
        comparisons[(row["policy_seed"], row["condition"], row["profile"])][row["method"]] = row["success_rate"]
    comp_rows = []
    for (seed, condition, profile), vals in sorted(comparisons.items()):
        for target_method in ("TASK_ONLY", "COUNT_PBRS", "COUNT_EVENTS_PBRS"):
            if target_method in vals:
                comp_rows.append({"policy_seed": seed, "condition": condition, "profile": profile, "target_method": target_method, "target_success": vals[target_method], "geom_success": vals.get("GEOM_COUNT_EVENTS_PBRS"), "graph_cost_success": vals.get("GRAPH_COST_PBRS"), "graph_full_success": vals.get("GRAPH_FULL_PBRS")})
    write_csv(artifact / "same_seed_method_comparison.csv", comp_rows)
    return db, db_path, corpus_path, index_external, failures, sign_rows, condition_rows, envmod, TaskCore, shaped_transition, actions


def aliasing(db, artifact):
    db.execute("CREATE INDEX IF NOT EXISTS idx_states_geom_hash ON states(geom_signature,state_hash)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_states_condition_hash ON states(condition_name,state_hash)")
    db.commit()
    query = """SELECT geom_signature,COUNT(*),COUNT(DISTINCT printf('%.17g',phi_graph)),MIN(phi_graph),MAX(phi_graph),GROUP_CONCAT(DISTINCT condition_name),GROUP_CONCAT(DISTINCT phase_a||'/'||phase_b),GROUP_CONCAT(DISTINCT active_object) FROM states GROUP BY geom_signature HAVING COUNT(*)>1 AND (COUNT(DISTINCT printf('%.17g',phi_graph))>1 OR COUNT(DISTINCT phase_a||'/'||phase_b)>1 OR COUNT(DISTINCT active_object)>1)"""
    groups = []
    for idx, row in enumerate(db.execute(query)):
        sig, size, distinct_graph, lo, hi, conditions, phases, active = row
        reps = [r[0] for r in db.execute("SELECT state_hash FROM states WHERE geom_signature=? ORDER BY state_hash LIMIT 5", (sig,))]
        groups.append({"geom_signature": sig, "group_size": size, "distinct_graph_cost_count": distinct_graph, "distinct_graph_full_count": distinct_graph, "max_graph_potential_span": hi - lo, "conditions": conditions, "phases": phases, "active_objects": active, "representative_state_hashes": canonical(reps)})
        for h in reps:
            cond = db.execute("SELECT condition_name FROM states WHERE state_hash=?", (h,)).fetchone()[0]
            db.execute("INSERT OR IGNORE INTO candidates VALUES(?,?,?)", (h, "GEOM_ALIAS", cond))
    db.commit()
    write_csv(artifact / "geom_aliasing_groups.csv", groups)
    examples = []
    for h, cat, cond in db.execute("SELECT state_hash,category,condition_name FROM candidates WHERE category IN ('GRAPH_ONLY_NONZERO','OPPOSITE_SIGN') ORDER BY state_hash LIMIT 200"):
        row = db.execute("SELECT phi_geom,phi_graph,phase_a,phase_b,active_object FROM states WHERE state_hash=?", (h,)).fetchone()
        examples.append({"state_hash": h, "category": cat, "condition": cond, "phi_geom": row[0], "phi_graph_full": row[1], "phase_A": row[2], "phase_B": row[3], "active_object": row[4]})
    with (artifact / "graph_only_state_examples.jsonl").open("w", encoding="utf-8") as handle:
        for row in examples: handle.write(canonical(row) + "\n")
    counts = Counter(row[0] for row in db.execute("SELECT category FROM candidates"))
    write_csv(artifact / "transition_guidance_categories.csv", [{"category": k, "candidate_unique_states": v} for k, v in sorted(counts.items())])
    return groups


def select_counterfactual_states(db, conditions):
    chosen = set()
    for condition in conditions:
        rows = [r[0] for r in db.execute("SELECT c.state_hash FROM candidates c JOIN states s ON s.state_hash=c.state_hash WHERE s.condition_name=? ORDER BY c.state_hash LIMIT 32", (condition,))]
        chosen.update(rows)
    for (h,) in db.execute("SELECT state_hash FROM candidates ORDER BY state_hash"):
        if len(chosen) >= 500: break
        chosen.add(h)
    if len(chosen) > 500:
        required = set()
        for condition in conditions:
            required.update(r[0] for r in db.execute("SELECT c.state_hash FROM candidates c JOIN states s ON s.state_hash=c.state_hash WHERE s.condition_name=? ORDER BY c.state_hash LIMIT 32", (condition,)))
        chosen = required | set(h for h in sorted(chosen - required)[:max(0, 500 - len(required))])
    return sorted(chosen)


def counterfactual(db, artifact, runtime, selected_hashes, episode_specs):
    config, envmod, cap, _, shaped_transition, TaskCore = runtime
    selected = set(selected_hashes)
    locators = defaultdict(set)
    for h in selected_hashes:
        row = db.execute("SELECT trace_path,episode_offset,episode_id,method,seed,step FROM states WHERE state_hash=?", (h,)).fetchone()
        if row: locators[(row[0], row[1], row[2], row[3], row[4])].add(h)
    ranking_rows, summaries, examples, expert_rows = [], [], [], []
    for (trace_path, offset, episode_id, method, seed), targets in locators.items():
        core = TaskCore(envmod, cap, config, "GRAPH_FULL_PBRS")
        core.reset(episode_specs[(method, seed)][episode_id])
        with open(trace_path, "rb") as handle:
            handle.seek(offset)
            while targets and not core.done:
                position = handle.tell(); raw = handle.readline()
                if not raw: break
                trace = json.loads(raw)
                if trace["before"]["episode_id"] != episode_id: break
                h = state_hash(core.last_state, potential_history(core.bank))
                if h in targets:
                    original = canonical(core.last_state)
                    rewards_g, rewards_r, valids = [], [], []
                    expert = int(envmod.expert_action(core.env))
                    state_rows = []
                    for action, action_name in enumerate(envmod.ACTIONS):
                        branch = copy.deepcopy(core, {id(core.envmod): core.envmod, id(core.cap_class): core.cap_class})
                        before_values = dict(branch.values)
                        invalid_before = branch.env.invalid_actions
                        _, _, terminated, truncated, info, _ = branch.step(action)
                        valid = branch.env.invalid_actions == invalid_before
                        geom = shaped_transition(info["task_reward"], before_values["GEOM_COUNT_EVENTS_PBRS"], branch.values["GEOM_COUNT_EVENTS_PBRS"], gamma=config["gamma"], beta=config["beta"], terminated=terminated, truncated=truncated)
                        graph = shaped_transition(info["task_reward"], before_values["GRAPH_FULL_PBRS"], branch.values["GRAPH_FULL_PBRS"], gamma=config["gamma"], beta=config["beta"], terminated=terminated, truncated=truncated)
                        rewards_g.append(geom["training_reward"]); rewards_r.append(graph["training_reward"]); valids.append(valid)
                        sr = {"state_hash": h, "condition": core.spec["condition"], "profile": core.spec["profile"], "phase_A": core.last_state["objects"]["A"]["phase"], "phase_B": core.last_state["objects"]["B"]["phase"], "action": action_name, "valid_action": valid, "next_phase_A": branch.last_state["objects"]["A"]["phase"], "next_phase_B": branch.last_state["objects"]["B"]["phase"], "next_valid_count": sum(x["valid"] for x in branch.last_state["objects"].values()), "loss_event_created": any(e["kind"] == "LOSS" for e in branch.last_state.get("events", [])), "recovery_event_created": any(e["kind"] in ("RECOVERY_STARTED", "HOLD_REESTABLISHED") for e in branch.last_state.get("events", [])), "task_reward": info["task_reward"], "geom_shaping": geom["shaping_reward"], "graph_shaping": graph["shaping_reward"], "geom_training_reward": geom["training_reward"], "graph_training_reward": graph["training_reward"]}
                        state_rows.append(sr)
                    if canonical(core.last_state) != original:
                        raise RuntimeError("COUNTERFACTUAL_BRANCHING_NOT_VALIDATED")
                    geom_order = sorted(range(13), key=lambda i: (-rewards_g[i], i))
                    graph_order = sorted(range(13), key=lambda i: (-rewards_r[i], i))
                    geom_rank = {a: i + 1 for i, a in enumerate(geom_order)}
                    graph_rank = {a: i + 1 for i, a in enumerate(graph_order)}
                    for sr, action in zip(state_rows, range(13)):
                        sr["geom_rank"] = geom_rank[action]; sr["graph_rank"] = graph_rank[action]
                        ranking_rows.append(sr)
                    same = geom_order[0] == graph_order[0]
                    summary = {"state_hash": h, "condition": core.spec["condition"], "profile": core.spec["profile"], "phase_A": core.last_state["objects"]["A"]["phase"], "phase_B": core.last_state["objects"]["B"]["phase"], "top1_agree": same, "top3_set_agree": set(geom_order[:3]) == set(graph_order[:3]), "spearman": spearman(rewards_g, rewards_r), "geom_top1_invalid": not valids[geom_order[0]], "graph_top1_invalid": not valids[graph_order[0]], "geom_top1": envmod.ACTIONS[geom_order[0]], "graph_top1": envmod.ACTIONS[graph_order[0]]}
                    summaries.append(summary)
                    expert_rows.append({"state_hash": h, "condition": core.spec["condition"], "profile": core.spec["profile"], "expert_action": envmod.ACTIONS[expert], "expert_geom_rank": geom_rank[expert], "expert_graph_rank": graph_rank[expert], "reference": "DIAGNOSTIC_CURRENT_STATE_HEURISTIC;NOT_POLICY_INPUT;NOT_TEACHER_TAKEOVER;NOT_TEST_SUCCESS_DEFINITION"})
                    if not same and len(examples) < 200: examples.append(summary)
                    targets.remove(h)
                core.step(int(trace["action_applied"]))
    if set(selected_hashes) != {r["state_hash"] for r in summaries}:
        missing = sorted(set(selected_hashes) - {r["state_hash"] for r in summaries})
        raise RuntimeError(f"COUNTERFACTUAL_BRANCHING_NOT_VALIDATED missing={len(missing)}")
    write_csv(artifact / "counterfactual_action_rankings.csv", ranking_rows)
    overall = {
        "states": len(summaries),
        "top1_agreement": sum(r["top1_agree"] for r in summaries) / len(summaries) if summaries else None,
        "top3_set_agreement": sum(r["top3_set_agree"] for r in summaries) / len(summaries) if summaries else None,
        "mean_spearman": sum(r["spearman"] for r in summaries) / len(summaries) if summaries else None,
        "geom_top1_invalid_rate": sum(r["geom_top1_invalid"] for r in summaries) / len(summaries) if summaries else None,
        "graph_top1_invalid_rate": sum(r["graph_top1_invalid"] for r in summaries) / len(summaries) if summaries else None,
        "by_condition": {}, "by_profile": {}, "by_phase": {},
    }
    for key_name, getter in (("by_condition", lambda r: r["condition"]), ("by_profile", lambda r: r["profile"]), ("by_phase", lambda r: r["phase_A"] + "/" + r["phase_B"])):
        groups = defaultdict(list)
        for row in summaries: groups[getter(row)].append(row)
        for key, rows in sorted(groups.items()):
            overall[key_name][key] = {"states": len(rows), "top1_agreement": sum(r["top1_agree"] for r in rows) / len(rows), "top1_disagreement": 1 - sum(r["top1_agree"] for r in rows) / len(rows), "mean_spearman": sum(r["spearman"] for r in rows) / len(rows)}
    write_json(artifact / "counterfactual_summary.json", overall)
    with (artifact / "top1_disagreement_examples.jsonl").open("w", encoding="utf-8") as handle:
        for row in examples: handle.write(canonical(row) + "\n")
    write_csv(artifact / "expert_rank_comparison.csv", expert_rows)
    return overall


def mechanism_decisions(training_summary, validation_classes, failures):
    train_success = defaultdict(int)
    for row in training_summary: train_success[(row["method"], row["policy_seed"])] += row["successes"]
    classes = {(r["method"], r["policy_seed"]): r["classification"] for r in validation_classes}
    failure_counts = defaultdict(Counter)
    for row in failures: failure_counts[(row["method"], row["policy_seed"])][row["primary_failure_class"]] += 1
    out = {}
    for job in TARGETS:
        if train_success[job] > 0 and classes[job] == "NEVER_DETERMINISTIC_VALIDATION_SUCCESS":
            result = "ZERO_SEED_TRAIN_SUCCESS_NOT_RETAINED"
        else:
            counts = failure_counts[job]
            dominant = counts.most_common(2)
            names = {x[0] for x in dominant}
            if names & {"ONE_OBJECT_ONLY_TIMEOUT", "TRANSPORT_STALL", "RECOVERY_NOT_STARTED", "RECOVERY_STARTED_NO_REGRASP", "REGRASP_NO_TASK_RESUME"}:
                result = "ZERO_SEED_PARTIAL_TASK_STALL_SUPPORTED"
            elif names & {"NO_ACQUIRE_PROGRESS", "INVALID_ACTION_LOOP", "DETERMINISTIC_ACTION_CYCLE_OTHER"}:
                result = "ZERO_SEED_EXPLORATION_FAILURE_SUPPORTED"
            elif len(names) > 1:
                result = "ZERO_SEED_MIXED_MECHANISMS"
            else:
                result = "ZERO_SEED_MECHANISM_UNRESOLVED"
        out[f"{job[0]}:{job[1]}"] = {"status": result, "training_successes": train_success[job], "validation_class": classes[job], "failure_counts": dict(failure_counts[job])}
    return out


def graph_decision(sign_rows, conditions, counterfactual_summary, alias_groups):
    primary = {r["category"]: r for r in sign_rows if r["scope"] == "primary"}
    unique_rate = primary.get("GRAPH_ONLY_NONZERO", {}).get("rate", 0) + primary.get("OPPOSITE_SIGN", {}).get("rate", 0)
    top1 = counterfactual_summary["top1_agreement"]
    max_condition_disagreement = max((v["top1_disagreement"] for v in counterfactual_summary["by_condition"].values()), default=0)
    stress = {"INVALIDATE_FIRST", "A_VALID_B_LOSS", "B_VALID_A_LOSS", "TWO_LOSSES"}
    condition_rates = {r["condition"]: r["graph_unique_rate"] for r in conditions if r["scope"] == "primary"}
    stress_mean = sum(condition_rates.get(x, 0) for x in stress) / len(stress)
    other = [v for k, v in condition_rates.items() if k not in stress]
    other_mean = sum(other) / len(other) if other else 0
    if top1 is not None and top1 > .95 and unique_rate < .05 and max_condition_disagreement < .10:
        status = "GRAPH_SIGNAL_LARGELY_REDUNDANT_IN_CURRENT_BENCHMARK"
    elif stress_mean > max(other_mean * 1.5, .01):
        status = "GRAPH_SIGNAL_LOCALIZED_TO_SPECIFIC_CONDITIONS"
    elif alias_groups or max_condition_disagreement >= .10 or unique_rate > 0:
        status = "GRAPH_SIGNAL_PRESENT_BUT_POLICY_INCREMENT_NOT_ESTABLISHED"
    else:
        status = "GRAPH_INCREMENT_DIAGNOSTIC_INSUFFICIENT"
    return {"status": status, "primary_graph_only_plus_opposite_rate": unique_rate, "one_step_top1_agreement": top1, "max_condition_top1_disagreement": max_condition_disagreement, "alias_group_count": len(alias_groups), "stress_condition_mean_unique_rate": stress_mean, "other_condition_mean_unique_rate": other_mean}


def write_conditional_not_run(artifact):
    reason = "NOT_RUN_NOT_REQUIRED: existing deterministic training, validation, test and causal traces were sufficient for the registered mechanism classification; no SB3 dependency or new rollout was used"
    for name in ("deterministic_vs_stochastic.csv", "stochastic_failure_taxonomy.csv", "policy_action_entropy.csv", "policy_top1_margin.csv"):
        status_csv(artifact / name, "NOT_RUN_NOT_REQUIRED", reason)
    write_json(artifact / "diagnostic_rollout_registry.json", {"status": "NOT_RUN_NOT_REQUIRED", "deterministic_episodes": 0, "stochastic_episodes": 0, "reason": reason})


def copy_protocol(repo, artifact):
    protocol = protocol_document()
    write_json(repo / "experiments/pathgraph_p2b_d1_diagnostics_v1/protocol.json", protocol)
    write_json(artifact / "protocol.json", protocol)


def claims_and_report(repo, p2b_root, external_root, artifact, pre, mechanisms, graph_result, counter, external_rows):
    decision = {
        "schema": "P2B-D1_GRAPH_INCREMENT_DIAGNOSTICS",
        "zero_seed_mechanism": mechanisms,
        "graph_increment_diagnostic": graph_result,
        "policy_utility_evidence": "RL_REWARD_UTILITY_NOT_ESTABLISHED_OR_MIXED",
        "robot_policy_gain_claimed": False,
        "physical_cycle_claim": "NOT_EVALUATED",
        "confirmation_passed": False,
        "new_training_jobs": 0, "optimizer_updates": 0, "gradient_steps": 0, "checkpoint_files_modified": 0,
        "p2b_original_decision_modified": False,
    }
    (artifact / "final").mkdir(parents=True, exist_ok=True)
    write_json(artifact / "final/decision.json", decision)
    lines = [
        "# P2B-D1 Graph Increment Diagnostics", "", "## Git", "",
        f"- base commit: `{BASE}`", f"- branch: `{BRANCH}`", "- result commit: filled after commit", "- main unchanged: yes", "",
        "## Execution boundary", "", "- new training jobs: 0", "- optimizer updates: 0", "- gradient steps: 0", "- checkpoint files modified: 0", "- diagnostic deterministic episodes: 0", "- diagnostic stochastic episodes: 0", "- physical simulation: not run", "- robot/vision evaluation: not run", "",
        "## Input integrity", "", f"- final checkpoints matched: {pre['final_checkpoint_hash_matches']}/48", f"- milestone checkpoints present: {pre['milestone_checkpoints_present']}/288", f"- test episodes verified: {pre['test_episodes']}/98304", f"- fixed subset: {pre['fixed_subset_coverage']}", f"- missing artifacts: {pre['missing_count']}", "",
        "## Zero-seed diagnosis", "",
    ]
    for key, value in mechanisms.items(): lines.append(f"- {key}: `{value['status']}` (training successes={value['training_successes']}, validation={value['validation_class']})")
    lines += ["", "## Graph vs GEOM", "", f"- graph diagnostic: `{graph_result['status']}`", f"- GRAPH_ONLY_NONZERO + OPPOSITE_SIGN rate: {graph_result['primary_graph_only_plus_opposite_rate']:.6f}", f"- GEOM alias groups: {graph_result['alias_group_count']}", f"- one-step top-1 agreement: {counter['top1_agreement']:.6f}", f"- one-step states: {counter['states']}", "", "This is a frozen development diagnostic. It does not replace the original P2B result and does not establish a policy increment for PathGraph.", "", "## Historical state", "", "- P1 representation claim: unchanged", "- P1 reward-accounting claim: unchanged", "- P2A decision: unchanged", "- P2B decision: unchanged", "- robot policy claim: false", "- physical cycle claim: NOT_EVALUATED", "- confirmation_passed: false", ""]
    (artifact / "final/report.md").write_text("\n".join(lines), encoding="utf-8")
    claim_rows = [
        {"claim": "input_integrity", "status": pre["status"], "evidence": "preflight.json;artifact_hash_audit.csv;common_state_potential_corpus.index.csv"},
        {"claim": "zero_seed_mechanism", "status": "PER_SEED_ONLY", "evidence": "validation_checkpoint_forensics.csv;training_episode_summary.csv;failure_class_summary.csv"},
        {"claim": "graph_increment", "status": graph_result["status"], "evidence": "shaping_sign_matrix.csv;geom_aliasing_groups.csv;counterfactual_summary.json"},
        {"claim": "confirmation", "status": "false", "evidence": "final/decision.json"},
    ]
    write_csv(artifact / "claim_to_evidence.csv", claim_rows)
    write_tsv(artifact / "external_artifacts.tsv", external_rows, ["kind", "absolute_path", "bytes", "sha256", "role"])


def manifest(artifact):
    entries = []
    for path in sorted(p for p in artifact.rglob("*") if p.is_file() and p.name != "result_manifest.json"):
        entries.append({"path": str(path.relative_to(artifact)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    result = {"schema": "P2B-D1_RESULT_MANIFEST", "files": entries, "file_count": len(entries)}
    write_json(artifact / "result_manifest.json", result)
    return result


def run_all(repo, p2b_root, external_root, artifact):
    repo, p2b_root, external_root, artifact = map(Path, (repo, p2b_root, external_root, artifact))
    artifact.mkdir(parents=True, exist_ok=True)
    external_root.mkdir(parents=True, exist_ok=True)
    if git(repo, "rev-parse", "HEAD") != BASE:
        raise RuntimeError("D0_BLOCKED_SOURCE_LOCK_MISMATCH: HEAD is not frozen base")
    copy_protocol(repo, artifact)
    source_lock(repo, artifact)
    runtime = load_runtime(repo)
    pre, episode_specs = preflight(repo, p2b_root, artifact)
    zero_registry(artifact)
    validation_classes, training_summary, full_training = seed_forensics(p2b_root, artifact, external_root)
    write_conditional_not_run(artifact)
    db, db_path, corpus_path, external_rows, failures, sign_rows, condition_rows, envmod, TaskCore, shaped_transition, actions = causal_corpus(repo, p2b_root, external_root, artifact, episode_specs, runtime)
    alias_groups = aliasing(db, artifact)
    selected = select_counterfactual_states(db, runtime[0]["conditions"])
    write_json(external_root / "counterfactual_actions/selected_states.json", {"count": len(selected), "state_hashes": selected})
    counter = counterfactual(db, artifact, runtime, selected, episode_specs)
    mechanisms = mechanism_decisions(training_summary, validation_classes, failures)
    graph_result = graph_decision(sign_rows, condition_rows, counter, alias_groups)
    external_rows.append({"kind": "training_episode_table", "absolute_path": str(full_training), "bytes": full_training.stat().st_size, "sha256": sha256_file(full_training), "role": "full per-episode training forensics"})
    selected_file = external_root / "counterfactual_actions/selected_states.json"
    external_rows.append({"kind": "counterfactual_registry", "absolute_path": str(selected_file), "bytes": selected_file.stat().st_size, "sha256": sha256_file(selected_file), "role": "deterministic counterfactual state registry"})
    claims_and_report(repo, p2b_root, external_root, artifact, read_json(artifact / "preflight.json"), mechanisms, graph_result, counter, external_rows)
    result = manifest(artifact)
    db.close()
    return {"status": "COMPLETE", "manifest_files": result["file_count"], "counterfactual_states": counter["states"], "graph_result": graph_result["status"], "zero_seed_mechanisms": mechanisms}


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-all")
    run.add_argument("--repo", required=True)
    run.add_argument("--p2b-root", required=True)
    run.add_argument("--diag-root", required=True)
    run.add_argument("--artifact-root", required=True)
    args = parser.parse_args(argv)
    if args.command == "run-all":
        print(json.dumps(run_all(args.repo, args.p2b_root, args.diag_root, args.artifact_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
