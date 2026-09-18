"""Read-only final-checkpoint evaluation repair for P2C-RL Attempt-03."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import sqlite3
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from p2cq_research.environment import SkillEnv
from p2cq_research.observations import encode_observation
from p2cq_research.task_contract import canonical
from p2crm.mask_contract import legal_mask_bool

from .data_access import load_families
from .evaluate import evaluate_episode
from .learner import assert_versions
from .oracle import action_q, conservative_optimal
from .qualify import load_family_contracts

METHODS = (
    "TASK_ONLY_ZERO_V1",
    "GEOM_COUNT_EVENTS_PBRS_V1",
    "FLAT_CONTRACT_PBRS_V1",
    "PATHGRAPH_MODEL_FULL_PBRS_V1",
    "PATHGRAPH_REMAINING_WORK_PBRS_V2",
)
PRIMARY = "PATHGRAPH_REMAINING_WORK_PBRS_V2"
GEOM = "GEOM_COUNT_EVENTS_PBRS_V1"
FLAT = "FLAT_CONTRACT_PBRS_V1"
SAMPLING_SEEDS = (2026091802, 2026091803, 2026091804, 2026091805)
BOOTSTRAP_SEED = 2026091806
BOOTSTRAP_REPLICATES = 200000
LOWER_QUANTILE = 0.025
UPPER_QUANTILE = 0.975
MARGIN = 0.03
MAIN_EXPECTED = 30720
STOCHASTIC_EXPECTED = 15360
CRITICAL_EXPECTED = 491520
_JOB_CONTEXT = None

MAIN_FIELDS = (
    "panel", "job_id", "method", "draw", "policy_seed", "family_id", "motif",
    "side", "contract_sha256", "sampling_seed", "deterministic", "success",
    "steps", "horizon", "invalid_actions", "nonfinite", "terminated", "truncated",
)
CRITICAL_FIELDS = (
    "panel", "job_id", "method", "draw", "policy_seed", "family_id", "motif",
    "side", "contract_sha256", "prefix_index", "prefix_length", "selected_action",
    "legal_action_count", "oracle_optimal_actions", "oracle_agreement",
)


class EvaluationRepairError(RuntimeError):
    def __init__(self, code, detail=""):
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}" if detail else self.code)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("xb") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_gzip_csv_atomic(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    count = 0
    with tmp.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=list(fields), lineterminator="\n")
                writer.writeheader()
                for row in rows:
                    writer.writerow({field: row.get(field, "") for field in fields})
                    count += 1
    os.replace(tmp, path)
    return count


def read_gzip_csv(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def read_checkpoint_manifest(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 60:
        raise EvaluationRepairError("CHECKPOINT_COUNT", len(rows))
    if len({row["job_id"] for row in rows}) != 60:
        raise EvaluationRepairError("DUPLICATE_CHECKPOINT_JOB")
    for row in rows:
        checkpoint = Path(row["external_path"])
        if checkpoint.name != "policy_524288.zip":
            raise EvaluationRepairError("INTERMEDIATE_CHECKPOINT_FORBIDDEN", checkpoint)
        if not checkpoint.is_file():
            raise EvaluationRepairError("CHECKPOINT_MISSING", checkpoint)
        if int(row["size_bytes"]) != checkpoint.stat().st_size:
            raise EvaluationRepairError("CHECKPOINT_SIZE_CHANGED", row["job_id"])
    _validate_checkpoint_design(rows)
    return rows


def _validate_checkpoint_design(rows):
    seen = defaultdict(set)
    for row in rows:
        draw = row["draw"]
        seed = int(row["policy_seed"])
        method = row["method"]
        if draw not in "ABC" or seed not in (317, 331, 347, 359) or method not in METHODS:
            raise EvaluationRepairError("CHECKPOINT_DESIGN", row["job_id"])
        seen[(draw, seed)].add(method)
    expected_units = {(d, s) for d in "ABC" for s in (317, 331, 347, 359)}
    if set(seen) != expected_units:
        raise EvaluationRepairError("CHECKPOINT_UNITS", len(seen))
    for unit, methods in seen.items():
        if methods != set(METHODS):
            raise EvaluationRepairError("INCOMPLETE_METHOD_PANEL", unit)


def verify_checkpoint_hashes(rows):
    for row in rows:
        observed = sha256_file(row["external_path"])
        if observed != row["sha256"]:
            raise EvaluationRepairError("CHECKPOINT_SHA256_CHANGED", row["job_id"])


def verify_source_database(attempt_root):
    db = Path(attempt_root) / "control" / "campaign.sqlite3"
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        campaign = dict(connection.execute("select * from campaign").fetchone())
        groups = {
            (row["kind"], row["status"]): (row["n"], row["steps"], row["updates"])
            for row in connection.execute(
                "select kind,status,count(*) n,sum(environment_steps) steps,"
                "sum(gradient_updates) updates from jobs group by kind,status"
            )
        }
    finally:
        connection.close()
    if campaign["status"] != "RESUMING":
        raise EvaluationRepairError("SOURCE_DATABASE_STATUS_CHANGED", campaign["status"])
    if groups != {
        ("formal", "COMPLETE"): (60, 31457280, 1228800),
        ("smoke", "COMPLETE"): (5, 2560, 100),
    }:
        raise EvaluationRepairError("SOURCE_DATABASE_COMPLETION_CHANGED", groups)
    return campaign


def load_test_cases(data_root):
    families = [item for item in load_families(data_root) if item["split"] == "test"]
    families.sort(key=lambda item: (item["motif"], int(item["root"]), int(item["family_id"])))
    if len(families) != 256:
        raise EvaluationRepairError("TEST_FAMILY_COUNT", len(families))
    cases = []
    for family in families:
        left, right = load_family_contracts(Path(data_root) / "dataset", family)
        for side, contract in (("left", left), ("right", right)):
            digest = hashlib.sha256(canonical(contract.public_dict()).encode("utf-8")).hexdigest()
            cases.append({
                "family": family,
                "family_id": int(family["family_id"]),
                "motif": family["motif"],
                "side": side,
                "contract": contract,
                "contract_sha256": digest,
            })
    if len(cases) != 512:
        raise EvaluationRepairError("TEST_CONTRACT_COUNT", len(cases))
    return cases


def select_stochastic_cases(cases):
    by_motif = defaultdict(list)
    for case in cases:
        by_motif[case["motif"]].append(case)
    selected = []
    for motif in sorted(by_motif):
        family_ids = []
        for case in by_motif[motif]:
            if case["family_id"] not in family_ids:
                family_ids.append(case["family_id"])
        wanted = set(family_ids[:8])
        selected.extend(case for case in by_motif[motif] if case["family_id"] in wanted)
    selected.sort(key=lambda case: (case["motif"], case["family_id"], case["side"]))
    if len(selected) != 64:
        raise EvaluationRepairError("STOCHASTIC_CONTRACT_COUNT", len(selected))
    return selected


def _forbid_training(model):
    def forbidden(*_args, **_kwargs):
        raise EvaluationRepairError("TRAINING_CALL_FORBIDDEN")
    model.learn = forbidden
    optimizer = getattr(getattr(model, "policy", None), "optimizer", None)
    if optimizer is not None:
        optimizer.step = forbidden
    model.policy.set_training_mode(False)
    for parameter in model.policy.parameters():
        parameter.requires_grad_(False)
    return model


def load_final_policy(checkpoint):
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(str(checkpoint), device="cpu")
    if int(model.num_timesteps) != 524288:
        raise EvaluationRepairError("LOADED_CHECKPOINT_STEP", model.num_timesteps)
    return _forbid_training(model)


def _episode_row(panel, checkpoint, case, record, sampling_seed, deterministic):
    return {
        "panel": panel,
        "job_id": checkpoint["job_id"],
        "method": checkpoint["method"],
        "draw": checkpoint["draw"],
        "policy_seed": int(checkpoint["policy_seed"]),
        "family_id": case["family_id"],
        "motif": case["motif"],
        "side": case["side"],
        "contract_sha256": record["contract_sha256"],
        "sampling_seed": "" if sampling_seed is None else int(sampling_seed),
        "deterministic": 1 if deterministic else 0,
        "success": int(record["success"]),
        "steps": int(record["steps"]),
        "horizon": int(record["horizon"]),
        "invalid_actions": int(record["invalid_actions"]),
        "nonfinite": int(record["nonfinite"]),
        "terminated": int(record["terminated"]),
        "truncated": int(record["truncated"]),
    }


def evaluate_main(model, checkpoint, cases):
    for case in cases:
        record = evaluate_episode(model, case["contract"], deterministic=True)
        yield _episode_row("main", checkpoint, case, record, None, True)


def evaluate_stochastic(model, checkpoint, cases):
    for sampling_seed in SAMPLING_SEEDS:
        for case in cases:
            record = evaluate_episode(
                model,
                case["contract"],
                deterministic=False,
                sampling_seed=sampling_seed,
            )
            yield _episode_row(
                "stochastic", checkpoint, case, record, sampling_seed, False
            )


def _run_registered_prefix(contract, prefix):
    env = SkillEnv(contract)
    env.reset()
    for position, action in enumerate(prefix):
        if env.terminated():
            raise EvaluationRepairError("CRITICAL_PREFIX_TERMINATED_EARLY", position)
        action = int(action)
        if action < 0 or action >= 37:
            raise EvaluationRepairError("CRITICAL_PREFIX_ACTION_RANGE", action)
        env.step(action)
    if env.terminated():
        raise EvaluationRepairError("CRITICAL_PREFIX_TERMINAL_STATE")
    return env


def _build_critical_case(case):
    observations = []
    masks = []
    optimal_masks = []
    oracle_cache = {}
    prefixes = case["family"].get("prefixes") or []
    if len(prefixes) != 16:
        raise EvaluationRepairError("CRITICAL_PREFIX_COUNT", case["family_id"])
    for prefix_index, prefix in enumerate(prefixes):
        env = _run_registered_prefix(case["contract"], prefix)
        observation = np.asarray(
            encode_observation(case["contract"], env), dtype=np.float32
        )
        mask = np.asarray(legal_mask_bool(env), dtype=bool)
        if not np.isfinite(observation).all():
            raise EvaluationRepairError("CRITICAL_NONFINITE_OBSERVATION")
        qs, oracle_mask = action_q(env, max_states=200000, cache=oracle_cache)
        optimal = conservative_optimal(qs)
        if not optimal:
            raise EvaluationRepairError(
                "CRITICAL_ORACLE_EMPTY",
                f"{case['family_id']}:{case['side']}:{prefix_index}",
            )
        if not np.array_equal(mask, np.asarray(oracle_mask, dtype=bool)):
            raise EvaluationRepairError("CRITICAL_ORACLE_MASK_MISMATCH")
        optimal_mask = np.zeros(len(mask), dtype=bool)
        optimal_mask[np.asarray(optimal, dtype=int)] = True
        observations.append(observation)
        masks.append(mask)
        optimal_masks.append(optimal_mask)
    return {
        "observation": observations,
        "action_mask": masks,
        "oracle_optimal_mask": optimal_masks,
        "family_id": [case["family_id"]] * 16,
        "motif": [case["motif"]] * 16,
        "side": [case["side"]] * 16,
        "contract_sha256": [case["contract_sha256"]] * 16,
        "prefix_index": list(range(16)),
        "prefix_length": [len(prefix) for prefix in prefixes],
    }


def build_critical_cache(cases, cache_path):
    cache_path = Path(cache_path)
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        if int(data["observation"].shape[0]) != 8192:
            raise EvaluationRepairError("CRITICAL_CACHE_ROWS", data["observation"].shape[0])
        return data
    collected = {
        "observation": [],
        "action_mask": [],
        "oracle_optimal_mask": [],
        "family_id": [],
        "motif": [],
        "side": [],
        "contract_sha256": [],
        "prefix_index": [],
        "prefix_length": [],
    }
    workers = int(os.environ.get("P2CRL_ORACLE_WORKERS", "1"))
    if workers < 1 or workers > 32:
        raise EvaluationRepairError("CRITICAL_ORACLE_WORKERS", workers)
    if workers == 1:
        results = map(_build_critical_case, cases)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(_build_critical_case, cases)
    try:
        for case_index, result in enumerate(results):
            for field in collected:
                collected[field].extend(result[field])
            if (case_index + 1) % 32 == 0:
                print(f"CRITICAL_CACHE {case_index + 1}/{len(cases)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown()
    observations = collected["observation"]
    masks = collected["action_mask"]
    optimal_masks = collected["oracle_optimal_mask"]
    family_ids = collected["family_id"]
    motifs = collected["motif"]
    sides = collected["side"]
    contract_hashes = collected["contract_sha256"]
    prefix_indices = collected["prefix_index"]
    prefix_lengths = collected["prefix_length"]
    arrays = {
        "observation": np.asarray(observations, dtype=np.float32),
        "action_mask": np.asarray(masks, dtype=bool),
        "oracle_optimal_mask": np.asarray(optimal_masks, dtype=bool),
        "family_id": np.asarray(family_ids, dtype=np.int64),
        "motif": np.asarray(motifs, dtype="U32"),
        "side": np.asarray(sides, dtype="U5"),
        "contract_sha256": np.asarray(contract_hashes, dtype="U64"),
        "prefix_index": np.asarray(prefix_indices, dtype=np.int16),
        "prefix_length": np.asarray(prefix_lengths, dtype=np.int16),
    }
    if arrays["observation"].shape[0] != 8192:
        raise EvaluationRepairError("CRITICAL_CACHE_ROWS", arrays["observation"].shape[0])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(f".{cache_path.name}.tmp.{os.getpid()}")
    with tmp.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, cache_path)
    return np.load(cache_path, allow_pickle=False)


def evaluate_critical(model, checkpoint, cache, batch_size=512):
    observations = cache["observation"]
    masks = cache["action_mask"]
    optimal_masks = cache["oracle_optimal_mask"]
    total = observations.shape[0]
    for start in range(0, total, int(batch_size)):
        stop = min(total, start + int(batch_size))
        actions, _ = model.predict(
            observations[start:stop],
            deterministic=True,
            action_masks=masks[start:stop],
        )
        actions = np.asarray(actions, dtype=int).reshape(-1)
        for offset, action in enumerate(actions):
            index = start + offset
            if not bool(masks[index, action]):
                raise EvaluationRepairError("CRITICAL_ILLEGAL_ACTION", index)
            optimal = np.flatnonzero(optimal_masks[index])
            yield {
                "panel": "critical",
                "job_id": checkpoint["job_id"],
                "method": checkpoint["method"],
                "draw": checkpoint["draw"],
                "policy_seed": int(checkpoint["policy_seed"]),
                "family_id": int(cache["family_id"][index]),
                "motif": str(cache["motif"][index]),
                "side": str(cache["side"][index]),
                "contract_sha256": str(cache["contract_sha256"][index]),
                "prefix_index": int(cache["prefix_index"][index]),
                "prefix_length": int(cache["prefix_length"][index]),
                "selected_action": int(action),
                "legal_action_count": int(np.count_nonzero(masks[index])),
                "oracle_optimal_actions": "|".join(str(int(value)) for value in optimal),
                "oracle_agreement": 1 if bool(optimal_masks[index, action]) else 0,
            }


def _job_paths(output_root, job_id):
    root = Path(output_root) / "jobs" / job_id
    return {
        "root": root,
        "main": root / "main.csv.gz",
        "stochastic": root / "stochastic.csv.gz",
        "critical": root / "critical.csv.gz",
        "receipt": root / "receipt.json",
    }


def _existing_job_valid(paths, checkpoint):
    if not paths["receipt"].is_file():
        return False
    receipt = load_json(paths["receipt"])
    expected = {"main": 512, "stochastic": 256, "critical": 8192}
    if (
        receipt.get("job_id") != checkpoint["job_id"]
        or receipt.get("checkpoint_sha256") != checkpoint["sha256"]
        or receipt.get("checkpoint_step") != 524288
        or receipt.get("training_calls") != 0
        or receipt.get("optimizer_steps") != 0
        or receipt.get("rows") != expected
    ):
        return False
    for name in ("main", "stochastic", "critical"):
        path = paths[name]
        if (
            not path.is_file()
            or sha256_file(path) != receipt.get("sha256", {}).get(name)
        ):
            return False
    return True


def evaluate_job(checkpoint, cases, stochastic_cases, critical_cache, output_root):
    paths = _job_paths(output_root, checkpoint["job_id"])
    if _existing_job_valid(paths, checkpoint):
        print(f"EVALUATION_SKIP_COMPLETE {checkpoint['job_id']}", flush=True)
        return load_json(paths["receipt"])
    if paths["root"].exists() and any(paths["root"].iterdir()):
        raise EvaluationRepairError("PARTIAL_JOB_OUTPUT", checkpoint["job_id"])
    model = load_final_policy(checkpoint["external_path"])
    main_count = write_gzip_csv_atomic(
        paths["main"], MAIN_FIELDS, evaluate_main(model, checkpoint, cases)
    )
    stochastic_count = write_gzip_csv_atomic(
        paths["stochastic"],
        MAIN_FIELDS,
        evaluate_stochastic(model, checkpoint, stochastic_cases),
    )
    critical_count = write_gzip_csv_atomic(
        paths["critical"],
        CRITICAL_FIELDS,
        evaluate_critical(model, checkpoint, critical_cache),
    )
    if (main_count, stochastic_count, critical_count) != (512, 256, 8192):
        raise EvaluationRepairError(
            "JOB_ROW_COUNT", (main_count, stochastic_count, critical_count)
        )
    receipt = {
        "schema": "P2CRL_EVALUATION_REPAIR_JOB_RECEIPT_V1",
        "job_id": checkpoint["job_id"],
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_step": 524288,
        "training_calls": 0,
        "optimizer_steps": 0,
        "rows": {
            "main": main_count,
            "stochastic": stochastic_count,
            "critical": critical_count,
        },
        "sha256": {
            name: sha256_file(paths[name])
            for name in ("main", "stochastic", "critical")
        },
    }
    write_json_atomic(paths["receipt"], receipt)
    print(f"EVALUATION_COMPLETE {checkpoint['job_id']}", flush=True)
    return receipt


def _evaluate_job_worker(checkpoint):
    if _JOB_CONTEXT is None:
        raise EvaluationRepairError("JOB_WORKER_CONTEXT_MISSING")
    cases, stochastic_cases, critical_cache, output_root = _JOB_CONTEXT
    return evaluate_job(
        checkpoint, cases, stochastic_cases, critical_cache, output_root
    )


def _records_from_jobs(output_root, checkpoint_rows, name):
    records = []
    for checkpoint in checkpoint_rows:
        path = _job_paths(output_root, checkpoint["job_id"])[name]
        records.extend(read_gzip_csv(path))
    return records


def _require_nonempty(records):
    if not records:
        raise EvaluationRepairError("EMPTY_RECORDS")


def _require_unique(records, fields):
    seen = set()
    for row in records:
        key = tuple(row[field] for field in fields)
        if key in seen:
            raise EvaluationRepairError("DUPLICATE_EPISODE_KEY", key)
        seen.add(key)
    return seen


def validate_main_records(records, checkpoint_rows, cases):
    _require_nonempty(records)
    keys = _require_unique(records, ("job_id", "family_id", "side"))
    expected = {
        (checkpoint["job_id"], str(case["family_id"]), case["side"])
        for checkpoint in checkpoint_rows
        for case in cases
    }
    if keys != expected:
        raise EvaluationRepairError(
            "MISSING_ROWS", f"expected={len(expected)} observed={len(keys)}"
        )
    methods = defaultdict(set)
    for row in records:
        methods[(
            row["draw"], row["policy_seed"], row["family_id"], row["side"]
        )].add(row["method"])
        if (
            int(row["invalid_actions"]) != 0
            or int(row["nonfinite"]) != 0
            or int(row["terminated"]) != 1
            or int(row["truncated"]) != 0
        ):
            raise EvaluationRepairError("INVALID_MAIN_EPISODE", row["job_id"])
    for unit, observed in methods.items():
        if observed != set(METHODS):
            raise EvaluationRepairError("INCOMPLETE_METHOD_PANEL", unit)
    if len(records) != MAIN_EXPECTED:
        raise EvaluationRepairError("MISSING_ROWS", len(records))


def validate_stochastic_records(records, checkpoint_rows, cases):
    _require_nonempty(records)
    keys = _require_unique(
        records, ("job_id", "family_id", "side", "sampling_seed")
    )
    expected = {
        (
            checkpoint["job_id"],
            str(case["family_id"]),
            case["side"],
            str(sampling_seed),
        )
        for checkpoint in checkpoint_rows
        for case in cases
        for sampling_seed in SAMPLING_SEEDS
    }
    if keys != expected or len(records) != STOCHASTIC_EXPECTED:
        raise EvaluationRepairError(
            "MISSING_ROWS", f"expected={len(expected)} observed={len(keys)}"
        )


def validate_critical_records(records, checkpoint_rows, cases):
    _require_nonempty(records)
    keys = _require_unique(
        records, ("job_id", "family_id", "side", "prefix_index")
    )
    expected = {
        (
            checkpoint["job_id"],
            str(case["family_id"]),
            case["side"],
            str(prefix_index),
        )
        for checkpoint in checkpoint_rows
        for case in cases
        for prefix_index in range(16)
    }
    if keys != expected or len(records) != CRITICAL_EXPECTED:
        raise EvaluationRepairError(
            "MISSING_ROWS", f"expected={len(expected)} observed={len(keys)}"
        )
    methods = defaultdict(set)
    for row in records:
        methods[(
            row["draw"], row["policy_seed"], row["family_id"],
            row["side"], row["prefix_index"],
        )].add(row["method"])
    for unit, observed in methods.items():
        if observed != set(METHODS):
            raise EvaluationRepairError("INCOMPLETE_METHOD_PANEL", unit)


def _balanced_delta_array(records, comparator):
    lookup = {}
    draws = tuple("ABC")
    seeds = (317, 331, 347, 359)
    motifs = tuple(sorted({row["motif"] for row in records}))
    families = {
        motif: tuple(sorted({
            int(row["family_id"]) for row in records if row["motif"] == motif
        }))
        for motif in motifs
    }
    if len(motifs) != 4 or any(len(values) != 64 for values in families.values()):
        raise EvaluationRepairError("INCOMPLETE_METHOD_PANEL", "motif/family design")
    for row in records:
        method = row["method"]
        if method not in (PRIMARY, comparator):
            continue
        key = (
            method, row["draw"], int(row["policy_seed"]), row["motif"],
            int(row["family_id"]), row["side"],
        )
        if key in lookup:
            raise EvaluationRepairError("DUPLICATE_EPISODE_KEY", key)
        lookup[key] = float(row["success"])
    delta = np.empty((3, 4, 4, 64), dtype=np.float64)
    for di, draw in enumerate(draws):
        for si, seed in enumerate(seeds):
            for mi, motif in enumerate(motifs):
                for fi, family_id in enumerate(families[motif]):
                    values = []
                    for side in ("left", "right"):
                        akey = (PRIMARY, draw, seed, motif, family_id, side)
                        bkey = (comparator, draw, seed, motif, family_id, side)
                        if akey not in lookup or bkey not in lookup:
                            raise EvaluationRepairError(
                                "INCOMPLETE_METHOD_PANEL",
                                (draw, seed, motif, family_id, side),
                            )
                        values.append(lookup[akey] - lookup[bkey])
                    delta[di, si, mi, fi] = float(np.mean(values))
    return delta


def _bootstrap_two_deltas(delta_geom, delta_flat):
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    out_geom = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    out_flat = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    denominator = float(3 * 4 * 4 * 64)
    chunk_size = 1000
    for start in range(0, BOOTSTRAP_REPLICATES, chunk_size):
        stop = min(BOOTSTRAP_REPLICATES, start + chunk_size)
        count = stop - start
        draw_weight = np.zeros((count, 3), dtype=np.uint8)
        seed_weight = np.zeros((count, 3, 4), dtype=np.uint8)
        family_weight = np.zeros((count, 4, 64), dtype=np.uint8)
        for index in range(count):
            draw_weight[index] = np.bincount(
                rng.randint(0, 3, size=3), minlength=3
            )
            for draw in range(3):
                seed_weight[index, draw] = np.bincount(
                    rng.randint(0, 4, size=4), minlength=4
                )
            for motif in range(4):
                family_weight[index, motif] = np.bincount(
                    rng.randint(0, 64, size=64), minlength=64
                )
        out_geom[start:stop] = np.einsum(
            "bd,bds,bmf,dsmf->b",
            draw_weight,
            seed_weight,
            family_weight,
            delta_geom,
            optimize=True,
        ) / denominator
        out_flat[start:stop] = np.einsum(
            "bd,bds,bmf,dsmf->b",
            draw_weight,
            seed_weight,
            family_weight,
            delta_flat,
            optimize=True,
        ) / denominator
        if stop % 10000 == 0:
            print(f"BOOTSTRAP {stop}/{BOOTSTRAP_REPLICATES}", flush=True)
    return out_geom, out_flat


def strict_confirmatory_statistics(records):
    _require_nonempty(records)
    delta_geom = _balanced_delta_array(records, GEOM)
    delta_flat = _balanced_delta_array(records, FLAT)
    boot_geom, boot_flat = _bootstrap_two_deltas(delta_geom, delta_flat)

    def result(delta, boot):
        point = float(np.mean(delta))
        lower = float(np.quantile(boot, LOWER_QUANTILE))
        upper = float(np.quantile(boot, UPPER_QUANTILE))
        return {
            "point": point,
            "ci_low": lower,
            "ci_high": upper,
            "lower_quantile": LOWER_QUANTILE,
            "upper_quantile": UPPER_QUANTILE,
            "n_boot": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "margin_ok": point >= MARGIN,
            "ci_ok": lower > 0.0,
        }

    geom = result(delta_geom, boot_geom)
    flat = result(delta_flat, boot_flat)
    passed = bool(
        geom["margin_ok"] and geom["ci_ok"]
        and flat["margin_ok"] and flat["ci_ok"]
    )
    return {
        "schema": "P2CRL_PRIMARY_COMPARISONS_REPAIRED_V1",
        "input_status": "COMPLETE_ACTUAL_EPISODE_RECORDS",
        "legacy_effect_disposition": "INVALID_EMPTY_RECORD_INPUT",
        "status": (
            "RW_V2_POLICY_UTILITY_SUPPORTED_IN_REGISTERED_ABSTRACT_DOMAIN"
            if passed else "POLICY_UTILITY_NOT_ESTABLISHED_OR_MIXED"
        ),
        "passed": passed,
        "RW_V2_minus_GEOM": geom,
        "RW_V2_minus_FLAT": flat,
        "global_confirmation_passed": False,
    }


def _combine_job_files(output_root, checkpoint_rows, name, fields, destination):
    def rows():
        for checkpoint in checkpoint_rows:
            yield from read_gzip_csv(_job_paths(output_root, checkpoint["job_id"])[name])
    return write_gzip_csv_atomic(destination, fields, rows())


def finalize(output_root, checkpoint_rows, cases, stochastic_cases):
    output_root = Path(output_root)
    for checkpoint in checkpoint_rows:
        paths = _job_paths(output_root, checkpoint["job_id"])
        if not _existing_job_valid(paths, checkpoint):
            raise EvaluationRepairError("MISSING_JOB_RECEIPT", checkpoint["job_id"])
    main_records = _records_from_jobs(output_root, checkpoint_rows, "main")
    stochastic_records = _records_from_jobs(
        output_root, checkpoint_rows, "stochastic"
    )
    critical_records = _records_from_jobs(output_root, checkpoint_rows, "critical")
    validate_main_records(main_records, checkpoint_rows, cases)
    validate_stochastic_records(
        stochastic_records, checkpoint_rows, stochastic_cases
    )
    validate_critical_records(critical_records, checkpoint_rows, cases)
    statistics = strict_confirmatory_statistics(main_records)

    final_root = output_root / "final"
    final_root.mkdir(parents=True, exist_ok=True)
    counts = {
        "main": _combine_job_files(
            output_root, checkpoint_rows, "main", MAIN_FIELDS,
            final_root / "main_records.csv.gz",
        ),
        "stochastic": _combine_job_files(
            output_root, checkpoint_rows, "stochastic", MAIN_FIELDS,
            final_root / "stochastic_records.csv.gz",
        ),
        "critical": _combine_job_files(
            output_root, checkpoint_rows, "critical", CRITICAL_FIELDS,
            final_root / "critical_records.csv.gz",
        ),
    }
    if counts != {
        "main": MAIN_EXPECTED,
        "stochastic": STOCHASTIC_EXPECTED,
        "critical": CRITICAL_EXPECTED,
    }:
        raise EvaluationRepairError("FINAL_ROW_COUNT", counts)
    write_json_atomic(final_root / "primary_comparisons.json", statistics)
    critical_agreement = (
        sum(int(row["oracle_agreement"]) for row in critical_records)
        / len(critical_records)
    )
    summaries = {
        "main": {
            "schema": "P2CRL_MAIN_TEST_PANEL_REPAIRED_V1",
            "rows": counts["main"],
            "success_mean": sum(int(row["success"]) for row in main_records) / counts["main"],
            "invalid_actions": sum(int(row["invalid_actions"]) for row in main_records),
            "nonfinite": sum(int(row["nonfinite"]) for row in main_records),
        },
        "stochastic": {
            "schema": "P2CRL_STOCHASTIC_PANEL_REPAIRED_V1",
            "rows": counts["stochastic"],
            "success_mean": (
                sum(int(row["success"]) for row in stochastic_records)
                / counts["stochastic"]
            ),
            "sampling_seeds": list(SAMPLING_SEEDS),
        },
        "critical": {
            "schema": "P2CRL_CRITICAL_STATE_PANEL_REPAIRED_V1",
            "rows": counts["critical"],
            "oracle_agreement_mean": critical_agreement,
            "policy_inputs": ["observation", "action_mask"],
            "oracle_process": "analysis_only",
        },
    }
    for name, summary in summaries.items():
        write_json_atomic(final_root / f"{name}_summary.json", summary)
    manifest = {
        "schema": "P2CRL_EVALUATION_REPAIR_OUTPUT_MANIFEST_V1",
        "rows": counts,
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(final_root.iterdir())
            if path.is_file()
        },
        "training_calls": 0,
        "optimizer_steps": 0,
        "source_database_mutated": False,
    }
    write_json_atomic(final_root / "output_manifest.json", manifest)
    return statistics


def run(args):
    assert_versions()
    verify_source_database(args.attempt_root)
    checkpoints = read_checkpoint_manifest(args.checkpoint_manifest)
    verify_checkpoint_hashes(checkpoints)
    cases = load_test_cases(args.data_root)
    stochastic_cases = select_stochastic_cases(cases)
    output_root = Path(args.output_root)
    cache = build_critical_cache(cases, output_root / "critical_state_cache.npz")
    selected = checkpoints
    if args.job_id:
        selected = [row for row in checkpoints if row["job_id"] == args.job_id]
        if len(selected) != 1:
            raise EvaluationRepairError("UNKNOWN_JOB", args.job_id)
    workers = 1 if args.job_id else int(os.environ.get("P2CRL_JOB_WORKERS", "1"))
    if workers < 1 or workers > 16:
        raise EvaluationRepairError("EVALUATION_JOB_WORKERS", workers)
    if workers == 1:
        for index, checkpoint in enumerate(selected, start=1):
            evaluate_job(checkpoint, cases, stochastic_cases, cache, output_root)
            print(f"EVALUATION_PROGRESS {index}/{len(selected)}", flush=True)
    else:
        global _JOB_CONTEXT
        _JOB_CONTEXT = (cases, stochastic_cases, cache, output_root)
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                receipts = executor.map(_evaluate_job_worker, selected)
                for index, _receipt in enumerate(receipts, start=1):
                    print(
                        f"EVALUATION_PROGRESS {index}/{len(selected)}",
                        flush=True,
                    )
        finally:
            _JOB_CONTEXT = None
    if not args.job_id:
        statistics = finalize(
            output_root, checkpoints, cases, stochastic_cases
        )
        verify_checkpoint_hashes(checkpoints)
        print(
            "EVALUATION_REPAIR_OK "
            + json.dumps(statistics, sort_keys=True, separators=(",", ":")),
            flush=True,
        )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--job-id")
    return parser


def main(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
