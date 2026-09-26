"""CP-DISR v1.2 Phase A first-evidence runner.

This module is intentionally a narrow adapter over the Method 2.1.1 production
Stage 2A Policy/Collector/PPO/runtime.  It authorizes only T_C Full/B2 seed 0,
binds the E16 caps, freezes a common dev10 before evaluation, and upgrades each
checkpoint publication to the validated immutable-generation protocol.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from .common import BindingError, DataIntegrityError, canonical
from .persistence import GenerationStore
from . import stage2a_v11 as v11

PLAN_VERSION = "1.2"
SPEC_VERSION = "3.1"
METHOD_VERSION = "2.1.1"
RUNTIME_REVISION = "clock_integrity_r2"
N_CAP = 16384
ROLLOUT_N = 1024
MAX_UPDATES = 16
EVAL_POINTS = (0, 4096, 8192, 16384)
DEV10_N = 10
AUTHORIZED = {
    "Full": "v12_E16_T_C_Full_s0",
    "B2": "v12_E16_T_C_B2_s0",
}
BASE_COMMIT = "f8719ff62072ab594eb71e76156690966927c973"
STAGE_DIR = Path("runs/phase_a")
STATUS_PATH = Path("status/phase_a.json")
REPORT_PATH = Path("reports/phase_a_summary.md")
DEV10_REL = Path("configs/splits/T_C_phase_a_v12_dev10.json")
FINAL_GATE_GLOB = "runs/phase_a_final_unblock/*/final_unblock_status.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(value) + "\n", encoding="utf-8")


def git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _latest_final_gate(root: Path) -> tuple[Path, dict]:
    paths = sorted(root.glob(FINAL_GATE_GLOB), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise BindingError("missing Phase A final-unblock status")
    path = paths[-1]
    doc = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "phase_a_e16_ready": True,
        "phase_a_status": "NOT_STARTED",
        "semantic_determinism": True,
        "classification": "RAW_RGB_LSB_VARIANCE_NO_DOWNSTREAM_EFFECT",
    }
    bad = {k: (doc.get(k), want) for k, want in required.items() if doc.get(k) != want}
    if bad:
        raise BindingError("Phase A final-unblock gate mismatch: %s" % bad)
    return path, doc


def _freeze_dev10(root: Path) -> tuple[Path, dict]:
    source = root / v11.ENABLED_SPLITS["T_C"]
    split = json.loads(source.read_text(encoding="utf-8"))
    if len(split.get("train") or []) != 64 or len(split.get("dev") or []) != 20:
        raise BindingError("T_C split must be train64/dev20")
    # Canonical pre-result selection: rank only by a namespaced SHA-256 of case ID.
    ranked = sorted(
        split["dev"],
        key=lambda row: hashlib.sha256(
            ("cp_disr_v1.2_phase_a_dev10\0" + row["case_id"]).encode("utf-8")
        ).hexdigest(),
    )
    selected_ids = {row["case_id"] for row in ranked[:DEV10_N]}
    selected = [row for row in split["dev"] if row["case_id"] in selected_ids]
    derived = dict(split)
    derived["dev"] = selected
    derived["dev_count"] = len(selected)
    derived["test"] = []
    derived["test_count"] = 0
    derived["phase_a_v12"] = {
        "selection_namespace": "cp_disr_v1.2_phase_a_dev10",
        "selection_rule": "sha256_rank_then_preserve_source_order",
        "source_split": str(source.relative_to(root)).replace("\\", "/"),
        "source_sha256": sha256_file(source),
        "test_ids_accessed": False,
    }
    path = root / DEV10_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(derived, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "frozen_before_any_formal_result": True,
        "created_at": utc_now(),
        "source_split": str(source.relative_to(root)).replace("\\", "/"),
        "source_sha256": sha256_file(source),
        "derived_split": str(path.relative_to(root)).replace("\\", "/"),
        "derived_sha256": sha256_file(path),
        "dev20_order": [r["case_id"] for r in split["dev"]],
        "dev10_order": [r["case_id"] for r in selected],
        "test_ids_accessed": False,
    }
    return path, manifest


def _configure_v11() -> None:
    v11.STAGE_DIR = STAGE_DIR
    v11.N_CAP = N_CAP
    v11.ROLLOUT_N = ROLLOUT_N
    v11.MAX_UPDATES = MAX_UPDATES
    v11.DEV_EPISODES = DEV10_N
    v11.TASKS = ("T_C",)
    v11.METHODS = ("B2", "Full")
    v11.PLANNED = {("T_C", method): pid for method, pid in AUTHORIZED.items()}
    v11.ENABLED_SPLITS = dict(v11.ENABLED_SPLITS)
    v11.ENABLED_SPLITS["T_C"] = DEV10_REL


def _phase_config(prof: dict, source_commit: str) -> dict:
    d_ref = float(prof["d_ref"]["T_C"])
    return {
        "plan_version": PLAN_VERSION,
        "research_specification": SPEC_VERSION,
        "method_profile": METHOD_VERSION,
        "runtime_revision": RUNTIME_REVISION,
        "training_source_commit": source_commit,
        "task": "T_C",
        "seed": 0,
        "authorized_planned_ids": list(AUTHORIZED.values()),
        "Ncap": N_CAP,
        "Tcap": N_CAP * d_ref,
        "d_ref": d_ref,
        "max_complete_updates": MAX_UPDATES,
        "rollout": ROLLOUT_N,
        "ppo_epochs": 4,
        "sequence_length": 16,
        "minibatch_target": 64,
        "actor_episode_discount_weight": False,
        "gamma_rule": "2**(-duration/H)",
        "eval_points_N": list(EVAL_POINTS),
        "dev10_n": DEV10_N,
        "full_prior": "episode_80pct_original_20pct_absent",
        "full_eval_prior": "original",
        "b2_prior": "always_absent",
        "formal_run_count": 2,
        "test_id_used": False,
        "vlm_requests": 0,
    }


def _source_hashes(root: Path) -> dict:
    files = {
        "phase_a_v12": root / "src/cp_disr/phase_a_v12.py",
        "stage2a_v11": root / "src/cp_disr/stage2a_v11.py",
        "collector": root / "src/cp_disr/collector.py",
        "torch_rl": root / "src/cp_disr/torch_rl.py",
        "neural": root / "src/cp_disr/neural.py",
        "persistence": root / "src/cp_disr/persistence.py",
        "runtime_factory": root / "src/cp_disr/platforms/libero/runtime_factory.py",
        "runtime_manifest": root / v11.RUNTIME_REL,
        "split_dev10": root / DEV10_REL,
    }
    return {
        k: sha256_file(p) if p.is_file() else None for k, p in files.items()
    } | {"git_commit": git_commit(root), "base_commit": BASE_COMMIT}


def _capture_episode(collector) -> bytes:
    bundle = collector.bundle
    sim = bundle.environment.sim
    payload = {
        "current_snapshot": bundle.current_snapshot,
        "qpos": sim.data.qpos.copy(),
        "qvel": sim.data.qvel.copy(),
        "ctrl": sim.data.ctrl.copy(),
        "sim_time": float(sim.data.time),
        "clock_origin": float(bundle.clock._origin),
        "episode_start_seconds": float(bundle.episode_start_seconds),
        "case_scheduler_n": int(bundle._n),
        "snapshot_episode_id": bundle.snapshot_builder.episode_id,
        "evaluator_rewarded": bool(bundle.evaluator._rewarded),
        "evaluator_success_time": bundle.evaluator._success_time,
        "evaluator_deadline": float(bundle.evaluator.deadline),
        "original_prior_edges": bundle.original_prior_edges,
        "prefixes": collector.prefixes,
        "weights": collector.weights,
        "success_seen": collector.success_seen,
        "case_id": getattr(bundle, "_phase_a_case_id", None),
        "prior": getattr(bundle, "_phase_a_prior", None),
        "source_n": getattr(bundle, "_phase_a_source_n", None),
        "cache_key": getattr(bundle, "_phase_a_cache_key", None),
    }
    return pickle.dumps(payload, protocol=4)


_ORIGINAL_SAVE_CKPT = v11.save_ckpt
_ORIGINAL_START_EPISODE = v11.start_episode
_COLLECTOR_PATCHED = False


def _select_checkpoint(eval_rows):
    usable = [r for r in eval_rows if r.get("mean_discounted_return") is not None]
    if not usable:
        return None, {"reason": "no eval windows"}
    annotated = []
    for i, row in enumerate(usable):
        window = usable[max(0, i - 2):i + 1]
        mean3 = float(sum(float(w["mean_discounted_return"]) for w in window) / len(window))
        worst = float(min(float(w["mean_discounted_return"]) for w in window))
        annotated.append({
            "index": i,
            "n_used": len(window),
            "mean3": mean3,
            "worst": worst,
            "transitions": int(row.get("skill_transitions") or 0),
            "checkpoint": row.get("checkpoint"),
            "row": row,
        })
    # Maximize the recent-window mean, then maximize its worst member, then
    # prefer the earlier evidence point.
    annotated.sort(key=lambda r: (-r["mean3"], -r["worst"], r["transitions"]))
    best = annotated[0]
    return best["checkpoint"], {
        "selection_rule": "max_mean3_then_max_worst_then_earlier_N",
        "ranked": annotated,
        "selected": best,
        "fewer_than_3_windows": best["n_used"] < 3,
    }


def _start_episode_with_state(*args, **kwargs):
    started = time.perf_counter()
    result = _ORIGINAL_START_EPISODE(*args, **kwargs)
    bundle = args[3]
    case, prior, source_n, cache_key = result
    bundle._phase_a_case_id = case
    bundle._phase_a_prior = prior
    bundle._phase_a_source_n = source_n
    bundle._phase_a_cache_key = cache_key
    collector = args[7]
    profile = getattr(collector, "_phase_a_profile", None)
    if profile is None:
        profile = {"reset_seconds": 0.0, "decision_attempts": 0}
        collector._phase_a_profile = profile
        collector._phase_a_train_wall_start = started
    profile["reset_seconds"] += time.perf_counter() - started
    return result


def _timed_method(owner, name, profile, key):
    original = getattr(owner, name)
    def wrapped(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            profile[key] = profile.get(key, 0.0) + (time.perf_counter() - started)
    setattr(owner, name, wrapped)
    return original


def _install_collector_profiler() -> None:
    global _COLLECTOR_PATCHED
    if _COLLECTOR_PATCHED:
        return
    from .collector import Collector
    original_step = Collector.step
    def profiled_step(self, snapshot, deterministic=False):
        if deterministic or os.environ.get("CP_DISR_PHASE_A_PROFILE") != "1":
            return original_step(self, snapshot, deterministic=deterministic)
        profile = getattr(self, "_phase_a_profile", None)
        if profile is None:
            profile = {"reset_seconds": 0.0, "decision_attempts": 0}
            self._phase_a_profile = profile
            self._phase_a_train_wall_start = time.perf_counter()
        profile["decision_attempts"] += 1
        restores = []
        restores.append((self.policy, "forward", _timed_method(self.policy, "forward", profile, "policy_graph_four_views_gru_seconds")))
        for owner, name, key in (
            (self.bundle.observations, "observe", "rgb_depth_seconds"),
            (self.bundle.executor, "execute", "physics_controller_seconds"),
            (self.bundle.perception, "infer", "perception_seconds"),
            (self.bundle.verifier, "verify", "verifier_seconds"),
            (self.bundle.evaluator, "evaluate", "evaluator_seconds"),
            (self.bundle.snapshot_builder, "build", "snapshot_seconds"),
        ):
            restores.append((owner, name, _timed_method(owner, name, profile, key)))
        try:
            return original_step(self, snapshot, deterministic=deterministic)
        finally:
            for owner, name, original in reversed(restores):
                setattr(owner, name, original)
    Collector.step = profiled_step
    _COLLECTOR_PATCHED = True


def _fresh_verify(store_root: Path, generation: str, out_path: Path) -> None:
    code = (
        "import json,sys,torch;"
        "from cp_disr.persistence import GenerationStore;"
        "s=GenerationStore(sys.argv[1]);d=s.load(sys.argv[2]);"
        "x=torch.load(d['path']+'/model.pt',map_location='cpu',weights_only=False);"
        "assert 'model' in x and 'optimizer' in x and 'manifest' in x;"
        "print(json.dumps({'generation':d['generation'],'fresh_process':True,'model':True,'adam':True}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(store_root), generation],
        text=True, capture_output=True, check=True,
    )
    write_json(out_path, json.loads(proc.stdout.strip()))


def _save_ckpt_generation(path, policy, optimizer, extra, rng_payload, prior_sampler, collector):
    persistence_started = time.perf_counter()
    path = Path(path)
    saved = _ORIGINAL_SAVE_CKPT(
        path, policy, optimizer, extra, rng_payload, prior_sampler, collector
    )
    job_dir = path.parent.parent
    store_root = job_dir / "persistence"
    store = GenerationStore(store_root)
    meta_path = path.with_suffix(".json")
    rng_path = path.with_suffix(".rng.json")
    manifest = dict(extra)
    manifest.update({
        "training_source_commit": git_commit(Path.cwd()),
        "runtime_revision": RUNTIME_REVISION,
        "generation_checkpoint": path.name,
        "N": int(extra.get("N") or extra.get("interaction_count") or 0),
        "T": float(extra.get("T") or extra.get("interaction_seconds") or 0.0),
        "update": int(extra.get("complete_updates") or extra.get("update_count") or 0),
        "source_hashes": _source_hashes(Path.cwd()),
    })
    generation = path.stem
    if (store.generations / generation).exists():
        store.verify(generation)
        return saved
    episode_blob = _capture_episode(collector)
    episode_sidecar = path.with_suffix(".episode.pkl")
    if not episode_sidecar.exists():
        fd = os.open(str(episode_sidecar), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "wb") as f:
            f.write(episode_blob)
            f.flush()
            os.fsync(f.fileno())
    generation = store.publish(
        {
            "model.pt": path,
            "model.json": meta_path,
            "rng.json": rng_path,
            "episode.pkl": episode_blob,
            "manifest.json": manifest,
        },
        manifest,
        generation=generation,
    )
    store.append_event({
        "event": "checkpoint_generation_published",
        "generation": generation,
        "N": manifest["N"],
        "T": manifest["T"],
        "update": manifest["update"],
        "time": utc_now(),
    })
    store.derive_csv()
    if generation == "update_complete_1" or generation.startswith("final_n_"):
        _fresh_verify(store_root, generation, job_dir / ("fresh_load_%s.json" % generation))
    if generation == "update_complete_1":
        profile = dict(getattr(collector, "_phase_a_profile", {}) or {})
        wall = time.perf_counter() - float(getattr(collector, "_phase_a_train_wall_start", persistence_started))
        persistence_seconds = time.perf_counter() - persistence_started
        sim_seconds = float(manifest["T"])
        profile.update({
            "ppo_forward_backward_optimizer_seconds": float(getattr(collector, "_phase_a_ppo_seconds", 0.0)),
            "checkpoint_fsync_logging_seconds": persistence_seconds,
            "evaluation_seconds": 0.0,
            "wall_seconds_through_first_generation": wall,
            "transitions_per_wall_second": float(manifest["N"]) / wall if wall > 0 else None,
            "sim_seconds_per_wall_second": sim_seconds / wall if wall > 0 else None,
            "N": int(manifest["N"]),
            "T": sim_seconds,
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
            "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "cpu_count": os.cpu_count(),
            "profile_scope": "first_complete_1024_transition_update",
        })
        write_json(job_dir / "profiling_summary.json", profile)
    diagnostic_label = None
    if path.stem == "n_000000":
        diagnostic_label = "N=0"
    elif path.stem == "n_008192":
        diagnostic_label = "N=8192"
    elif path.stem.startswith("final_n_"):
        diagnostic_label = "final"
    if diagnostic_label is not None:
        _fixed_mechanism_diagnostic(
            Path.cwd(), str(extra["method"]), path, diagnostic_label,
            next(policy.parameters()).device,
        )
    if generation == "update_complete_1" and os.environ.get("CP_DISR_PHASE_A_FIRST_UPDATE_BARRIER") == "1":
        run_token = job_dir.name
        stamp = run_token.split("_", 1)[0]
        pair_dir = job_dir.parents[3] / stamp
        continue_path = pair_dir / "CONTINUE_AFTER_FIRST_UPDATE"
        stop_path = pair_dir / "STOP_AFTER_FIRST_UPDATE"
        write_json(job_dir / "waiting_for_pair_first_update_gate.json", {
            "waiting": True,
            "safe_generation": generation,
            "time": utc_now(),
        })
        deadline = time.monotonic() + 24 * 60 * 60
        while not continue_path.is_file():
            if stop_path.is_file():
                raise DataIntegrityError("pair first-update gate ordered a safe stop")
            if time.monotonic() >= deadline:
                raise DataIntegrityError("pair first-update gate timed out at safe generation")
            time.sleep(2.0)
        write_json(job_dir / "pair_first_update_gate_released.json", {
            "released": True,
            "authorization_sha256": sha256_file(continue_path),
            "time": utc_now(),
        })
    return saved


def _maybe_eval_at_points(
    root, task_id, method, policy, trainer, collector, sampler, count,
    interaction_seconds, hashes_doc, job_dir, eval_cases, split_index, device,
    eval_rows, extra_base, done_ns,
):
    if count in done_ns or count not in EVAL_POINTS:
        return
    extra = dict(extra_base)
    extra.update({
        "update_count": extra_base.get("complete_updates"),
        "interaction_count": count,
        "interaction_seconds": interaction_seconds,
        "N": count,
        "T": interaction_seconds,
    })
    ckpt = job_dir / "checkpoints" / ("n_%06d.pt" % count)
    _save_ckpt_generation(
        ckpt, policy, trainer.optimizer, extra, v11.s1.capture_rng(), sampler, collector
    )
    ev = v11.eval_episodes(
        root, task_id, method, ckpt, eval_cases, split_index, device, hashes_doc,
        job_dir / ("eval_n_%06d.json" % count), n_episodes=DEV10_N, label="dev10",
    )
    eval_rows.append({
        "update": extra_base.get("complete_updates"),
        "skill_transitions": count,
        "success_rate": ev["success_rate"],
        "success_n": ev["success_n"],
        "mean_discounted_return": ev["mean_discounted_return"],
        "checkpoint": str(ckpt),
    })
    v11.s1.csv_write(job_dir / "eval_metrics.csv", eval_rows)
    done_ns.add(count)


def _install_adapters() -> None:
    _configure_v11()
    v11.save_ckpt = _save_ckpt_generation
    v11.start_episode = _start_episode_with_state
    v11.maybe_eval_at_n = _maybe_eval_at_points
    v11.select_checkpoint = _select_checkpoint
    _install_collector_profiler()


def _nonempty_train_rows(root: Path) -> list[dict]:
    split = json.loads((root / "configs/splits/T_C_stage_2a_v11.json").read_text(encoding="utf-8"))
    rows = []
    from .platforms.libero.snapshot import load_cache_edges
    for row in split["train"]:
        cache_dir = root / row["cache_dir"]
        if load_cache_edges(cache_dir):
            rows.append(row)
        if len(rows) == 8:
            break
    if len(rows) != 8:
        raise BindingError("fewer than 8 nonempty-prior T_C train cases")
    return rows


def _fixed_mechanism_diagnostic(root: Path, method: str, checkpoint: Path, label: str, device) -> None:
    from .torch_rl import load_checkpoint
    rows = _nonempty_train_rows(root)
    bundle = v11.make_bundle(root, "T_C", Path("configs/splits/T_C_stage_2a_v11.json"))
    prof = v11.resolve_runtime(root)
    v11.attach_split_cases(bundle, root, rows, "T_C", prof["task_deadlines"]["T_C"])
    policy = v11.make_policy(bundle.template, method, device)
    load_checkpoint(checkpoint, policy)
    policy.eval()
    out_rows = []
    rng = v11.s1.capture_rng()
    try:
        for row in rows:
            case_id = row["case_id"]
            snap = bundle.start_case(case_id)
            snap, prior, source_n = v11.apply_episode_prior(
                method, bundle, snap, sampler=None, eval_original=True
            )
            with torch.no_grad():
                out = policy(snap, policy.initial_hidden())
            candidates = []
            for cid in snap.candidate_ids:
                struct = v11.s1.candidate_struct(out, snap, cid)
                candidates.append(struct)
            if method == "B2" and any(
                c["dp_rms"] != 0 or c["up_rms"] != 0 or (c["residual_abs"] or 0) != 0
                for c in candidates
            ):
                raise DataIntegrityError("B2 fixed diagnostic DP/uP/Delta not exactly zero")
            out_rows.append({
                "label": label,
                "classification": "IN_SAMPLE_MECHANISM_DIAGNOSTIC",
                "method": method,
                "case_id": case_id,
                "source_prior_n": source_n,
                "effective_prior_n": len(snap.prior_edges),
                "prior_mode": prior.audit_mode,
                "candidate_n": len(snap.candidate_ids),
                "single_candidate": len(snap.candidate_ids) == 1,
                "empty_patch_n": sum(1 for c in candidates if not c["nominal_patch_nonempty"]),
                "dp_nonzero_n": sum(1 for c in candidates if c["dp_nonzero"]),
                "delta_nonzero_n": sum(1 for c in candidates if c["residual_nonzero"]),
                "dk_rms_max": max(c["dk_rms"] for c in candidates),
                "dh_rms_max": max(c["dh_rms"] for c in candidates),
                "dp_rms_max": max(c["dp_rms"] for c in candidates),
                "up_rms_max": max(c["up_rms"] for c in candidates),
                "delta_abs_max": max((c["residual_abs"] or 0) for c in candidates),
                "used_for_training": False,
                "used_for_checkpoint_selection": False,
                "test_id_used": False,
            })
    finally:
        bundle.environment.close()
        v11.s1.restore_rng(rng)
    job_dir = checkpoint.parent.parent
    path = job_dir / "diagnostics.csv"
    existing = _read_csv(path) if path.is_file() else []
    existing = [r for r in existing if r.get("label") != label]
    all_rows = existing + out_rows
    with path.open("w", encoding="utf-8", newline="") as f:
        fields = sorted({k for row in all_rows for k in row})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(all_rows)


def freeze(root: Path, stamp: str | None = None) -> dict:
    root = Path(root).resolve()
    os.chdir(root)
    _install_adapters()
    gate_path, gate = _latest_final_gate(root)
    current = git_commit(root)
    if not subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_COMMIT, current], cwd=root
    ).returncode == 0:
        raise BindingError("training source is not descended from authorized base commit")
    dev10_path, dev10 = _freeze_dev10(root)
    prof = v11.bind_H(root)
    config = _phase_config(prof, current)
    configsha8 = hashlib.sha256(canonical(config).encode("utf-8")).hexdigest()[:8]
    stamp = stamp or utc_stamp()
    pair_dir = root / STAGE_DIR / stamp
    pair_dir.mkdir(parents=True, exist_ok=False)
    write_json(pair_dir / "phase_a_authorization.json", {
        **config,
        "authorization_time": utc_now(),
        "final_unblock_status": str(gate_path.relative_to(root)).replace("\\", "/"),
        "final_unblock_sha256": sha256_file(gate_path),
        "old_stage2a_stop_intact": (root / v11.STOP_REL).is_file(),
        "old_clock_stop_intact": (root / v11.CLOCK_STOP_REL).is_file(),
        "authorized_to_start": True,
    })
    write_json(pair_dir / "dev10_manifest.json", dev10)
    write_json(pair_dir / "source_hashes.json", _source_hashes(root))
    write_json(pair_dir / "pair_run_index.json", {
        "stamp": stamp,
        "configsha8": configsha8,
        "jobs": {
            pid: {
                "method": method,
                "status": "NOT_STARTED",
                "run_dir": str(STAGE_DIR / "T_C" / method / "seed_0" / f"{stamp}_{configsha8}"),
            }
            for method, pid in AUTHORIZED.items()
        },
    })
    write_json(root / STAGE_DIR / "CURRENT.json", {"stamp": stamp, "configsha8": configsha8})
    write_json(root / STATUS_PATH, {
        "phase_a_e16_ready": True,
        "phase_a_status": "RUNNING",
        "started_at": utc_now(),
        "stamp": stamp,
        "configsha8": configsha8,
        "training_source_commit": current,
        "authorized_jobs": list(AUTHORIZED.values()),
        "jobs": {pid: "NOT_STARTED" for pid in AUTHORIZED.values()},
        "formal_rl_jobs": 2,
        "test_id_used": False,
        "vlm_requests": 0,
    })
    return {"status": "FROZEN", "stamp": stamp, "configsha8": configsha8, "config": config}


def _current(root: Path, stamp: str | None, configsha: str | None) -> tuple[str, str]:
    if stamp and configsha:
        return stamp, configsha
    doc = json.loads((root / STAGE_DIR / "CURRENT.json").read_text(encoding="utf-8"))
    return stamp or doc["stamp"], configsha or doc["configsha8"]


def _verify_resume_generation(root: Path, method: str, stamp: str, configsha: str) -> dict:
    job_dir = _job_dir(root, method, stamp, configsha)
    resume_path = job_dir / "resume.json"
    if not resume_path.is_file():
        raise BindingError("resume requested without resume.json")
    resume = json.loads(resume_path.read_text(encoding="utf-8"))
    legacy = Path(resume["checkpoint"])
    store = GenerationStore(job_dir / "persistence")
    latest = store.latest()
    loaded = store.load(latest)
    generation_dir = Path(loaded["path"])
    manifest = loaded["manifest"]
    expected_generation = legacy.stem
    if latest != expected_generation:
        raise DataIntegrityError(
            "LATEST generation %s does not match resume checkpoint %s" % (latest, expected_generation)
        )
    pairs = (
        (legacy, generation_dir / "model.pt"),
        (legacy.with_suffix(".json"), generation_dir / "model.json"),
        (legacy.with_suffix(".rng.json"), generation_dir / "rng.json"),
        (legacy.with_suffix(".episode.pkl"), generation_dir / "episode.pkl"),
    )
    for legacy_path, immutable_path in pairs:
        if not legacy_path.is_file() or not immutable_path.is_file():
            raise DataIntegrityError("resume generation file missing: %s / %s" % (legacy_path, immutable_path))
        if sha256_file(legacy_path) != sha256_file(immutable_path):
            raise DataIntegrityError("resume generation mismatch: %s" % legacy_path.name)
    expected = {
        "N": int(resume.get("count") or 0),
        "T": float(resume.get("interaction_seconds") or 0.0),
        "update": int(resume.get("complete_updates") or 0),
        "runtime_revision": RUNTIME_REVISION,
        "training_source_commit": git_commit(root),
    }
    for key, value in expected.items():
        actual = manifest.get(key)
        if key == "T":
            if float(actual) != value:
                raise DataIntegrityError("resume manifest T mismatch")
        elif actual != value:
            raise DataIntegrityError("resume manifest %s mismatch: %r != %r" % (key, actual, value))
    recorded_split = (manifest.get("source_hashes") or {}).get("split_dev10")
    if recorded_split != _source_hashes(root).get("split_dev10"):
        raise DataIntegrityError("resume split hash mismatch")
    return {"latest": latest, "manifest": manifest, "verified": True}


def train(
    root: Path, method: str, gpu: int, stamp: str | None = None,
    configsha: str | None = None, max_updates: int = MAX_UPDATES,
    resume: bool = False,
) -> dict:
    root = Path(root).resolve()
    os.chdir(root)
    if method not in AUTHORIZED:
        raise BindingError("only Full and B2 are authorized")
    if int(max_updates) < 1 or int(max_updates) > MAX_UPDATES:
        raise BindingError("max_updates must be in [1,16]")
    _install_adapters()
    stamp, configsha = _current(root, stamp, configsha)
    pair_dir = root / STAGE_DIR / stamp
    auth = json.loads((pair_dir / "phase_a_authorization.json").read_text(encoding="utf-8"))
    if auth.get("training_source_commit") != git_commit(root):
        raise BindingError("production source changed after Phase A freeze")
    if auth.get("authorized_to_start") is not True:
        raise BindingError("Phase A authorization is not active")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(gpu))
        device = torch.device("cuda", int(gpu))
        device_name = torch.cuda.get_device_name(int(gpu))
    else:
        raise BindingError("CUDA required for formal Phase A run")
    prof = v11.bind_H(root)
    hashes = _source_hashes(root)
    if resume:
        write_json(
            _job_dir(root, method, stamp, configsha) / "resume_generation_verification.json",
            _verify_resume_generation(root, method, stamp, configsha),
        )
    status_path = root / STATUS_PATH
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["jobs"][AUTHORIZED[method]] = "RUNNING"
    status["last_started_at"] = utc_now()
    write_json(status_path, status)
    try:
        result = v11.train_job(
            root, "T_C", method, device, device_name, prof, hashes, stamp,
            configsha, max_updates=int(max_updates), resume=bool(resume),
        )
    except Exception as exc:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["phase_a_e16_ready"] = False
        status["phase_a_status"] = "NEEDS_RERUN"
        status["jobs"][AUTHORIZED[method]] = "NEEDS_RERUN"
        status["hard_failure"] = {"method": method, "error": str(exc), "time": utc_now()}
        write_json(status_path, status)
        raise
    state = "COMPLETED" if result.get("complete_updates") == MAX_UPDATES else "SAFE_BOUNDARY"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["jobs"][AUTHORIZED[method]] = state
    write_json(status_path, status)
    write_json(pair_dir / ("job_T_C_%s.json" % method), result)
    return {"status": state, "job": result, "stamp": stamp, "configsha8": configsha}


def _job_dir(root: Path, method: str, stamp: str, configsha: str) -> Path:
    return root / STAGE_DIR / "T_C" / method / "seed_0" / f"{stamp}_{configsha}"


def first_update_gate(root: Path, stamp: str | None = None, configsha: str | None = None) -> dict:
    root = Path(root).resolve()
    stamp, configsha = _current(root, stamp, configsha)
    rows = {}
    issues = []
    for method in ("Full", "B2"):
        job = _job_dir(root, method, stamp, configsha)
        check_path = job / "first_update_selfcheck.json"
        if not check_path.is_file():
            issues.append("%s missing first-update evidence" % method)
            continue
        check = json.loads(check_path.read_text(encoding="utf-8"))
        generation = GenerationStore(job / "persistence").load("update_complete_1")
        audit = check.get("clock_audit") or {}
        ok = (
            check.get("n_transitions") == ROLLOUT_N
            and check.get("param_changed") is True
            and audit.get("N_logged") == ROLLOUT_N
            and audit.get("N_used_by_PPO") == ROLLOUT_N
            and audit.get("N_clock_verified_valid") == ROLLOUT_N
            and not audit.get("N_clock_invalid")
            and not audit.get("N_unknown")
            and not audit.get("mask_mismatch_n")
            and generation["generation"] == "update_complete_1"
            and (job / "fresh_load_update_complete_1.json").is_file()
        )
        rows[method] = {"passed": bool(ok), "check": check, "generation": generation}
        if not ok:
            issues.append("%s first-update hard gate failed" % method)
    result = {"passed": not issues, "issues": issues, "jobs": rows, "time": utc_now()}
    pair_dir = root / STAGE_DIR / stamp
    write_json(pair_dir / "first_update_pair_gate.json", result)
    if issues:
        write_json(pair_dir / "STOP_AFTER_FIRST_UPDATE", {
            "authorized": False, "issues": issues, "time": utc_now(),
        })
        status = json.loads((root / STATUS_PATH).read_text(encoding="utf-8"))
        status["phase_a_e16_ready"] = False
        status["phase_a_status"] = "NEEDS_RERUN"
        status["hard_failure"] = {"gate": "first_update_pair", "issues": issues}
        write_json(root / STATUS_PATH, status)
        raise DataIntegrityError("; ".join(issues))
    write_json(pair_dir / "CONTINUE_AFTER_FIRST_UPDATE", {
        "authorized": True,
        "gate_sha256": sha256_file(pair_dir / "first_update_pair_gate.json"),
        "time": utc_now(),
    })
    return result


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def report_8192(root: Path, stamp: str | None = None, configsha: str | None = None) -> dict:
    root = Path(root).resolve()
    stamp, configsha = _current(root, stamp, configsha)
    jobs = {}
    issues = []
    for method in ("Full", "B2"):
        job = _job_dir(root, method, stamp, configsha)
        eval_rows = _read_csv(job / "eval_metrics.csv")
        train_rows = _read_csv(job / "train_metrics.csv")
        ns = {int(float(r["skill_transitions"])) for r in eval_rows}
        if not {0, 4096, 8192}.issubset(ns):
            issues.append("%s missing 0/4096/8192 dev10 evaluation" % method)
        jobs[method] = {"eval": eval_rows, "train": train_rows}
    payload = {"status": "PASS" if not issues else "NEEDS_RERUN", "issues": issues, "jobs": jobs}
    write_json(root / STAGE_DIR / stamp / "first_evidence_8192.json", payload)
    lines = ["# CP-DISR v1.2 Phase A first evidence at N=8192", "", "Status: %s" % payload["status"], ""]
    for method, data in jobs.items():
        lines += ["## %s" % method, "", "Dev10 curve:", ""]
        for row in data["eval"]:
            lines.append("- N=%s: mean discounted return=%s, success=%s/%s" % (
                row.get("skill_transitions"), row.get("mean_discounted_return"),
                row.get("success_n"), DEV10_N,
            ))
        if data["train"]:
            last = data["train"][-1]
            lines += ["", "Latest structure diagnostics: DP opportunities=%s, DP nonzero=%s, Delta nonzero=%s." % (
                last.get("dp_opportunity_n"), last.get("dp_nonzero_n"), last.get("delta_nonzero_n"),
            )]
        lines.append("")
    report = "\n".join(lines) + "\n"
    for path in (root / STAGE_DIR / stamp / "first_evidence_8192.md", root / "reports/phase_a_first_evidence_8192.md"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
    if issues:
        raise DataIntegrityError("; ".join(issues))
    return payload


def finalize(root: Path, gpu: int, stamp: str | None = None, configsha: str | None = None) -> dict:
    root = Path(root).resolve()
    os.chdir(root)
    _install_adapters()
    stamp, configsha = _current(root, stamp, configsha)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(gpu))
        device = torch.device("cuda", int(gpu))
    else:
        raise BindingError("CUDA required")
    source_split = json.loads((root / v11.SOURCE_SPLITS["T_C"]).read_text(encoding="utf-8"))
    enabled_split = json.loads((root / "configs/splits/T_C_stage_2a_v11.json").read_text(encoding="utf-8"))
    dev20 = [r["case_id"] for r in enabled_split["dev"]]
    split_index = {r["case_id"]: r for r in enabled_split["train"] + enabled_split["dev"]}
    selected = {}
    dev20_rows = []
    for method in ("Full", "B2"):
        job = _job_dir(root, method, stamp, configsha)
        summary = json.loads((job / "job_summary.json").read_text(encoding="utf-8"))
        if summary.get("complete_updates") != MAX_UPDATES or summary.get("valid_transitions") != N_CAP:
            raise BindingError("%s did not complete E16/N16384" % method)
        selection = json.loads((job / "checkpoint_selection.json").read_text(encoding="utf-8"))
        ckpt = Path(selection["selected"]["checkpoint"])
        selected[method] = {"checkpoint": str(ckpt), "sha256": sha256_file(ckpt), "selection": selection}
        ev = v11.eval_episodes(
            root, "T_C", method, ckpt, dev20, split_index, device,
            _source_hashes(root), job / "common_dev20.json", n_episodes=20, label="common_dev20",
        )
        dev20_rows.append({
            "method": method, "checkpoint": str(ckpt), "sha256": sha256_file(ckpt),
            "success_n": ev["success_n"], "success_rate": ev["success_rate"],
            "mean_discounted_return": ev["mean_discounted_return"],
        })
    pair_dir = root / STAGE_DIR / stamp
    write_json(pair_dir / "selected_checkpoints.json", {"frozen_before_common_dev20": True, "selected": selected})
    with (pair_dir / "common_dev20_results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dev20_rows[0]))
        w.writeheader(); w.writerows(dev20_rows)
    status = json.loads((root / STATUS_PATH).read_text(encoding="utf-8"))
    status.update({
        "phase_a_e16_ready": True,
        "phase_a_status": "COMPLETED",
        "completed_at": utc_now(),
        "jobs": {pid: "COMPLETED" for pid in AUTHORIZED.values()},
        "formal_rl_jobs": 2,
        "real_environment_ppo_updates": 32,
        "VLM_requests": 0,
        "final_test_episodes": 0,
        "test_id_used": False,
        "B1-K": "NOT_AUTHORIZED",
        "A_CAT": "NOT_AUTHORIZED",
        "B0": "NOT_AUTHORIZED",
        "T_B": "NOT_AUTHORIZED",
    })
    write_json(root / STATUS_PATH, status)
    summary = {
        "status": "COMPLETED",
        "selected": selected,
        "common_dev20": dev20_rows,
        "test_id_used": False,
        "formal_rl_jobs": 2,
        "real_environment_ppo_updates": 32,
        "vlm_requests": 0,
        "final_test_episodes": 0,
    }
    write_json(pair_dir / "phase_a_pair_summary.json", summary)
    lines = ["# CP-DISR v1.2 Phase A summary", "", "Phase A status: COMPLETED", "", "No final test-ID evaluation was run.", ""]
    for row in dev20_rows:
        lines.append("- %s common dev20: mean discounted return=%s; success=%s/20" % (
            row["method"], row["mean_discounted_return"], row["success_n"],
        ))
    lines += ["", "Formal RL jobs: 2; real-environment PPO updates: 32; VLM requests: 0; final test episodes: 0.", ""]
    report = "\n".join(lines)
    for path in (pair_dir / "phase_a_pair_summary.md", root / REPORT_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
    return summary


def cmd_phase_a_v12(
    root, phase="freeze", method=None, gpu=0, stamp=None, configsha=None,
    max_updates=MAX_UPDATES, resume=False,
):
    root = Path(root)
    if phase == "freeze":
        return freeze(root, stamp=stamp)
    if phase == "train":
        if method not in AUTHORIZED:
            raise BindingError("train requires --method Full or B2")
        return train(root, method, gpu, stamp, configsha, max_updates, resume)
    if phase == "first-update-gate":
        return first_update_gate(root, stamp, configsha)
    if phase == "report-8192":
        return report_8192(root, stamp, configsha)
    if phase == "finalize":
        return finalize(root, gpu, stamp, configsha)
    raise BindingError("unsupported Phase A v1.2 phase: %s" % phase)
