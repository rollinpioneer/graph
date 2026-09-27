"""CP-DISR v1.3 R2: the single authorized T_B/B0/seed-0 E16 run.

This adapter is intentionally kept separate from the production runner.  It
binds the R2 output/status namespace, reuses the frozen R1 train/dev split and
the Method 2.1.1 stage2a_v11 trainer, and adds only B0 allowlisting plus
evaluation interval provenance.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from cp_disr.common import BindingError, DataIntegrityError, canonical
from cp_disr import phase_a_v12 as v12
from cp_disr import stage2a_v11 as v11

BASE_COMMIT = "55a3b7ce35edebbbb8a587fe1e3a98ca1967db07"
TASK = "T_B"
METHODS = ("B0",)
AUTHORIZED = {"B0": "v13_R2_T_B_B0_s0_E16"}
N_CAP = 16384
ROLLOUT_N = 1024
MAX_UPDATES = 16
DEV10_N = 10
EVAL_POINTS = (0, 4096, 8192, 16384)
STAGE_DIR = Path("runs/v13_r2")
STATUS_PATH = Path("status/v13_r2.json")
SPLIT_REL = Path("configs/splits/T_B_stage_2a_v11.json")
DEV10_REL = Path("configs/splits/T_B_phase_a_v13_r1_dev10.json")
RUNTIME_REL = Path("experiments/manifests/runtime_manifest_v211_r2.yaml")
R1_DIR = Path("runs/v13_r1/20260926T112613Z")
PLAN_VERSION = "1.3-R2-B0"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(value) + "\n", encoding="utf-8")


def git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def configure() -> None:
    v12.STAGE_DIR = STAGE_DIR
    v12.STATUS_PATH = STATUS_PATH
    v12.EVAL_POINTS = EVAL_POINTS
    v12.DEV10_N = DEV10_N
    v11.STAGE_DIR = STAGE_DIR
    v11.N_CAP = N_CAP
    v11.ROLLOUT_N = ROLLOUT_N
    v11.MAX_UPDATES = MAX_UPDATES
    v11.DEV_EPISODES = DEV10_N
    v11.TASKS = (TASK,)
    v11.RUNTIME_REL = RUNTIME_REL
    v11.METHODS = METHODS
    v11.PLANNED = {(TASK, "B0"): AUTHORIZED["B0"]}
    v11.ENABLED_SPLITS = dict(v11.ENABLED_SPLITS)
    v11.ENABLED_SPLITS[TASK] = DEV10_REL
    v11.save_ckpt = v12._save_ckpt_generation
    v11.start_episode = v12._start_episode_with_state
    v11.maybe_eval_at_n = _maybe_eval_r2
    v11.select_checkpoint = v12._select_checkpoint
    v12._source_hashes = _source_hashes
    v12._fixed_mechanism_diagnostic = _fixed_mechanism_diagnostic
    v12._install_collector_profiler()


def _source_hashes(root: Path) -> dict:
    files = {
        "r2_adapter": root / "src/cp_disr/phase_a_v13_r2_b0.py",
        "r1_adapter": root / "src/cp_disr/phase_a_v13_r1.py",
        "stage2a_v11": root / "src/cp_disr/stage2a_v11.py",
        "collector": root / "src/cp_disr/collector.py",
        "torch_rl": root / "src/cp_disr/torch_rl.py",
        "neural": root / "src/cp_disr/neural.py",
        "persistence": root / "src/cp_disr/persistence.py",
        "runtime_factory": root / "src/cp_disr/platforms/libero/runtime_factory.py",
        "clock": root / "src/cp_disr/platforms/libero/clock.py",
        "safety": root / "src/cp_disr/platforms/libero/safety.py",
        "skill_executor": root / "src/cp_disr/platforms/libero/skill_executor.py",
        "runtime_manifest": root / v11.RUNTIME_REL,
        "split": root / SPLIT_REL,
        "split_dev10": root / DEV10_REL,
    }
    return {k: sha(p) if p.is_file() else None for k, p in files.items()} | {
        "git_commit": git_head(root),
        "baseline_commit": BASE_COMMIT,
    }


def _fixed_mechanism_diagnostic(root: Path, method: str, checkpoint: Path, label: str, device) -> None:
    job = Path(checkpoint).parent.parent
    with (job / "fixed_mechanism_scope.jsonl").open("a", encoding="utf-8") as f:
        f.write(canonical({
            "label": label, "method": method, "checkpoint": str(checkpoint),
            "effective_prior": 0, "test_id_used": False,
            "used_for_training": False,
            "graph_message_passing": False,
            "nominal_successor": False,
            "note": "B0 uses the production two-SAB Set Transformer path.",
        }) + "\n")


def _job_dir(root: Path, stamp: str, configsha: str) -> Path:
    return root / STAGE_DIR / TASK / "B0" / "seed_0" / f"{stamp}_{configsha}"


def _assert_startup(root: Path) -> dict:
    root = root.resolve()
    if not (root / "src/cp_disr/stage2a_v11.py").is_file():
        raise BindingError("active repository missing stage2a_v11")
    if os.environ.get("PYTHONPATH", "").find(str(root / "src")) < 0:
        raise BindingError("cp_disr import path is not the active R2 repository")
    imported = subprocess.check_output(
        [sys.executable, "-c", "import cp_disr; print(cp_disr.__file__)"], text=True
    ).strip()
    if str(root / "src") not in imported:
        raise BindingError(f"cp_disr imported outside active repository: {imported}")
    split = json.loads((root / SPLIT_REL).read_text(encoding="utf-8"))
    dev10 = json.loads((root / DEV10_REL).read_text(encoding="utf-8"))
    if split.get("task_id") != TASK or len(split.get("train") or []) != 64 or len(split.get("dev") or []) != 20:
        raise BindingError("T_B source split identity/count mismatch")
    if len(dev10.get("train") or []) != 64 or len(dev10.get("dev") or []) != 10:
        raise BindingError("R1 dev10 split identity/count mismatch")
    if any(Path(p).exists() for p in root.glob("runs/v13_r2/**/checkpoints/*.pt")):
        raise BindingError("R2 output already contains a checkpoint")
    if any(Path(p).exists() for p in root.glob("runs/v13_r1/**/checkpoints/*.pt")):
        raise BindingError("R1 checkpoints unexpectedly present in the baseline worktree")
    return {
        "python": sys.executable,
        "cp_disr_import": imported,
        "split_sha256": sha(root / SPLIT_REL),
        "dev10_sha256": sha(root / DEV10_REL),
        "test_ids": 0,
        "r1_readonly": str(R1_DIR),
    }


def _b0_startup_gate(root: Path, gpu: int = 1) -> dict:
    """Run only the direct B0 structural gate; no learning or qualification run."""
    import torch
    if not torch.cuda.is_available():
        raise BindingError("CUDA required for B0 startup gate")
    torch.cuda.set_device(int(gpu))
    split = json.loads((root / DEV10_REL).read_text(encoding="utf-8"))
    bundle = v11.make_bundle(root, TASK, DEV10_REL)
    v11.attach_split_cases(bundle, root, list(split["train"] + split["dev"]), TASK, v11.bind_H(root)["task_deadlines"][TASK])
    case_id = split["dev"][0]["case_id"]
    try:
        dry = v11.local_dry_run(bundle, "B0", torch.device("cuda", int(gpu)), case_id, empty=True)
    finally:
        try:
            bundle.environment.close()
        except Exception:
            pass
    diag = dry.get("diagnostics") or {}
    issues = []
    if not dry.get("logits_finite"):
        issues.append("masked logits nonfinite")
    if dry.get("effective_prior_relation_count") != 0:
        issues.append("effective prior not empty")
    if any(s.get("successor_used") for s in dry.get("forward_structs") or []):
        issues.append("nominal successor or graph successor used")
    if diag.get("method") != "B0":
        issues.append("policy method diagnostic mismatch")
    dry["gate"] = "PASS" if not issues else "FAIL"
    dry["issues"] = issues
    dry["training_update"] = False
    dry["qualification_run"] = False
    return dry


def _read_r1_manifest(root: Path) -> dict:
    out = {}
    for name in ("r1_summary.md", "job_T_B_B1-K.json", "job_T_B_B2.json", "checkpoint_index.json", "checkpoint_generations.csv", "final_dev_comparison.csv", "dev10_manifest.json", "source_hashes.json", "plan_resolution.json", "Results_draft.md", "cost_ledger.csv", "historical_evidence.md"):
        p = root / R1_DIR / name
        out[name] = {"path": str(p.relative_to(root)).replace("\\", "/"), "sha256": sha(p), "bytes": p.stat().st_size}
    return out


def _source_compatibility(root: Path) -> dict:
    shared = [
        "stage2a_v11.py", "neural.py", "torch_rl.py", "collector.py",
        "contracts.py", "facts.py", "graph.py", "execution.py",
        "platforms/libero/runtime_factory.py", "platforms/libero/clock.py",
        "platforms/libero/safety.py", "platforms/libero/skill_executor.py",
    ]
    return {
        "baseline_commit": BASE_COMMIT,
        "r2_training_commit": git_head(root),
        "new_source": "src/cp_disr/phase_a_v13_r2_b0.py",
        "changed_scope": [
            "B0 allowlist and v13_r2 output/status routing",
            "R2 plan/provenance files",
            "evaluation records add episode start, first valid success, complete duration, and terminal skill duration",
            "path-relocated copy of the existing runtime manifest for the isolated checkout",
        ],
        "shared_training_modules_unchanged": shared,
        "shared_module_sha256": {p: sha(root / "src/cp_disr" / p) for p in shared},
        "r1_adapter_unchanged": sha(root / "src/cp_disr/phase_a_v13_r1.py"),
        "no_changes": ["policy/PPO/environment/reward/clock/mask/candidate encoding behavior"],
    }


def freeze(root: Path, stamp: str | None = None, gpu: int = 1) -> dict:
    root = Path(root).resolve()
    os.chdir(root)
    configure()
    current = git_head(root)
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASE_COMMIT, current], cwd=root).returncode:
        raise BindingError(f"R2 source must descend from baseline {BASE_COMMIT}, got {current}")
    startup = _assert_startup(root)
    b0_gate = _b0_startup_gate(root, gpu)
    startup["b0_structural_gate"] = b0_gate
    prof = v11.bind_H(root)
    d_ref = float(prof["d_ref"][TASK])
    tcap = N_CAP * d_ref
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    split_sha = sha(root / SPLIT_REL)
    dev10_sha = sha(root / DEV10_REL)
    config = {
        "plan_version": PLAN_VERSION,
        "task": TASK, "method": "B0", "seed": 0,
        "planned_id": AUTHORIZED["B0"],
        "initialization": "FROM_SCRATCH",
        "Ncap": N_CAP, "Tcap": tcap, "H": float(prof["H"]), "d_ref": d_ref,
        "rollout": ROLLOUT_N, "max_complete_updates": MAX_UPDATES,
        "ppo_epochs": 4, "sequence_length": 16, "minibatch_target": 64,
        "actor_episode_discount_weight": False,
        "gamma_rule": "2**(-duration_seconds/H)",
        "prior_mode": "always_absent",
        "graph_message_passing": False,
        "nominal_successor": False,
        "b0_encoder": "two_SAB_SetTransformer_padding_mask_only",
        "test_id_used": False, "vlm_requests": 0,
        "training_source_commit": current,
        "runtime_revision": v12.RUNTIME_REVISION,
        "runtime_manifest": str(RUNTIME_REL),
        "split": str(SPLIT_REL), "split_sha256": split_sha,
        "dev10": str(DEV10_REL), "dev10_sha256": dev10_sha,
        "startup": startup,
    }
    configsha = hashlib.sha256(canonical(config).encode()).hexdigest()[:8]
    pair = root / STAGE_DIR / stamp
    pair.mkdir(parents=True, exist_ok=False)
    write_json(pair / "r2_authorization.json", {
        "status": "FROZEN", "created_at": utc_now(), "baseline_commit": BASE_COMMIT,
        "training_source_commit": current, "planned_id": AUTHORIZED["B0"],
        "scope": "exactly one new formal run; R1 training/evaluation reused read-only",
        "forbidden": ["B1-K/B2 retraining", "R1 reevaluation", "VLM", "test-ID", "T_C", "extra seeds", "Full", "A_CAT"],
        "gpu": "selected at launch; requested GPU 1",
    })
    write_json(pair / "plan_resolution.json", config)
    write_json(pair / "dev10_manifest.json", {
        "frozen_before_b0_result": True, "source_split": str(SPLIT_REL),
        "source_sha256": split_sha, "derived_split": str(DEV10_REL),
        "derived_sha256": dev10_sha, "dev20_order": [r["case_id"] for r in json.loads((root / SPLIT_REL).read_text())["dev"]],
        "dev10_order": [r["case_id"] for r in json.loads((root / DEV10_REL).read_text())["dev"]],
        "test_ids_accessed": False, "selection_reused_from_r1": True,
    })
    write_json(pair / "source_hashes.json", _source_hashes(root))
    write_json(pair / "r1_readonly_evidence_manifest.json", _read_r1_manifest(root))
    (pair / "r1_metric_definitions.md").write_text(
        "# R1 metric definitions reused by R2\n\n"
        "R1 files are read-only evidence. N=8192 is the last common measured point; "
        "B1-K final N=13623 and B2 final N=14464 remain distinct. B1-K selected N=0; "
        "B2 selected N=14464. dev20 is single-seed, post-selection development data.\n\n"
        "Full-task success duration is episode start to independent first valid success. "
        "Terminal skill duration is not substituted for full-task duration; unrecoverable "
        "historical fields remain NOT_RECOVERABLE.\n", encoding="utf-8")
    (pair / "r1_metric_provenance.csv").write_text(
        "method,metric,source,notes\n"
        "B1-K,dev10,job_T_B_B1-K.json,raw frozen R1 evaluation\n"
        "B2,dev10,job_T_B_B2.json,raw frozen R1 evaluation\n"
        "B1-K/B2,dev20,final_dev_comparison.csv,raw frozen single-seed post-selection development\n"
        "B1-K/B2,success_seconds,historical R1 evaluation payload,interval audited in R2; no retroactive reconstruction\n",
        encoding="utf-8")
    (pair / "source_compatibility_report.md").write_text(
        "# R2 source compatibility\n\n"
        f"Baseline: `{BASE_COMMIT}`. R2 training commit: `{current}`.\n\n"
        "Only the R2 adapter is new. The shared Policy, B0 two-SAB Set Transformer, "
        "PPO, collector, reward, clock, safety, candidate encoding and runtime modules "
        "are unchanged. R2 adds the B0 allowlist, output/status routing, provenance, "
        "and evaluation interval fields.\n", encoding="utf-8")
    write_json(root / STAGE_DIR / "CURRENT.json", {"stamp": stamp, "configsha8": configsha, "planned_id": AUTHORIZED["B0"]})
    write_json(root / STATUS_PATH, {
        "phase": "R2", "status": "FROZEN", "task": TASK, "method": "B0", "seed": 0,
        "planned_id": AUTHORIZED["B0"], "stamp": stamp, "configsha8": configsha,
        "training_source_commit": current, "jobs": {AUTHORIZED["B0"]: "NOT_STARTED"},
        "r1_training_added": 0, "r1_evaluation_added": 0, "vlm_requests": 0,
        "test_id_used": False, "t_c_prototype": 0, "other_methods": 0, "other_seeds": 0,
    })
    return {"status": "FROZEN", "stamp": stamp, "configsha8": configsha, "config": config}


def _current(root: Path, stamp: str | None, configsha: str | None) -> tuple[str, str]:
    doc = json.loads((root / STAGE_DIR / "CURRENT.json").read_text())
    return stamp or doc["stamp"], configsha or doc["configsha8"]


def train(root: Path, gpu: int, stamp: str | None = None, configsha: str | None = None) -> dict:
    root = Path(root).resolve()
    os.chdir(root)
    configure()
    stamp, configsha = _current(root, stamp, configsha)
    auth = json.loads((root / STAGE_DIR / stamp / "plan_resolution.json").read_text())
    if auth["training_source_commit"] != git_head(root):
        raise BindingError("production source changed after R2 freeze")
    import torch
    if not torch.cuda.is_available():
        raise BindingError("CUDA required")
    torch.cuda.set_device(int(gpu))
    device = torch.device("cuda", int(gpu))
    prof = v11.bind_H(root)
    prof["Ncap"] = N_CAP; prof["max_updates"] = MAX_UPDATES
    prof["Tcap"][TASK] = auth["Tcap"]
    hashes = _source_hashes(root)
    status_path = root / STATUS_PATH
    status = json.loads(status_path.read_text())
    status.update({"status": "RUNNING", "phase": "R2", "gpu": int(gpu), "started_at": utc_now()})
    status["jobs"][AUTHORIZED["B0"]] = "RUNNING"
    write_json(status_path, status)
    try:
        result = v11.train_job(root, TASK, "B0", device, torch.cuda.get_device_name(int(gpu)), hashes, stamp, configsha, max_updates=MAX_UPDATES, resume=False)
    except Exception as exc:
        status = json.loads(status_path.read_text())
        status.update({"status": "FAILED", "phase": "R2", "hard_failure": {"error": repr(exc), "time": utc_now()}})
        status["jobs"][AUTHORIZED["B0"]] = "FAILED"
        write_json(status_path, status)
        raise
    write_json(root / STAGE_DIR / stamp / "job_T_B_B0.json", result)
    status = json.loads(status_path.read_text())
    status.update({"status": "TRAIN_COMPLETE", "phase": "R2", "completed_training_at": utc_now(), "actual_N": result.get("valid_transitions"), "actual_T": result.get("interaction_seconds"), "complete_updates": result.get("complete_updates"), "fragment_updates": result.get("fragment_updates"), "optimizer_steps": result.get("optimizer_steps")})
    status["jobs"][AUTHORIZED["B0"]] = "COMPLETE"
    write_json(status_path, status)
    return {"status": "COMPLETE", "job": result, "stamp": stamp, "configsha8": configsha}


def _enhanced_eval(root, task_id, method, ckpt_path, cases, split_index, device, hashes_doc, out_eval, n_episodes=None, label="dev10"):
    """The production evaluator with explicit interval provenance fields."""
    from cp_disr.collector import Collector
    from cp_disr.torch_rl import load_checkpoint
    v11.bind_H(root)
    rng_before = v11.s1.capture_rng()
    bundle = v11.make_bundle(root, task_id, v11.ENABLED_SPLITS[task_id])
    prof = v11.resolve_runtime(root)
    v11.attach_split_cases(bundle, root, list(split_index.values()), task_id, prof["task_deadlines"][task_id])
    v11.s1.seed_all(0)
    policy = v11.make_policy(bundle.template, method, device)
    load_checkpoint(ckpt_path, policy)
    policy.eval(); collector = Collector(bundle, policy)
    rows = []; success_n = 0; returns = []
    try:
        for case in cases[:(n_episodes or len(cases))]:
            rec = split_index[case]; v11.require_case_cache(root, rec)
            snap = bundle.start_case(case)
            snap, prior, source_n = v11.apply_episode_prior(method, bundle, snap, sampler=None, eval_original=True)
            collector.reset_episode(snap.env_id, snap.episode_id)
            episode_start_clock = float(bundle.episode_start_seconds)
            G = 0.0; steps = 0; success = False; reason = None
            first_success_seconds = None; terminal_skill_duration = None
            while True:
                t, result = collector.step(snap, deterministic=True)
                if t is None:
                    reason = result.get("reason") if isinstance(result, dict) else getattr(result, "reason", None)
                    success = bool(result.get("success") if isinstance(result, dict) else getattr(result, "success", False))
                    break
                G += float(t.reward) * float(t.weight); steps += 1
                result_success = bool(result.get("success") if isinstance(result, dict) else getattr(result, "success", False))
                if result_success and not success:
                    success = True
                    first_success_seconds = float(getattr(t.snapshot, "elapsed_seconds", 0.0) + t.duration)
                    terminal_skill_duration = float(t.duration)
                snap = t.next_snapshot
                if t.terminated or t.truncated:
                    reason = result.reason if not isinstance(result, dict) else result.get("reason")
                    break
            if success: success_n += 1
            returns.append(G)
            rows.append({
                "case_id": case, "success": success, "G": G, "steps": steps, "reason": reason,
                "episode_start_clock_seconds": episode_start_clock,
                "first_valid_success_seconds": first_success_seconds,
                "complete_episode_duration_seconds": first_success_seconds if success else None,
                "terminal_skill_duration_seconds": terminal_skill_duration,
                "success_seconds": first_success_seconds if success else None,
                "duration_interval": "episode_start_to_first_valid_success; terminal_skill_duration_separate",
                "source_n": source_n, "prior_mode": prior.audit_mode,
            })
    finally:
        try: bundle.environment.close()
        except Exception: pass
        v11.s1.restore_rng(rng_before)
    payload = {
        "task": task_id, "method": method, "alias": method,
        "checkpoint": str(ckpt_path), "success_n": success_n, "n": len(rows),
        "success_rate": success_n / max(1, len(rows)) if rows else None,
        "mean_discounted_return": float(sum(returns) / len(returns)) if returns else None,
        "rows": rows, "hashes": hashes_doc, "isolated_rng": True,
        "H": v11.suite_half_life(), "eval_action": "deterministic_argmax",
        "label": label, "optimizer_steps": 0,
        "duration_definition": "complete_episode_duration_seconds = first_valid_success_seconds - episode_start_elapsed_seconds; terminal_skill_duration_seconds is separate",
    }
    write_json(out_eval, payload)
    return payload


def _maybe_eval_r2(root, task_id, method, policy, trainer, collector, sampler, count, interaction_seconds, hashes_doc, job_dir, eval_cases, split_index, device, eval_rows, extra_base, done_ns):
    if count in done_ns or count not in EVAL_POINTS:
        return
    extra = dict(extra_base)
    extra.update({"update_count": extra_base.get("complete_updates"), "interaction_count": count, "interaction_seconds": interaction_seconds, "N": count, "T": interaction_seconds})
    ckpt = job_dir / "checkpoints" / ("n_%06d.pt" % count)
    v12._save_ckpt_generation(ckpt, policy, trainer.optimizer, extra, v11.s1.capture_rng(), sampler, collector)
    ev = _enhanced_eval(root, task_id, method, ckpt, eval_cases, split_index, device, hashes_doc, job_dir / ("eval_n_%06d.json" % count), n_episodes=DEV10_N, label="dev10")
    eval_rows.append({"update": extra_base.get("complete_updates"), "skill_transitions": count, "success_rate": ev["success_rate"], "success_n": ev["success_n"], "mean_discounted_return": ev["mean_discounted_return"], "checkpoint": str(ckpt)})
    v11.s1.csv_write(job_dir / "eval_metrics.csv", eval_rows)
    done_ns.add(count)


def finalize(root: Path, stamp: str | None = None, configsha: str | None = None, gpu: int = 1) -> dict:
    root = Path(root).resolve(); os.chdir(root); configure()
    stamp, configsha = _current(root, stamp, configsha)
    job_dir = _job_dir(root, stamp, configsha)
    summary = json.loads((job_dir / "job_summary.json").read_text())
    import torch
    device = torch.device("cuda", int(gpu)) if torch.cuda.is_available() else torch.device("cpu")
    rows = list(csv.DictReader((job_dir / "eval_metrics.csv").open(encoding="utf-8")))
    # Ensure numeric fields are available for deterministic R2 selection.
    for r in rows:
        for k in ("update", "skill_transitions", "success_n"):
            if r.get(k) not in (None, ""): r[k] = int(float(r[k]))
        for k in ("success_rate", "mean_discounted_return"):
            if r.get(k) not in (None, ""): r[k] = float(r[k])
    selected, selection = v12._select_checkpoint(rows)
    if not selected: raise BindingError("B0 checkpoint selection unavailable")
    write_json(job_dir / "checkpoint_selection.json", selection)
    selected_eval = _enhanced_eval(root, TASK, "B0", Path(selected), [r["case_id"] for r in json.loads((root / DEV10_REL).read_text())["dev"]], {r["case_id"]: r for r in json.loads((root / SPLIT_REL).read_text())["train"] + json.loads((root / SPLIT_REL).read_text())["dev"]}, device, _source_hashes(root), job_dir / "common_dev20.json", n_episodes=20, label="common_dev20")
    pair = root / STAGE_DIR / stamp
    write_json(pair / "b0_checkpoint_selection.json", {"frozen_before_common_dev20": True, "checkpoint": str(Path(selected).relative_to(root)), "sha256": sha(Path(selected)), "selection": selection, "N": int(summary["valid_transitions"]), "T": float(summary["interaction_seconds"])})
    write_json(pair / "selected_dev20_comparison.json", {"method": "B0", "checkpoint": str(Path(selected).relative_to(root)), "checkpoint_sha256": sha(Path(selected)), "success_n": selected_eval["success_n"], "success_rate": selected_eval["success_rate"], "mean_discounted_return": selected_eval["mean_discounted_return"], "label": "DEVELOPMENT/SINGLE_SEED/POST_SELECTION"})
    status = json.loads((root / STATUS_PATH).read_text())
    status.update({"status": "COMPLETE", "completed_at": utc_now(), "selected_checkpoint": str(Path(selected).relative_to(root)), "selected_checkpoint_sha256": sha(Path(selected)), "dev20_success_n": selected_eval["success_n"], "dev20_success_rate": selected_eval["success_rate"], "dev20_mean_discounted_return": selected_eval["mean_discounted_return"], "evaluation_episodes": 120})
    write_json(root / STATUS_PATH, status)
    return {"status": "COMPLETE", "selected": str(selected), "selection": selection, "common_dev20": selected_eval, "summary": summary}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["freeze", "train", "finalize"])
    ap.add_argument("--gpu", type=int, default=1); ap.add_argument("--stamp"); ap.add_argument("--configsha")
    a = ap.parse_args(argv); root = Path.cwd()
    if a.phase == "freeze": out = freeze(root, a.stamp, a.gpu)
    elif a.phase == "train": out = train(root, a.gpu, a.stamp, a.configsha)
    else: out = finalize(root, a.stamp, a.configsha, a.gpu)
    print(canonical(out)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
