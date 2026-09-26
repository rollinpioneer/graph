"""CP-DISR v1.3 R1: existing-task T_B B1-K/B2 seed-0 E16.

This is a narrow configuration adapter over the production Method 2.1.1
runner. It does not add a trainer or alter task semantics.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common import BindingError, DataIntegrityError, canonical
from . import phase_a_v12 as v12
from . import stage2a_v11 as v11

BASE_COMMIT = "66614acb443dcacee2b16c5472d4ecbd697bf561"
TASK = "T_B"
METHODS = ("B1-K", "B2")
AUTHORIZED = {"B1-K": "v13_R1_T_B_B1K_s0_E16", "B2": "v13_R1_T_B_B2_s0_E16"}
N_CAP = 16384
ROLLOUT_N = 1024
MAX_UPDATES = 16
DEV10_N = 10
EVAL_POINTS = (0, 4096, 8192, 16384)
STAGE_DIR = Path("runs/v13_r1")
STATUS_PATH = Path("status/v13_r1.json")
DEV10_REL = Path("configs/splits/T_B_phase_a_v13_r1_dev10.json")
SPLIT_REL = Path("configs/splits/T_B_stage_2a_v11.json")
PLAN_VERSION = "1.3"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(value) + "\n", encoding="utf-8")


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
    v11.METHODS = METHODS
    v11.PLANNED = {(TASK, method): planned for method, planned in AUTHORIZED.items()}
    v11.ENABLED_SPLITS = dict(v11.ENABLED_SPLITS)
    v11.ENABLED_SPLITS[TASK] = DEV10_REL
    # Use the v12 persistence/profiling adapter without its T_C-specific gate.
    v11.save_ckpt = v12._save_ckpt_generation
    v11.start_episode = v12._start_episode_with_state
    v11.maybe_eval_at_n = v12._maybe_eval_at_points
    v11.select_checkpoint = v12._select_checkpoint
    v12._source_hashes = _source_hashes
    v12._fixed_mechanism_diagnostic = _fixed_mechanism_diagnostic
    v12._install_collector_profiler()


def _source_hashes(root: Path) -> dict:
    files = {
        "r1_runner": root / "src/cp_disr/phase_a_v13_r1.py",
        "stage2a_v11": root / "src/cp_disr/stage2a_v11.py",
        "collector": root / "src/cp_disr/collector.py",
        "torch_rl": root / "src/cp_disr/torch_rl.py",
        "neural": root / "src/cp_disr/neural.py",
        "persistence": root / "src/cp_disr/persistence.py",
        "runtime_factory": root / "src/cp_disr/platforms/libero/runtime_factory.py",
        "runtime_manifest": root / v11.RUNTIME_REL,
        "split": root / SPLIT_REL,
        "split_dev10": root / DEV10_REL,
    }
    return {k: sha(p) if p.is_file() else None for k, p in files.items()} | {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "base_commit": BASE_COMMIT,
    }


def _fixed_mechanism_diagnostic(root: Path, method: str, checkpoint: Path, label: str, device) -> None:
    """R1 relies on per-update production diagnostics; record the fixed scope."""
    job = Path(checkpoint).parent.parent
    path = job / "fixed_mechanism_scope.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(canonical({"label": label, "method": method, "checkpoint": str(checkpoint),
                           "effective_prior": 0, "test_id_used": False,
                           "used_for_training": False,
                           "note": "B1-K/B2 structure is audited in production transition/update diagnostics"}) + "\n")


def _job_dir(root: Path, method: str, stamp: str, configsha: str) -> Path:
    return root / STAGE_DIR / TASK / method / "seed_0" / f"{stamp}_{configsha}"


def _resolve_dev10(root: Path) -> tuple[Path, dict]:
    source = root / SPLIT_REL
    doc = json.loads(source.read_text(encoding="utf-8"))
    if len(doc.get("train") or []) != 64 or len(doc.get("dev") or []) != 20:
        raise BindingError("T_B source split must be train64/dev20")
    ranked = sorted(doc["dev"], key=lambda row: hashlib.sha256(
        ("cp_disr_v1.3_r1_dev10\0" + row["case_id"]).encode()).hexdigest())
    selected_ids = {r["case_id"] for r in ranked[:DEV10_N]}
    derived = dict(doc)
    derived["dev"] = [r for r in doc["dev"] if r["case_id"] in selected_ids]
    derived["dev_count"] = DEV10_N
    derived["test"] = []
    derived["test_count"] = 0
    derived["v13_r1"] = {
        "selection_namespace": "cp_disr_v1.3_r1_dev10",
        "selection_rule": "sha256_rank_then_preserve_source_order",
        "source_split": str(SPLIT_REL), "source_sha256": sha(source),
        "test_ids_accessed": False,
    }
    path = root / DEV10_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(derived, indent=2) + "\n", encoding="utf-8")
    return path, {
        "frozen_before_any_formal_result": True,
        "source_split": str(SPLIT_REL), "source_sha256": sha(source),
        "derived_split": str(DEV10_REL), "derived_sha256": sha(path),
        "dev20_order": [r["case_id"] for r in doc["dev"]],
        "dev10_order": [r["case_id"] for r in derived["dev"]],
        "test_ids_accessed": False,
    }


def _assert_startup(root: Path) -> dict:
    if root.resolve() != Path("/home/xushijie2/graph_cp_disr_v2_1").resolve():
        raise BindingError("unexpected repository path")
    if os.environ.get("PYTHONPATH", "").find("/home/xushijie2/graph_cp_disr_v2_1/src") < 0:
        raise BindingError("cp_disr import path is not the active repository")
    if subprocess.check_output([sys.executable, "-c", "import cp_disr; print(cp_disr.__file__)"], text=True).strip().find("/home/xushijie2/graph_cp_disr_v2_1/src") < 0:
        raise BindingError("cp_disr imported outside active repository")
    split = json.loads((root / SPLIT_REL).read_text(encoding="utf-8"))
    if split.get("task_id") != TASK or len(split.get("train") or []) != 64 or len(split.get("dev") or []) != 20:
        raise BindingError("T_B split identity/count mismatch")
    if any(r.get("second_role") != "second_object" for r in split["train"] + split["dev"]):
        raise BindingError("T_B is not bound to its double-target second object")
    if any(Path(p).exists() for p in root.glob("runs/v13_r1/**/checkpoints/*.pt")):
        raise BindingError("R1 output already contains a checkpoint")
    return {"python": sys.executable, "cp_disr_import": subprocess.check_output([sys.executable, "-c", "import cp_disr; print(cp_disr.__file__)"], text=True).strip(), "split_sha256": sha(root / SPLIT_REL), "test_ids": 0}


def freeze(root: Path, stamp: str | None = None) -> dict:
    root = Path(root).resolve(); os.chdir(root); configure()
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASE_COMMIT, current], cwd=root).returncode:
        raise BindingError(f"R1 source is not descended from {BASE_COMMIT}")
    startup = _assert_startup(root)
    dev10_path, dev10 = _resolve_dev10(root)
    prof = v11.bind_H(root)
    d_ref = float(prof["d_ref"][TASK]); tcap = N_CAP * d_ref
    config = {"plan_version": PLAN_VERSION, "task": TASK, "methods": list(METHODS),
              "authorized_planned_ids": list(AUTHORIZED.values()), "seed": 0,
              "Ncap": N_CAP, "Tcap": tcap, "H": float(prof["H"]), "d_ref": d_ref,
              "rollout": ROLLOUT_N, "max_complete_updates": MAX_UPDATES,
              "ppo_epochs": 4, "sequence_length": 16, "minibatch_target": 64,
              "actor_episode_discount_weight": False, "prior_mode": "absent",
              "B1-K": {"nominal_successor": False, "effective_prior": 0},
              "B2": {"contract_intervention": True, "effective_prior": 0, "DP": 0, "uP": 0, "Delta": 0},
              "test_id_used": False, "vlm_requests": 0,
              "authorization_documents": {
                  "plan": "CP_DISR_Experimental_Plan_v1.3_Existing_Task_First.md",
                  "agent_message": "MISSING_NOT_PROVIDED",
                  "r1_manifest": "MISSING_NOT_PROVIDED",
                  "resolution": "explicit user request and plan fields only; no missing values invented",
              },
              "training_source_commit": current, "runtime_revision": v12.RUNTIME_REVISION,
              "split": str(SPLIT_REL), "split_sha256": sha(root / SPLIT_REL),
              "dev10": str(dev10_path), "dev10_sha256": sha(dev10_path), "startup": startup}
    raw = canonical(config).encode(); configsha = hashlib.sha256(raw).hexdigest()[:8]
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pair = root / STAGE_DIR / stamp
    pair.mkdir(parents=True, exist_ok=False)
    write_json(pair / "plan_resolution.json", config)
    write_json(pair / "dev10_manifest.json", dev10)
    write_json(pair / "source_hashes.json", _source_hashes(root))
    write_json(pair / "pair_run_index.json", {"stamp": stamp, "configsha8": configsha, "jobs": {
        planned: {"method": method, "status": "NOT_STARTED", "run_dir": str(_job_dir(root, method, stamp, configsha))}
        for method, planned in AUTHORIZED.items()}})
    write_json(root / STAGE_DIR / "CURRENT.json", {"stamp": stamp, "configsha8": configsha})
    write_json(root / STATUS_PATH, {"phase_a_status": "NOT_STARTED", "phase_a_e16_ready": False,
        "task": TASK, "methods": list(METHODS), "authorized_jobs": list(AUTHORIZED.values()),
        "jobs": {p: "NOT_STARTED" for p in AUTHORIZED.values()}, "test_id_used": False, "vlm_requests": 0})
    return {"status": "FROZEN", "stamp": stamp, "configsha8": configsha, "config": config}


def _current(root: Path, stamp: str | None, configsha: str | None) -> tuple[str, str]:
    doc = json.loads((root / STAGE_DIR / "CURRENT.json").read_text())
    return stamp or doc["stamp"], configsha or doc["configsha8"]


def train(root: Path, method: str, gpu: int, stamp: str | None = None, configsha: str | None = None) -> dict:
    root = Path(root).resolve(); os.chdir(root); configure()
    if method not in METHODS: raise BindingError("R1 only permits B1-K and B2")
    stamp, configsha = _current(root, stamp, configsha)
    auth = json.loads((root / STAGE_DIR / stamp / "plan_resolution.json").read_text())
    if auth["training_source_commit"] != subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip():
        raise BindingError("production source changed after R1 freeze")
    import torch
    if not torch.cuda.is_available(): raise BindingError("CUDA required")
    torch.cuda.set_device(int(gpu)); device = torch.device("cuda", int(gpu))
    prof = v11.bind_H(root); prof["Ncap"] = N_CAP; prof["max_updates"] = MAX_UPDATES
    prof["Tcap"][TASK] = auth["Tcap"]
    hashes = _source_hashes(root)
    status_path = root / STATUS_PATH; status = json.loads(status_path.read_text())
    status["phase_a_status"] = "RUNNING"; status["jobs"][AUTHORIZED[method]] = "RUNNING"; status["active_method"] = method
    write_json(status_path, status)
    try:
        result = v11.train_job(root, TASK, method, device, torch.cuda.get_device_name(int(gpu)), prof, hashes, stamp, configsha, max_updates=MAX_UPDATES, resume=False)
    except Exception as exc:
        status = json.loads(status_path.read_text()); status.update({"phase_a_status": "NEEDS_RERUN", "phase_a_e16_ready": False, "hard_failure": {"method": method, "error": repr(exc), "time": now()}}); status["jobs"][AUTHORIZED[method]] = "NEEDS_RERUN"; write_json(status_path, status); raise
    status = json.loads(status_path.read_text()); status["jobs"][AUTHORIZED[method]] = "COMPLETE" if result.get("complete_updates") == MAX_UPDATES else "SAFE_BOUNDARY"; write_json(status_path, status)
    write_json(root / STAGE_DIR / stamp / f"job_T_B_{method}.json", result)
    return {"status": status["jobs"][AUTHORIZED[method]], "job": result, "stamp": stamp, "configsha8": configsha}


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f: return list(csv.DictReader(f))


def report_8192(root: Path, stamp: str | None = None, configsha: str | None = None) -> dict:
    root = Path(root).resolve(); stamp, configsha = _current(root, stamp, configsha)
    jobs = {}; issues = []
    for method in METHODS:
        job = _job_dir(root, method, stamp, configsha); rows = _read_csv(job / "eval_metrics.csv")
        ns = {int(float(r["skill_transitions"])) for r in rows}
        if not {0, 4096, 8192}.issubset(ns): issues.append(f"{method} missing common 0/4096/8192 dev10")
        jobs[method] = rows
    payload = {"status": "PASS" if not issues else "NEEDS_RERUN", "issues": issues, "jobs": jobs}
    pair = root / STAGE_DIR / stamp; write_json(pair / "first_comparison.json", payload)
    lines = ["# v1.3 R1 first comparison", "", f"Status: {payload['status']}", ""]
    for method, rows in jobs.items():
        lines += [f"## {method}", ""]
        for r in rows: lines.append(f"- N={r.get('skill_transitions')}: return={r.get('mean_discounted_return')}, success={r.get('success_n')}/10")
        lines.append("")
    (pair / "first_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def finalize(root: Path, stamp: str | None = None, configsha: str | None = None) -> dict:
    root = Path(root).resolve(); stamp, configsha = _current(root, stamp, configsha)
    split = json.loads((root / SPLIT_REL).read_text()); index = {r["case_id"]: r for r in split["train"] + split["dev"]}
    selected = {}; common = []
    import torch
    device = torch.device("cuda", 5 if torch.cuda.is_available() else "cpu")
    for method in METHODS:
        job = _job_dir(root, method, stamp, configsha); rows = _read_csv(job / "eval_metrics.csv")
        ckpt, selection = v12._select_checkpoint(rows)
        if not ckpt: raise BindingError(f"no checkpoint selection for {method}")
        selected[method] = {"checkpoint": ckpt, "sha256": sha(Path(ckpt)), "selection": selection}
        ev = v11.eval_episodes(root, TASK, method, Path(ckpt), [r["case_id"] for r in split["dev"]], index, device, _source_hashes(root), job / "common_dev20.json", n_episodes=20, label="common_dev20")
        common.append({"method": method, "checkpoint": ckpt, "sha256": sha(Path(ckpt)), "success_n": ev["success_n"], "success_rate": ev["success_rate"], "mean_discounted_return": ev["mean_discounted_return"]})
    pair = root / STAGE_DIR / stamp; write_json(pair / "checkpoint_index.json", {"frozen_before_common_dev20": True, "selected": selected})
    with (pair / "final_dev_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(common[0])); w.writeheader(); w.writerows(common)
    status = json.loads((root / STATUS_PATH).read_text()); status.update({"phase_a_status": "COMPLETE", "phase_a_e16_ready": False, "completed_at": now(), "formal_rl_jobs": 2, "real_environment_ppo_updates": 32, "test_id_used": False, "vlm_requests": 0}); write_json(root / STATUS_PATH, status)
    return {"status": "COMPLETE", "selected": selected, "common_dev20": common, "formal_rl_jobs": 2, "real_environment_ppo_updates": 32, "test_id_used": False, "vlm_requests": 0}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["freeze", "train", "report-8192", "finalize"]); ap.add_argument("--method", choices=METHODS); ap.add_argument("--gpu", type=int, default=5); ap.add_argument("--stamp"); ap.add_argument("--configsha")
    a = ap.parse_args(argv); root = Path.cwd()
    if a.phase == "freeze": out = freeze(root, a.stamp)
    elif a.phase == "train": out = train(root, a.method, a.gpu, a.stamp, a.configsha)
    elif a.phase == "report-8192": out = report_8192(root, a.stamp, a.configsha)
    else: out = finalize(root, a.stamp, a.configsha)
    print(canonical(out)); return 0


if __name__ == "__main__": raise SystemExit(main())
