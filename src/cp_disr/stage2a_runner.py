"""Stage 2A 24-job orchestrator. P0 validates registry/resume/probe; no PPO update."""
from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

from .common import BindingError, canonical, digest

TASKS = ("T_A", "T_C")
METHODS = ("B0", "B1", "B2", "Full")
SEEDS = (0, 1, 2)
TRANSITIONS = 65536
UPDATES = 64
UPDATE_EVERY = 1024
EVAL_EVERY_UPDATES = 8
CHECKPOINTS = (0, 8, 16, 24, 32, 40, 48, 56, 64)
P0 = Path("experiments/part_2_exploration/stage_2a_p0")


def job_spec(task_id: str, method: str, seed: int) -> dict:
    run_id = f"stage2a_{task_id}_{method}_seed{seed}"
    out = f"experiments/part_2_exploration/stage_2a/{run_id}"
    rec = {
        "run_id": run_id,
        "task_id": task_id,
        "method": method,
        "seed": int(seed),
        "init": "from_scratch",
        "planned_transitions": TRANSITIONS,
        "planned_updates": UPDATES,
        "update_every_transitions": UPDATE_EVERY,
        "eval_split": "dev20",
        "eval_at_updates": "0,8,16,24,32,40,48,56,64",
        "checkpoint_updates": "0,8,16,24,32,40,48,56,64",
        "b1_full_train_prior": "80/20" if method in ("B1", "Full") else "empty",
        "eval_prior": "Original" if method in ("B1", "Full") else "empty",
        "effective_prior_train": "empty" if method in ("B0", "B2") else "80/20",
        "test_access": False,
        "share_optimizer": False,
        "share_checkpoint": False,
        "stop_on_full_win": False,
        "output_dir": out,
        "attempt_dir": f"{out}/attempts/attempt_000",
        "status": "PLANNED",
        "learning_enabled": True,
        "p0_execute": False,
    }
    rec["config_hash"] = digest({k: rec[k] for k in rec if k not in ("status",)})
    return rec


def planned_jobs() -> list[dict]:
    rows = [job_spec(t, m, s) for t in TASKS for m in METHODS for s in SEEDS]
    ids = [r["run_id"] for r in rows]
    hashes = [r["config_hash"] for r in rows]
    dirs = [r["output_dir"] for r in rows]
    if len(rows) != 24:
        raise BindingError("Stage 2A registry must contain 24 jobs")
    if len(set(ids)) != 24 or len(set(hashes)) != 24 or len(set(dirs)) != 24:
        raise BindingError("Stage 2A jobs are not unique")
    return rows


def write_registry(root: Path) -> dict:
    rows = planned_jobs()
    out = root / P0
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "planned_run_registry.csv"
    json_path = out / "planned_run_registry.json"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json_path.write_text(canonical({"count": len(rows), "jobs": rows}) + "\n", encoding="utf-8")
    resource = {
        "max_parallel_jobs": 2,
        "gpu_policy": "exclusive_per_job",
        "per_run_transition_cap": TRANSITIONS,
        "per_run_update_cap": UPDATES,
        "crash_recovery": "resume_from_last_complete_checkpoint_without_overwriting_attempt",
        "attempt_policy": "never_overwrite_existing_attempt_dir",
        "rng_resume": True,
        "optimizer_resume": True,
        "prior_resume": True,
        "case_scheduler_resume": True,
        "raw_log_archive": True,
        "p0_learning": False,
        "note": "Formal Stage 2A jobs are not started in P0.",
    }
    (out / "resource_plan.yaml").write_text(
        __import__("yaml").safe_dump(resource, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return {"count": 24, "csv": str(csv_path.relative_to(root)), "json": str(json_path.relative_to(root)), "jobs": rows, "resource": resource}


def _atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(canonical(obj) + "\n", encoding="utf-8")
    tmp.replace(path)


class Stage2ARunner:
    """Registry/state-machine runner. Learning is opt-in and disabled for P0."""

    def __init__(self, root: Path, jobs=None, learning_enabled: bool = False):
        self.root = Path(root)
        self.jobs = {r["run_id"]: deepcopy(r) for r in (jobs or planned_jobs())}
        self.learning_enabled = bool(learning_enabled)
        self.optimizer_steps = 0
        self.states = {}
        for run_id, job in self.jobs.items():
            self.states[run_id] = {
                "status": "PLANNED",
                "rng": {"python": job["seed"], "numpy": job["seed"] + 17, "torch": job["seed"] + 31},
                "case_scheduler": {"index": 0, "split": "train"},
                "prior": {"mode": job["effective_prior_train"]},
                "optimizer": {"steps": 0, "state": None},
                "transitions": 0,
                "updates": 0,
                "attempt": 0,
            }

    def output_dir(self, run_id: str) -> Path:
        if not self.learning_enabled:
            return self.root / P0 / "runner_probe" / run_id
        return self.root / self.jobs[run_id]["output_dir"]

    def attempt_dir(self, run_id: str, attempt: int) -> Path:
        return self.output_dir(run_id) / "attempts" / f"attempt_{attempt:03d}"

    def start(self, run_id: str) -> dict:
        if run_id not in self.jobs:
            raise BindingError("unknown run_id " + run_id)
        st = self.states[run_id]
        dest = self.attempt_dir(run_id, st["attempt"])
        if dest.exists():
            raise BindingError("refusing to overwrite existing attempt " + str(dest))
        dest.mkdir(parents=True, exist_ok=False)
        st["status"] = "STARTED"
        rec = {"run_id": run_id, "job": self.jobs[run_id], "state": st, "learning_enabled": self.learning_enabled}
        _atomic_json(dest / "run_state.json", rec)
        return rec

    def save_resume(self, run_id: str) -> dict:
        st = self.states[run_id]
        payload = {
            "run_id": run_id,
            "rng": deepcopy(st["rng"]),
            "optimizer": deepcopy(st["optimizer"]),
            "prior": deepcopy(st["prior"]),
            "case_scheduler": deepcopy(st["case_scheduler"]),
            "transitions": st["transitions"],
            "updates": st["updates"],
            "attempt": st["attempt"],
            "status": st["status"],
        }
        dest = self.attempt_dir(run_id, st["attempt"])
        dest.mkdir(parents=True, exist_ok=True)
        _atomic_json(dest / "resume.json", payload)
        return payload

    def load_resume(self, run_id: str) -> dict:
        st = self.states[run_id]
        path = self.attempt_dir(run_id, st["attempt"]) / "resume.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        st["rng"] = payload["rng"]
        st["optimizer"] = payload["optimizer"]
        st["prior"] = payload["prior"]
        st["case_scheduler"] = payload["case_scheduler"]
        st["transitions"] = payload["transitions"]
        st["updates"] = payload["updates"]
        st["status"] = "RESUMED"
        return payload

    def ppo_update(self, run_id: str, rollout=None):
        if not self.learning_enabled:
            raise BindingError("P0 runner forbids PPO optimizer updates")
        self.states[run_id]["optimizer"]["steps"] += 1
        self.optimizer_steps += 1
        self.states[run_id]["updates"] += 1
        return {"steps": self.optimizer_steps}

    def no_learning_probe(self, run_id: str | None = None) -> dict:
        run_id = run_id or next(iter(self.jobs))
        while self.attempt_dir(run_id, self.states[run_id]["attempt"]).exists():
            self.states[run_id]["attempt"] += 1
        started = self.start(run_id)
        saved = self.save_resume(run_id)
        loaded = self.load_resume(run_id)
        if loaded["rng"] != saved["rng"] or loaded["optimizer"] != saved["optimizer"]:
            raise BindingError("resume mismatch")
        if loaded["prior"] != saved["prior"] or loaded["case_scheduler"] != saved["case_scheduler"]:
            raise BindingError("resume scheduler/prior mismatch")
        blocked = False
        err = None
        try:
            self.ppo_update(run_id)
        except BindingError as exc:
            blocked = True
            err = str(exc)
        return {
            "run_id": run_id,
            "started": True,
            "resume_ok": True,
            "optimizer_steps": self.optimizer_steps,
            "learning_enabled": self.learning_enabled,
            "ppo_update_blocked": blocked,
            "error": err,
            "attempt_dir": str(self.attempt_dir(run_id, self.states[run_id]["attempt"])),
        }


def validate_runner(root: Path) -> dict:
    root = Path(root)
    written = write_registry(root)
    runner = Stage2ARunner(root, written["jobs"], learning_enabled=False)
    probe = runner.no_learning_probe(written["jobs"][0]["run_id"])
    unique = len({j["run_id"] for j in written["jobs"]}) == 24
    isolated = len({j["output_dir"] for j in written["jobs"]}) == 24
    report = {
        "registry_count": 24,
        "unique_run_ids": unique,
        "isolated_output_dirs": isolated,
        "unique_config_hashes": len({j["config_hash"] for j in written["jobs"]}) == 24,
        "resume_unit_ok": probe["resume_ok"],
        "no_learning_probe_ok": probe["optimizer_steps"] == 0 and probe["ppo_update_blocked"],
        "optimizer_steps": probe["optimizer_steps"],
        "probe": probe,
    }
    md = root / P0 / "stage_2a_runner_validation.md"
    md.write_text(
        "# Stage 2A runner validation\n\n"
        + json.dumps(report, indent=2, default=str)
        + "\n\nNo PPO update was executed.\n",
        encoding="utf-8",
    )
    report["passed"] = all(
        [
            report["unique_run_ids"],
            report["isolated_output_dirs"],
            report["unique_config_hashes"],
            report["resume_unit_ok"],
            report["no_learning_probe_ok"],
        ]
    )
    return report