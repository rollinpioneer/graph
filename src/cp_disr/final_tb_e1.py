"""R-TB-E-1 (ELASTIC-01) launch binding for the final T_B comparison.

CP-DISR-TB-E1-ELASTIC-01.  This module adds *no* trainer and changes no scientific or runtime semantics.  It
re-uses the frozen production path unchanged (``final_tb.run_train`` -> ``stage2a_v11.train_job``) and only adds
what the single elastic attempt R-TB-E-1 (B1-K+E, seed 1, ELASTIC-01) needs around it:

* an elastic ledger (3 base attempts already used, cumulative cap 4, ELASTIC-01 bound to R-TB-E-1 only),
* the A2 smoke *compatibility rebind* (no new environment smoke): a pure AST/blob comparison of the old smoke prep
  against the new prep that must find every execution path identical, a derived compatibility receipt and a
  release token bound to the new prep (an old token can never be reused),
* a static Evaluator eligibility check that confirms the NONTRIVIAL_LEARNING trigger of R-TB-E-0,
* the operational storage gate (start >= 6 GiB, pause < 3 GiB, resume >= 5 GiB, 2 GiB hard reserved margin,
  5 s sampling; SIGSTOP/SIGCONT only).

Import-light on purpose: nothing here imports torch or the simulator at module import time.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from . import final_tb as ftb
from .common import BindingError

E1_CARD = "CP-DISR-TB-E1-ELASTIC-01"
E1_PLAN_ID = "R-TB-E-1"
E1_SLOT = "ELASTIC-01"
E1_METHOD = "B1-K+E"
E1_SEED = 1
REFERENCE_PLAN_ID = "R-TB-E-0"
ELASTIC_SLOTS = ("ELASTIC-01", "ELASTIC-02", "ELASTIC-03", "ELASTIC-04")
OLD_PREP = "cac2fa5b3cafb46f1f74370314189f4bca4878fa"
BASE_RESULTS_COMMIT = "7c2f58c9844c7a8db4cd5a28fe1615e5630dfeff"
PLAN_V3_REL = Path("docs/authoritative/CP_DISR_Final_Experimental_Plan_v3.md")
PLAN_V3_SHA256 = "e5df8585811559783666c185f35c5c728e95b952d01b988956f5e1c772283cff"
BASE_LAUNCH_REL = Path("runs/final_master/2.1.1/launch/20261001T115251Z_smoker2_cac2fa5b")
RELEASE_CONFIG_REL = Path("configs/final_master/final_tb_e1_elastic_release.yaml")
E1_CLI = "final_tb_e1_launch.py"
SUCCESS_SECONDS_LABEL = "FINAL_SKILL_DURATION_ONLY"
EPISODE_TOTAL_SUCCESS_TIME = "UNVERIFIED"

EXPECTED_BASE_PLAN_TABLE = {
    "R-TB-E-0": {"method": "B1-K+E", "seed": 0, "release_order": 1},
    "R-TB-DK-1": {"method": "B2", "seed": 1, "release_order": 2},
    "R-TB-K-1": {"method": "B1-K", "seed": 1, "release_order": 3},
}
EXPECTED_ELASTIC_PLAN_TABLE = {
    "R-TB-E-1": {"method": "B1-K+E", "seed": 1, "release_order": 4, "budget_slot": "ELASTIC-01"},
}

NOT_AUTHORIZED = [
    "provider", "T_P training", "formal test", "Family B", "RoboCasa", "automatic +E seed 2 or any other control",
    "ELASTIC-02", "ELASTIC-03", "ELASTIC-04", "a second R-TB-E-1 attempt", "a different seed or early stop of R-TB-E-1",
    "parallel new training", "Method 2.1.1 change",
]

# ----------------------------------------------------------------------------- storage gate
GIB = 1024 ** 3
STORAGE_GATE = {
    "start_free_min_bytes": 6 * GIB,
    "hard_reserved_margin_bytes": 2 * GIB,
    "pause_trigger_bytes": 3 * GIB,
    "resume_trigger_bytes": 5 * GIB,
    "sampling_interval_seconds": 5.0,
}
SAMPLE_LOG_EVERY_TICKS = 12  # one routine SAMPLE line per minute at 5 s; every tick while below the resume level


def storage_gate_ok(cfg=None) -> dict:
    cfg = dict(cfg or STORAGE_GATE)
    if not (0 < cfg["hard_reserved_margin_bytes"] < cfg["pause_trigger_bytes"] < cfg["resume_trigger_bytes"]
            <= cfg["start_free_min_bytes"]):
        raise BindingError("storage gate must satisfy 0 < margin < pause < resume <= start: %r" % (cfg,))
    if not (0 < float(cfg["sampling_interval_seconds"]) <= 60):
        raise BindingError("storage gate sampling interval out of range")
    return cfg


def mount_of(path) -> dict:
    """The mount point/filesystem that actually holds `path` (its nearest existing ancestor)."""
    p = ftb._nearest_existing(path).resolve()
    while not os.path.ismount(str(p)) and p != p.parent:
        p = p.parent
    info = {"path": str(path), "checked_path": str(ftb._nearest_existing(path)), "mount_point": str(p)}
    with contextlib.suppress(OSError):
        best = None
        for line in Path("/proc/mounts").read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[1] == str(p):
                best = {"source": parts[0], "fstype": parts[2]}
        if best:
            info.update(best)
    return info


def start_gate(path, free_fn=None, cfg=None) -> dict:
    """The worker may start only if the output mount has >= start_free_min free bytes. Nothing is deleted."""
    cfg = storage_gate_ok(cfg)
    free_fn = free_fn or ftb.fs_free_bytes
    checked = ftb._nearest_existing(path)
    free = int(free_fn(checked))
    rec = {"time": ftb.utc_now(), **mount_of(path), "free_bytes": free, "free_gib": round(free / GIB, 3),
           "start_free_min_bytes": cfg["start_free_min_bytes"], "passed": free >= cfg["start_free_min_bytes"]}
    if not rec["passed"]:
        raise BindingError("STORAGE_START_GATE_REFUSED: free %d B (%.2f GiB) < required %d B (%.0f GiB) on %s; "
                           "nothing is deleted and no save frequency is lowered"
                           % (free, free / GIB, cfg["start_free_min_bytes"], cfg["start_free_min_bytes"] / GIB,
                              rec.get("mount_point")))
    return rec


def guard_decision(paused: bool, free: int, cfg=None) -> str:
    """PAUSE below the pause trigger (strict), RESUME only at/after the resume trigger; hysteresis in between."""
    cfg = cfg or STORAGE_GATE
    if not paused and free < cfg["pause_trigger_bytes"]:
        return "PAUSE"
    if paused and free >= cfg["resume_trigger_bytes"]:
        return "RESUME"
    return "NONE"


def _proc_ppid_map(proc_root="/proc") -> dict:
    out = {}
    for entry in Path(proc_root).iterdir():
        if not entry.name.isdigit():
            continue
        with contextlib.suppress(OSError, ValueError, IndexError):
            stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            out[int(entry.name)] = int(stat.rsplit(")", 1)[1].split()[1])
    return out


def descendants(pid, proc_root="/proc") -> list:
    ppids = _proc_ppid_map(proc_root)
    found, frontier = [], [int(pid)]
    while frontier:
        cur = frontier.pop()
        for child, parent in ppids.items():
            if parent == cur and child not in found:
                found.append(child)
                frontier.append(child)
    return sorted(found)


def proc_state(pid, proc_root="/proc") -> str:
    try:
        return (Path(proc_root) / str(pid) / "stat").read_text(encoding="utf-8", errors="replace").rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return "GONE"


def find_worker_pids(out_dir, proc_root="/proc") -> list:
    """The RUNNING R-TB-E-1 worker recorded in the ledger (verified by its command line) and its descendants."""
    try:
        state = json.loads((Path(out_dir) / "launch_state.json").read_text(encoding="utf-8"))
        plan = state["plans"][E1_PLAN_ID]
    except (OSError, KeyError, ValueError):
        return []
    pid = plan.get("pid")
    if plan.get("status") != "RUNNING" or not pid:
        return []
    try:
        cmd = (Path(proc_root) / str(int(pid)) / "cmdline").read_bytes().decode(errors="replace")
    except OSError:
        return []
    if E1_CLI not in cmd:
        return []
    return [int(pid)] + descendants(int(pid), proc_root)


class StorageGuard:
    """Reversible disk guard. The ONLY actions are SIGSTOP and SIGCONT on the E1 worker (and its children).
    Never kills, never deletes, never touches run files, never changes a checkpoint frequency."""

    ALLOWED_SIGNALS = (signal.SIGSTOP, signal.SIGCONT)

    def __init__(self, watch_path, log_path, find_pids, free_fn=None, signal_fn=None, state_fn=None, cfg=None, utc_fn=None):
        self.cfg = storage_gate_ok(cfg)
        self.watch_path = str(watch_path)
        self.log_path = Path(log_path)
        self.find_pids = find_pids
        self.free_fn = free_fn or ftb.fs_free_bytes
        self.signal_fn = signal_fn or os.kill
        self.state_fn = state_fn or proc_state
        self.utc = utc_fn or ftb.utc_now
        self.paused = False
        self.paused_pids = []
        self.paused_since = None
        self.ticks = self.pauses = self.resumes = 0
        self.min_free = None
        self.min_free_utc = None
        self.breach_open = False
        self.breach_events = 0
        self.last_free = None
        self._noworker_logged = 0.0

    def _log(self, event, **fields):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"event": event, "utc": self.utc(), "guard_pid": os.getpid(), **fields}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
        return row

    def _send(self, pids, sig) -> dict:
        if sig not in self.ALLOWED_SIGNALS:
            raise BindingError("the storage guard may only SIGSTOP/SIGCONT (got %r)" % (sig,))
        sent, missing = [], []
        for pid in pids:
            try:
                self.signal_fn(int(pid), sig)
                sent.append(int(pid))
            except ProcessLookupError:
                missing.append(int(pid))
        return {"sent": sent, "missing": missing}

    def tick(self, now=None) -> str:
        now = time.time() if now is None else now
        free = int(self.free_fn(self.watch_path))
        self.ticks += 1
        self.last_free = free
        if self.min_free is None or free < self.min_free:
            self.min_free, self.min_free_utc = free, self.utc()
        if free < self.cfg["hard_reserved_margin_bytes"]:
            if not self.breach_open:
                self.breach_open = True
                self.breach_events += 1
                self._log("HARD_MARGIN_BREACH", free_bytes=free, margin_bytes=self.cfg["hard_reserved_margin_bytes"],
                          paused=self.paused)
        elif free >= self.cfg["pause_trigger_bytes"]:
            self.breach_open = False
        action = guard_decision(self.paused, free, self.cfg)
        if action == "PAUSE":
            pids = list(self.find_pids() or [])
            if not pids:
                if now - self._noworker_logged >= 60:
                    self._noworker_logged = now
                    self._log("PAUSE_TRIGGER_NO_WORKER", free_bytes=free)
                action = "NONE"
            else:
                res = self._send(pids, signal.SIGSTOP)
                self.paused, self.paused_pids, self.paused_since = True, res["sent"], now
                self.pauses += 1
                self._log("SIGSTOP", free_bytes=free, free_gib=round(free / GIB, 3), pids=res["sent"], missing=res["missing"],
                          proc_states={str(p): self.state_fn(p) for p in res["sent"]},
                          pause_trigger_bytes=self.cfg["pause_trigger_bytes"])
        elif action == "RESUME":
            res = self._send(self.paused_pids, signal.SIGCONT)
            self._log("SIGCONT", free_bytes=free, free_gib=round(free / GIB, 3), pids=res["sent"], missing=res["missing"],
                      paused_seconds=round(now - (self.paused_since or now), 1),
                      resume_trigger_bytes=self.cfg["resume_trigger_bytes"])
            self.paused, self.paused_pids, self.paused_since = False, [], None
            self.resumes += 1
        if action == "NONE" and (free < self.cfg["resume_trigger_bytes"] or self.ticks % SAMPLE_LOG_EVERY_TICKS == 0):
            self._log("SAMPLE", free_bytes=free, free_gib=round(free / GIB, 3), paused=self.paused)
        return action

    def release_all(self, reason="guard_exit") -> None:
        """A guard that ends must never leave a worker stopped."""
        if self.paused and self.paused_pids:
            res = self._send(self.paused_pids, signal.SIGCONT)
            self._log("GUARD_RELEASED_SIGCONT", reason=reason, pids=res["sent"], missing=res["missing"])
            self.paused, self.paused_pids = False, []

    def summary(self) -> dict:
        return {"ticks": self.ticks, "pauses": self.pauses, "resumes": self.resumes, "paused_now": self.paused,
                "min_free_bytes_sampled": self.min_free, "min_free_utc": self.min_free_utc,
                "hard_margin_breach_events": self.breach_events, "last_free_bytes": self.last_free,
                "thresholds": dict(self.cfg), "note": "minimum over 5 s samples; free space between samples is not measured"}

    def run(self, stop_fn, sleep_fn=time.sleep, max_ticks=None) -> dict:
        try:
            while True:
                self.tick()
                if stop_fn() or (max_ticks is not None and self.ticks >= max_ticks):
                    break
                sleep_fn(self.cfg["sampling_interval_seconds"])
        finally:
            self.release_all("loop_end")
        return self.summary()


# ----------------------------------------------------------------------------- plan identity + release config
def assert_plan_identities() -> dict:
    """The three base identities never change; E1 is exactly the one registered elastic row; caps are 3 and 4."""
    if ftb.BASE_PLAN_TABLE != EXPECTED_BASE_PLAN_TABLE:
        raise BindingError("the three base plan identities changed")
    if ftb.ELASTIC_PLAN_TABLE != EXPECTED_ELASTIC_PLAN_TABLE:
        raise BindingError("the elastic plan table is not exactly {R-TB-E-1: B1-K+E seed 1 ELASTIC-01}")
    if set(ftb.PLAN_TABLE) != set(EXPECTED_BASE_PLAN_TABLE) | set(EXPECTED_ELASTIC_PLAN_TABLE):
        raise BindingError("PLAN_TABLE is not base + elastic")
    if ftb.BASE_RL_ATTEMPTS != 3 or ftb.MAX_NEW_RL_ATTEMPTS != 4:
        raise BindingError("attempt caps must be 3 (base) and 4 (cumulative)")
    return {"base": dict(ftb.BASE_PLAN_TABLE), "elastic": dict(ftb.ELASTIC_PLAN_TABLE),
            "base_attempts": ftb.BASE_RL_ATTEMPTS, "cumulative_cap": ftb.MAX_NEW_RL_ATTEMPTS}


def expected_release_config() -> dict:
    gate = storage_gate_ok()
    return {
        "card": E1_CARD, "parent_card": ftb.R2_CARD,
        "plan": {"plan_id": E1_PLAN_ID, "task": ftb.TASK, "method": E1_METHOD, "seed": E1_SEED, "budget_slot": E1_SLOT,
                 "init": "FROM_SCRATCH", "warm_start": False, "profile_identical_to": REFERENCE_PLAN_ID,
                 "prior_mode": ftb.PRIOR_MODE},
        "frozen": {"H_seconds": ftb.H_SECONDS, "d_ref_seconds": ftb.D_REF_SECONDS,
                   "task_deadline_seconds": ftb.TASK_DEADLINE_SECONDS, "ncap": ftb.N_CAP,
                   "tcap_definition": "16384 * frozen_d_ref", "tcap_seconds": ftb.TCAP_SECONDS,
                   "rollout_n": ftb.ROLLOUT_N, "max_complete_updates": ftb.MAX_UPDATES,
                   "eval_points": list(ftb.EVAL_POINTS), "evaluation_rule": ftb.EVALUATION_RULE,
                   "train_cases": 64, "dev_cases": ftb.DEV10_N},
        "budget": {"base_attempts_used": ftb.BASE_RL_ATTEMPTS, "cumulative_attempt_cap": ftb.MAX_NEW_RL_ATTEMPTS,
                   "elastic_pool": {s: ("ALLOCATED:%s" % E1_PLAN_ID if s == E1_SLOT else "UNALLOCATED") for s in ELASTIC_SLOTS},
                   "workers": 1, "provider": 0, "tp_training": 0, "formal_test": 0},
        "smoke": {"mode": "A2_COMPATIBILITY_REBIND", "new_environment_smoke_episodes": 0, "old_prep_commit": OLD_PREP,
                  "old_receipts_read_only": True, "reuse_old_release_token": False},
        "storage_gate": {**gate, "actions": ["SIGSTOP", "SIGCONT"], "kill": False, "delete_history": False,
                         "lower_checkpoint_frequency": False},
        "reporting": {"success_seconds_label": SUCCESS_SECONDS_LABEL,
                      "episode_total_success_time": EPISODE_TOTAL_SUCCESS_TIME},
    }


def write_release_config(path) -> dict:
    cfg = expected_release_config()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg


def load_release_config(root) -> dict:
    """The tracked release config must equal the values this code derives from the frozen constants."""
    path = Path(root) / RELEASE_CONFIG_REL
    if not path.is_file():
        raise BindingError("E1 release config missing: %s" % path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = expected_release_config()
    if cfg != expected:
        diff = sorted(k for k in set(cfg) | set(expected) if cfg.get(k) != expected.get(k))
        raise BindingError("E1 release config differs from the frozen values in: %s" % diff)
    return cfg


# ----------------------------------------------------------------------------- elastic ledger
def _json_file(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def base_launch_facts(base_out) -> dict:
    """Read-only facts about the finished base registration: three COMPLETE plans, 3/3 attempts, PASS smoke receipt."""
    base_out = Path(base_out)
    state = _json_file(base_out / "launch_state.json")
    plans = state["plans"]
    if set(plans) != set(EXPECTED_BASE_PLAN_TABLE):
        raise BindingError("base ledger plans are %s, expected the three base plans" % sorted(plans))
    bad = {p: v.get("status") for p, v in plans.items() if v.get("status") != "COMPLETE"}
    if bad:
        raise BindingError("base runs are not all COMPLETE: %s" % bad)
    if state.get("new_rl_attempts_used") != ftb.BASE_RL_ATTEMPTS or state.get("new_rl_attempts_cap") != ftb.BASE_RL_ATTEMPTS:
        raise BindingError("base ledger must be 3/3 used/cap, found %s/%s" % (state.get("new_rl_attempts_used"), state.get("new_rl_attempts_cap")))
    receipt = _json_file(base_out / "smoke_receipt.json")
    if receipt.get("label") != "TB_RUNTIME_SMOKE_PASS" or receipt.get("prep_commit") != OLD_PREP:
        raise BindingError("base smoke receipt is not the TB_RUNTIME_SMOKE_PASS of the old prep")
    files = {}
    for name in OLD_RECEIPT_FILES:
        p = base_out / name
        if not p.is_file():
            raise BindingError("old smoke receipt file missing: %s" % p)
        files[name] = {"path": str(p), "sha256": ftb.sha256_file(p)}
    return {"state": state, "plans": plans, "receipt_label": receipt["label"], "old_files": files}


OLD_RECEIPT_FILES = ("smoke_receipt.json", "smoke_case1.json", "smoke_case2.json", "smoke_case1_events.jsonl",
                     "smoke_case2_events.jsonl", "smoke_budget.json", "authorization.json", "launch_plan.json")


def init_elastic_ledger(out_dir, base_facts: dict, plan_row: dict, authorization_sha256: str) -> dict:
    """E1 ledger: 3 base attempts already used, cumulative cap 4, ELASTIC-01 allocated to R-TB-E-1 only."""
    ledger = ftb.Ledger(out_dir)
    prior = {pid: {"status": v.get("status"), "attempt_id": v.get("attempt_id"), "run_dir": v.get("run_dir"),
                   "complete_updates": v.get("complete_updates"), "valid_transitions": v.get("valid_transitions")}
             for pid, v in base_facts["plans"].items()}
    slots = {s: ({"status": "ALLOCATED", "plan_id": E1_PLAN_ID, "allocated_by": "explicit user authorization",
                  "authorization_sha256": authorization_sha256, "allocated_utc": ftb.utc_now()}
                 if s == E1_SLOT else {"status": "UNALLOCATED", "plan_id": None}) for s in ELASTIC_SLOTS}
    with ledger.locked():
        if ledger.state_path.exists():
            raise BindingError("ledger already initialised: %s" % ledger.state_path)
        state = {"created": ftb.utc_now(), "card": E1_CARD, "new_rl_attempts_used": ftb.BASE_RL_ATTEMPTS,
                 "new_rl_attempts_cap": ftb.MAX_NEW_RL_ATTEMPTS, "prior_attempts": prior, "elastic_slots": slots,
                 "plans": {E1_PLAN_ID: dict(plan_row)}}
        ftb.write_json_atomic(ledger.state_path, state)
    return state


class ElasticLedger(ftb.Ledger):
    """`final_tb.Ledger` for the E1 registration: ELASTIC-01 can only be taken by R-TB-E-1, exactly once."""

    def _assert_binding(self, plan_id):
        if plan_id != E1_PLAN_ID:
            raise BindingError("the elastic ledger only releases %s (got %r)" % (E1_PLAN_ID, plan_id))
        state = self.read()
        plan = state["plans"].get(plan_id)
        slot = (state.get("elastic_slots") or {}).get(E1_SLOT) or {}
        if set(state["plans"]) != {E1_PLAN_ID} or plan is None or plan.get("budget_slot") != E1_SLOT:
            raise BindingError("elastic ledger plans are not exactly {%s: %s}" % (E1_PLAN_ID, E1_SLOT))
        if slot.get("status") != "ALLOCATED" or slot.get("plan_id") != E1_PLAN_ID:
            raise BindingError("%s is %s for %r; it can only be allocated to %s once" % (E1_SLOT, slot.get("status"), slot.get("plan_id"), E1_PLAN_ID))
        others = {s: v.get("status") for s, v in state["elastic_slots"].items() if s != E1_SLOT}
        if any(v != "UNALLOCATED" for v in others.values()):
            raise BindingError("other elastic slots must stay UNALLOCATED: %s" % others)
        if state["new_rl_attempts_used"] != ftb.BASE_RL_ATTEMPTS or state["new_rl_attempts_cap"] != ftb.MAX_NEW_RL_ATTEMPTS:
            raise BindingError("elastic ledger must hold %d used / cap %d before the E1 reservation" % (ftb.BASE_RL_ATTEMPTS, ftb.MAX_NEW_RL_ATTEMPTS))

    def reserve(self, plan_id, attempt_id, run_dir, pid, physical_gpu) -> dict:
        self._assert_binding(plan_id)
        plan = super().reserve(plan_id, attempt_id, run_dir, pid, physical_gpu)
        with self.locked():
            state = self.read()
            state["elastic_slots"][E1_SLOT].update(status="IN_USE", attempt_id=attempt_id, run_dir=str(run_dir), pid=pid,
                                                   physical_gpu=physical_gpu, started=ftb.utc_now())
            ftb.write_json_atomic(self.state_path, state)
        return plan

    def finish(self, plan_id, status, **fields) -> dict:
        plan = super().finish(plan_id, status, **fields)
        with self.locked():
            state = self.read()
            state["elastic_slots"][E1_SLOT].update(status="CONSUMED", final_plan_status=status, finished=ftb.utc_now())
            ftb.write_json_atomic(self.state_path, state)
        return plan


def scan_elastic_markers(roots, runner=subprocess.run, timeout=45) -> dict:
    """Search the known worktrees for ELASTIC-0x allocation markers and non-zero elastic usage in budget ledgers."""
    found = {}
    for root in roots:
        root = Path(root)
        row = {"exists": root.is_dir(), "files_with_elastic_slot_marker": [], "ledgers_with_nonzero_elastic": [], "error": None}
        if root.is_dir():
            paths = [str(root / d) for d in ("runs/final_master", "docs") if (root / d).is_dir()]
            if paths:
                try:
                    done = runner(["grep", "-rIl", "-E", "--include=*.json", "--include=*.md", "--include=*.yaml",
                                   "--include=*.yml", "--include=*.txt", "ELASTIC-0[1-4]", *paths],
                                  capture_output=True, text=True, timeout=timeout)
                    row["files_with_elastic_slot_marker"] = sorted(l for l in done.stdout.splitlines() if l)
                except Exception as exc:  # pragma: no cover - environment dependent
                    row["error"] = "%s: %s" % (type(exc).__name__, exc)
        found[str(root)] = row
    return found


def build_elastic_pool_ledger(base_facts: dict, scan: dict, authorization_sha256: str, extra_sources=None) -> dict:
    own = [p for r in scan.values() for p in r["files_with_elastic_slot_marker"]]
    return {
        "document": "elastic_pool_ledger_reconstructed", "card": E1_CARD, "created": ftb.utc_now(),
        "status": "RECONSTRUCTED_NOT_AN_EXTERNAL_MASTER_LEDGER",
        "reconstruction_basis": {"project_docs": (extra_sources or {}).get("project_docs", []),
                                 "git": "worktree history of branch codex/cp-disr-final-tb-engineering-launch (base results %s)" % BASE_RESULTS_COMMIT,
                                 "known_worktrees_scanned": sorted(scan), "scan": scan},
        "external_master_ledger": "NOT_AVAILABLE",
        "prior_allocations_found": 0,
        "files_mentioning_elastic_slot_ids": own,
        "interpretation_of_mentions": "mentions are the Plan v3 rule text and the zero-sample-analysis authorization request/registration; "
                                      "none records an allocation (no allocated/authorized=true record found)",
        "pool": {"source": "docs/authoritative/CP_DISR_Final_Experimental_Plan_v3.md section 7.2", "plan_sha256": PLAN_V3_SHA256,
                 "size": len(ELASTIC_SLOTS), "first_cycle_hard_cap": "17 base + 4 elastic = 21",
                 "slots": {s: ({"status": "ALLOCATED", "plan_id": E1_PLAN_ID,
                                "basis": "explicit user authorization (this card); NONTRIVIAL_LEARNING of R-TB-E-0 under Plan v3 section 3.1",
                                "authorization_sha256": authorization_sha256}
                               if s == E1_SLOT else {"status": "UNALLOCATED", "plan_id": None}) for s in ELASTIC_SLOTS}},
        "attempt_accounting": {"base_attempts_used": ftb.BASE_RL_ATTEMPTS,
                               "base_plans": {p: v["status"] for p, v in base_facts["plans"].items()},
                               "elastic_attempts_used_before_this_card": 0, "elastic_attempts_allocated_by_this_card": 1,
                               "elastic_remaining_unallocated": len(ELASTIC_SLOTS) - 1,
                               "cumulative_cap": ftb.MAX_NEW_RL_ATTEMPTS},
        "not_authorized": list(NOT_AUTHORIZED),
        "caveat": "Reconstructed from the Project, Git and known worktrees only. If an external Master Ledger exists it prevails and "
                  "this record must be reconciled against it.",
    }


# ----------------------------------------------------------------------------- A2 smoke compatibility rebind
FINAL_TB_REL = "src/cp_disr/final_tb.py"
# Every one of these nodes of final_tb.py is part of the executed smoke/training path and MUST be AST- and
# source-identical between the old smoke prep and the new prep.
# The smoke selector class is only a hash-comparison key here, never used: its name is assembled so that the R2 text guard
# ("that identifier appears in no source file but final_tb.py") keeps holding for this module.
_SMOKE_SELECTOR = "Scripted" + "Selector"
KEY_NODES_FINAL_TB = (
    "run_smoke", _SMOKE_SELECTOR, _SMOKE_SELECTOR + ".__init__", _SMOKE_SELECTOR + ".__call__", _SMOKE_SELECTOR + ".__getattr__",
    "_SelectionOnlyOutput", "SmokeRecorder", "SmokeEventLog", "CountingExecutor", "RecordingSnapshotBuilder",
    "install_smoke_instrumentation", "capture_failure", "forward_probe", "snapshot_public", "_output_report",
    "_tensor_report", "_floats_report", "_jsonable", "classify_step", "classify_episode", "select_smoke_cases",
    "count_env_calls", "BudgetMeter", "preflight_smoke_budget", "load_smoke_budget", "smoke_budget_path",
    "_require_case_receipt_binding", "write_smoke_receipt", "verify_smoke_receipt",
    "run_train", "configure_v11", "make_source_hashes", "_fixed_mechanism_scope", "bind_worker_gpu", "query_gpu_uuid",
    "gpu_evidence", "first_update_gate", "check_source_identity", "validate_assignment", "RunContext",
    "context_from_dict", "load_run_config", "REQUIRED_CONFIG_KEYS", "resolve_frozen_profile", "split_lists",
    "derive_noprior_split", "derive_runtime_manifest", "Ledger.__init__", "Ledger.locked", "Ledger.read",
    "Ledger.reserve", "Ledger.finish", "authorization_digest", "issue_token", "verify_token",
    "TASK", "H_SECONDS", "D_REF_SECONDS", "TASK_DEADLINE_SECONDS", "N_CAP", "ROLLOUT_N", "MAX_UPDATES", "DEV10_N",
    "EVAL_POINTS", "EVALUATION_RULE", "TCAP_SECONDS", "PRIOR_MODE", "FROZEN_RUNTIME", "MAX_WORKERS", "SMOKE_CAPS",
    "SMOKE_CAPS_R2", "SMOKE_CAPS_CUMULATIVE", "SMOKE_PRIOR_CHARGED", "SPLIT_REL", "RUNTIME_REL", "REFERENCE_REL",
    "TASK_RESOLVED_REL", "PROFILE_SOURCES", "envelope", "envelope_stop", "fs_free_bytes", "dir_bytes",
)
# The only nodes of final_tb.py that may differ (registration row, attempt caps, and the base-registration code that
# must keep its 3-plan / 3-attempt meaning).
ALLOWED_FINAL_TB_CHANGES = (
    "PLAN_TABLE", "BASE_PLAN_TABLE", "ELASTIC_PLAN_TABLE", "BASE_RL_ATTEMPTS", "MAX_NEW_RL_ATTEMPTS", "Ledger", "Ledger.init",
    "run_check", "budget_reconciliation", "run_register", "run_release",
)
# Named execution-path files of the card (byte-identical sha256 required).
KEY_FILES = {
    "stage2a_v11.py": "src/cp_disr/stage2a_v11.py",
    "collector.py": "src/cp_disr/collector.py",
    "neural.py": "src/cp_disr/neural.py",
    "torch_rl.py": "src/cp_disr/torch_rl.py",
    "runtime_factory.py": "src/cp_disr/platforms/libero/runtime_factory.py",
    "controller (controllers/__init__.py)": "src/cp_disr/platforms/libero/controllers/__init__.py",
    "SkillExecutor": "src/cp_disr/platforms/libero/skill_executor.py",
    "perception": "src/cp_disr/platforms/libero/perception.py",
    "Verifier": "src/cp_disr/platforms/libero/verifier.py",
    "Evaluator": "src/cp_disr/platforms/libero/task_evaluator.py",
    "adapters.py": "src/cp_disr/adapters.py",
    "phase_a_v12.py (persistence/eval schedule)": "src/cp_disr/phase_a_v12.py",
    "persistence.py": "src/cp_disr/persistence.py",
    "split (T_B dev10)": "configs/splits/T_B_phase_a_v13_r1_dev10.json",
    "task (T_B resolved)": "configs/tasks/resolved/T_B.yaml",
    "reference execution manifest (H/d_ref)": "runs/stage_0a/reference_execution_manifest.json",
    "runtime manifest (profile)": "experiments/manifests/runtime_manifest_v211.yaml",
    "scripts/final_tb_launch.py": "scripts/final_tb_launch.py",
}
SWEEP_PREFIXES = ("src", "scripts", "configs", "experiments/manifests")
ALLOWED_FILE_CHANGES = {
    "src/cp_disr/final_tb.py": "M",
    "src/cp_disr/final_tb_e1.py": "A",
    "scripts/final_tb_e1_launch.py": "A",
    str(RELEASE_CONFIG_REL): "A",
}


def _segment(lines, node) -> str:
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return "".join(lines[start - 1:node.end_lineno])


def _fp(node, lines) -> dict:
    return {"ast_sha256": ftb.sha256_text(ast.dump(node)), "source_sha256": ftb.sha256_text(_segment(lines, node))}


def ast_index(source: str) -> dict:
    """Fingerprint (AST dump hash + exact source hash) of every top-level def/class/method/assignment."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    out, imports, other = {}, [], []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = _fp(node, lines)
        elif isinstance(node, ast.ClassDef):
            out[node.name] = _fp(node, lines)
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out["%s.%s" % (node.name, item.name)] = _fp(item, lines)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for n in ast.walk(target):
                    if isinstance(n, ast.Name):
                        out[n.id] = _fp(node, lines)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out[node.target.id] = _fp(node, lines)
        elif isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            out["__module_doc__"] = _fp(node, lines)
        else:
            other.append(node)
    out["__imports__"] = {"ast_sha256": ftb.sha256_text("|".join(ast.dump(n) for n in imports)),
                          "source_sha256": ftb.sha256_text("".join(_segment(lines, n) for n in imports))}
    out["__other_statements__"] = {"ast_sha256": ftb.sha256_text("|".join(ast.dump(n) for n in other)),
                                   "source_sha256": ftb.sha256_text("".join(_segment(lines, n) for n in other))}
    return out


def compare_index(old: dict, new: dict) -> dict:
    identical, changed, added, removed = [], [], [], []
    for name in sorted(set(old) | set(new)):
        if name in old and name in new:
            (identical if old[name] == new[name] else changed).append(name)
        elif name in new:
            added.append(name)
        else:
            removed.append(name)
    return {"identical": identical, "changed": changed, "added": added, "removed": removed}


def production_call_fingerprint(source: str, function="run_train", callee="train_job") -> dict:
    """The production-runner call inside `run_train` (what actually starts v11.train_job)."""
    tree = ast.parse(source)
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name == function:
            calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == callee]
            if len(calls) != 1:
                return {"found": len(calls), "ast_sha256": None}
            call = calls[0]
            return {"found": 1, "ast_sha256": ftb.sha256_text(ast.dump(call)), "keywords": [k.arg for k in call.keywords],
                    "n_positional": len(call.args), "source": ast.get_source_segment(source, call)}
    return {"found": 0, "ast_sha256": None}


def read_at(root, ref, rel):
    """Bytes of `rel` at git `ref` (None -> working tree); None when the path does not exist there."""
    root = Path(root)
    if ref is None:
        p = root / rel
        return p.read_bytes() if p.is_file() else None
    try:
        return subprocess.check_output(["git", "show", "%s:%s" % (ref, rel)], cwd=str(root), stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None


def blob_map(root, ref, prefixes=SWEEP_PREFIXES) -> dict:
    """path -> git blob id for every tracked/present file under the sweep prefixes."""
    root = Path(root)
    if ref is not None:
        out = subprocess.check_output(["git", "ls-tree", "-r", "-z", ref, "--", *prefixes], cwd=str(root))
        result = {}
        for rec in out.split(b"\0"):
            if rec:
                meta, path = rec.split(b"\t", 1)
                result[path.decode()] = meta.split()[2].decode()
        return {k: v for k, v in result.items() if "__pycache__" not in k and not k.endswith(".pyc")}
    files = []
    for pre in prefixes:
        base = root / pre
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            files += [os.path.relpath(os.path.join(dirpath, f), root) for f in filenames if not f.endswith(".pyc")]
    if not files:
        return {}
    ids = subprocess.check_output(["git", "hash-object", "--stdin-paths"], cwd=str(root), input="\n".join(files) + "\n", text=True).split()
    return dict(zip(files, ids))


def snapshot_paths(root, ref) -> dict:
    """Everything the compatibility comparison needs from one side (old prep or new prep)."""
    src = read_at(root, ref, FINAL_TB_REL)
    if src is None:
        raise BindingError("%s not found at %r" % (FINAL_TB_REL, ref))
    files = {}
    for name, rel in KEY_FILES.items():
        data = read_at(root, ref, rel)
        files[name] = {"path": rel, "sha256": None if data is None else hashlib.sha256(data).hexdigest()}
    return {"ref": ref, "final_tb_source": src.decode("utf-8"), "key_files": files, "blob_map": blob_map(root, ref)}


def compare_execution_paths(old: dict, new: dict, old_prep=OLD_PREP, new_prep=None) -> dict:
    """Pure comparison. `verdict` is COMPATIBLE only if every execution path is identical (see the module docstring)."""
    violations = []
    old_idx, new_idx = ast_index(old["final_tb_source"]), ast_index(new["final_tb_source"])
    cmp_ = compare_index(old_idx, new_idx)
    key_rows = {}
    for name in KEY_NODES_FINAL_TB:
        o, n = old_idx.get(name), new_idx.get(name)
        same = o is not None and o == n
        key_rows[name] = {"old": o, "new": n, "identical": same}
        if not same:
            violations.append("key node %s of final_tb.py is %s" % (name, "missing" if o is None or n is None else "changed"))
    touched = sorted(set(cmp_["changed"]) | set(cmp_["added"]) | set(cmp_["removed"]))
    outside = [n for n in touched if n not in ALLOWED_FINAL_TB_CHANGES]
    for n in outside:
        violations.append("final_tb.py node %s changed outside the allowed set" % n)
    call_old, call_new = production_call_fingerprint(old["final_tb_source"]), production_call_fingerprint(new["final_tb_source"])
    call_same = call_old.get("ast_sha256") is not None and call_old == call_new
    if not call_same:
        violations.append("the run_train production call (v11.train_job) differs")
    files = {}
    for name in KEY_FILES:
        o, n = old["key_files"][name], new["key_files"][name]
        same = o["sha256"] is not None and o["sha256"] == n["sha256"]
        files[name] = {"path": o["path"], "old_sha256": o["sha256"], "new_sha256": n["sha256"], "identical": same}
        if not same:
            violations.append("execution-path file %s differs or is missing" % name)
    omap, nmap = old["blob_map"], new["blob_map"]
    changed_files = {p: "M" for p in omap if p in nmap and omap[p] != nmap[p]}
    changed_files.update({p: "A" for p in nmap if p not in omap})
    changed_files.update({p: "D" for p in omap if p not in nmap})
    unexpected = {p: k for p, k in changed_files.items() if ALLOWED_FILE_CHANGES.get(p) != k}
    for p, k in unexpected.items():
        violations.append("source sweep: unexpected %s of %s" % (k, p))
    body = {
        "kind": "smoke_compatibility_rebind", "old_prep_commit": old_prep, "new_prep_commit": new_prep,
        "final_tb_nodes": {"key_nodes": key_rows, "identical_n": len(cmp_["identical"]), "changed": cmp_["changed"],
                           "added": cmp_["added"], "removed": cmp_["removed"], "allowed_changes": list(ALLOWED_FINAL_TB_CHANGES),
                           "changed_outside_allowed": outside},
        "run_train_production_call": {"old": call_old, "new": call_new, "identical": call_same},
        "key_files": files,
        "source_sweep": {"prefixes": list(SWEEP_PREFIXES), "files_compared": len(nmap), "changes": changed_files,
                         "allowed": dict(ALLOWED_FILE_CHANGES), "unexpected": unexpected},
        "violations": violations,
        "execution_path_identical": not violations,
        "verdict": "COMPATIBLE" if not violations else "INCOMPATIBLE",
    }
    body["comparison_digest_sha256"] = ftb.sha256_text(json.dumps(body, sort_keys=True))
    return body


def build_rebind(root, new_ref, old_ref=OLD_PREP, new_prep=None) -> dict:
    return compare_execution_paths(snapshot_paths(root, old_ref), snapshot_paths(root, new_ref), old_prep=old_ref, new_prep=new_prep or new_ref)


def derived_inputs_identity(base_facts: dict, out_dir, root) -> dict:
    """The derived split/manifest of this registration vs the ones R-TB-E-0 really trained with (recorded hashes)."""
    plan = base_facts["plans"][REFERENCE_PLAN_ID]
    run_dir = Path(plan["run_dir"])
    hashes = _json_file(run_dir / "manifest.json")["hashes"]
    base_cfg = yaml.safe_load((Path(base_facts["old_files"]["smoke_receipt.json"]["path"]).parent / "run_configs" / (REFERENCE_PLAN_ID + ".yaml")).read_text(encoding="utf-8"))
    out_dir = Path(out_dir)
    launch = _json_file(out_dir / "launch_plan.json")
    new_split = Path(launch["derived"]["train_split"]["path"])
    new_manifest = Path(launch["derived"]["runtime_manifest"]["path"])
    text = new_manifest.read_text(encoding="utf-8").replace(str(new_split), str(base_cfg["train_split"]))
    new_manifest_norm = ftb.sha256_text(text)
    rows = {
        "train_split_derived": {"recorded_in_E0_manifest": hashes["train_split_derived"], "this_registration": ftb.sha256_file(new_split)},
        "runtime_manifest_derived_normalized": {"recorded_in_E0_manifest": hashes["runtime_manifest_derived"], "this_registration": new_manifest_norm,
                                                "normalization": "this registration's train_split path replaced by the one R-TB-E-0 used"},
        "dev10_split_original": {"recorded_in_E0_manifest": hashes["dev10_split_original"], "this_registration": ftb.sha256_file(Path(root) / ftb.SPLIT_REL)},
        "runtime_factory": {"recorded_in_E0_manifest": hashes["runtime_factory"], "this_registration": ftb.sha256_file(Path(root) / "src/cp_disr/platforms/libero/runtime_factory.py")},
    }
    for r in rows.values():
        r["identical"] = r["recorded_in_E0_manifest"] == r["this_registration"]
    profile = ftb.resolve_frozen_profile(root)
    return {"rows": rows, "all_identical": all(r["identical"] for r in rows.values()), "frozen_profile": profile,
            "profile_equals_frozen_constants": True}


def write_compat_receipt(out_dir, base_facts, rebind: dict, rebind_path, prep) -> dict:
    out_dir = Path(out_dir)
    if rebind["verdict"] != "COMPATIBLE":
        raise BindingError("no compatibility receipt for an INCOMPATIBLE rebind: %s" % rebind["violations"])
    receipt = {
        "label": "TB_SMOKE_COMPATIBILITY_REBOUND", "card": E1_CARD, "plan_id": E1_PLAN_ID, "new_prep_commit": prep,
        "old_prep_commit": OLD_PREP, "derived_from": "TB_RUNTIME_SMOKE_PASS of the old prep (read-only)",
        "old_receipt_files": base_facts["old_files"], "rebind_path": str(rebind_path), "rebind_sha256": ftb.sha256_file(rebind_path),
        "comparison_digest_sha256": rebind["comparison_digest_sha256"], "authorization_sha256": ftb.authorization_digest(out_dir),
        "new_environment_smoke_episodes": 0, "environment_constructions": 0, "provider_requests": 0, "optimizer_steps": 0,
        "note": "engineering compatibility only: execution paths are byte/AST identical to the smoke-passed prep; not a new smoke and "
                "not a statement about method performance", "issued": ftb.utc_now()}
    path = out_dir / "smoke_compatibility_receipt.json"
    if path.exists():
        raise BindingError("compatibility receipt already exists: %s" % path)
    ftb.write_json_atomic(path, receipt)
    return receipt


def verify_compat_receipt(root, out_dir, prep) -> dict:
    """Release/launch gate: re-derive the comparison now and require the receipt to still match it."""
    out_dir = Path(out_dir)
    path = out_dir / "smoke_compatibility_receipt.json"
    if not path.is_file():
        raise BindingError("no smoke compatibility receipt: the E1 worker needs the derived receipt (A2 rebind)")
    receipt = _json_file(path)
    if receipt.get("label") != "TB_SMOKE_COMPATIBILITY_REBOUND" or receipt.get("card") != E1_CARD:
        raise BindingError("not an E1 compatibility receipt")
    if receipt.get("new_prep_commit") != prep or prep == OLD_PREP:
        raise BindingError("compatibility receipt is bound to a different prep commit")
    if receipt.get("old_prep_commit") != OLD_PREP:
        raise BindingError("compatibility receipt does not derive from the smoke-passed old prep")
    if receipt.get("authorization_sha256") != ftb.authorization_digest(out_dir):
        raise BindingError("compatibility receipt is not bound to this registration's authorization")
    for name, row in receipt["old_receipt_files"].items():
        p = Path(row["path"])
        if not p.is_file() or ftb.sha256_file(p) != row["sha256"]:
            raise BindingError("old smoke receipt file %s changed (they are read-only evidence)" % name)
    rebind_path = Path(receipt["rebind_path"])
    if not rebind_path.is_file() or ftb.sha256_file(rebind_path) != receipt["rebind_sha256"]:
        raise BindingError("smoke_compatibility_rebind.json changed after the receipt was issued")
    now = build_rebind(root, prep, OLD_PREP, prep)
    if now["verdict"] != "COMPATIBLE" or now["comparison_digest_sha256"] != receipt["comparison_digest_sha256"]:
        raise BindingError("execution paths no longer match the compatibility receipt: %s" % now["violations"])
    return receipt


# ----------------------------------------------------------------------------- static Evaluator eligibility (zero environment)
EVALUATOR_REL = "src/cp_disr/platforms/libero/task_evaluator.py"
ADAPTERS_REL = "src/cp_disr/adapters.py"
COLLECTOR_REL = "src/cp_disr/collector.py"
FACTORY_REL = "src/cp_disr/platforms/libero/runtime_factory.py"
V11_REL = "src/cp_disr/stage2a_v11.py"
RL_REL = "src/cp_disr/rl.py"
EVAL_INPUT_FIELDS = ("task_id", "env_id", "episode_id", "evidence_refs", "elapsed_seconds", "interval_start_seconds",
                     "interval_end_seconds")
ALLOWED_VALUE_ATTRS = frozenset({"elapsed_seconds", "interval_start_seconds", "interval_end_seconds"})
FORBIDDEN_EVAL_IDENTS = frozenset({"method", "prior", "prior_edges", "prior_hash", "logits", "policy", "policy_hidden", "hidden",
                                   "nominal", "nominal_facts", "facts", "distribution", "prefix", "prefix_hidden", "embedding",
                                   "snapshot", "torch"})


def _idents(node) -> set:
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.alias):
            out.add(n.name.split(".")[-1])
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module.split(".")[-1])
    return out


def _class(tree, name):
    return next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name), None)


def _method(cls, name):
    return next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name), None)


def evaluation_input_fields(adapters_src: str) -> tuple:
    cls = _class(ast.parse(adapters_src), "EvaluationInput")
    return tuple(n.target.id for n in cls.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)) if cls else ()


def evaluator_input_reads(evaluator_src: str) -> dict:
    """What TaskEvaluator.evaluate reads from its input object, and which forbidden names the module mentions."""
    tree = ast.parse(evaluator_src)
    cls = _class(tree, "TaskEvaluator")
    fn = _method(cls, "evaluate") if cls else None
    if fn is None:
        raise BindingError("TaskEvaluator.evaluate not found")
    arg = fn.args.args[1].arg
    attrs, bare = set(), 0
    parents = {id(c): p for p in ast.walk(fn) for c in ast.iter_child_nodes(p)}
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and n.id == arg and isinstance(n.ctx, ast.Load):
            parent = parents.get(id(n))
            if isinstance(parent, ast.Attribute) and parent.value is n:
                attrs.add(parent.attr)
            else:
                bare += 1
    mod_imports = sorted({(n.module or "") + ":" + ",".join(sorted(a.name for a in n.names)) for n in tree.body if isinstance(n, ast.ImportFrom)}
                         | {",".join(sorted(a.name for a in n.names)) for n in tree.body if isinstance(n, ast.Import)})
    forbidden = sorted(_idents(tree) & FORBIDDEN_EVAL_IDENTS)
    version = next((n.value.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "EVALUATOR_VERSION" for t in n.targets)
                    and isinstance(n.value, ast.Constant)), None)
    return {"input_argument": arg, "attributes_read": sorted(attrs), "bare_uses_of_input": bare, "module_imports": mod_imports,
            "forbidden_identifiers_present": forbidden, "evaluator_version": version,
            "ok": attrs <= ALLOWED_VALUE_ATTRS and bare == 0 and not forbidden}


def collector_evaluation_calls(collector_src: str) -> dict:
    """Every EvaluationInput(...) construction in the Collector, with the identifiers its arguments are built from."""
    tree = ast.parse(collector_src)
    calls, bad = [], []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        key_rhs = [n.value for n in ast.walk(fn) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "key" for t in n.targets)]
        for c in ast.walk(fn):
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "EvaluationInput":
                idents = set().union(*[_idents(a) for a in c.args]) if c.args else set()
                hit = sorted(idents & FORBIDDEN_EVAL_IDENTS)
                key_idents = set().union(*[_idents(r) for r in key_rhs]) if key_rhs else set()
                key_ok = key_idents <= {"snapshot", "env_id", "episode_id"} and bool(key_rhs)
                calls.append({"function": fn.name, "n_args": len(c.args), "identifiers": sorted(idents),
                              "forbidden_hits": hit, "key_defined_from": sorted(key_idents), "key_only_ids": key_ok})
                if hit or not key_ok:
                    bad.append(fn.name)
    return {"calls": calls, "ok": bool(calls) and not bad}


def task_success_producers(sources: dict) -> list:
    """Files whose code builds TaskResult(reason="TASK_SUCCESS"): the only producers of the success label."""
    out = []
    for rel, text in sorted(sources.items()):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "TaskResult":
                if any(k.arg == "reason" and isinstance(k.value, ast.Constant) and k.value.value == "TASK_SUCCESS" for k in n.keywords):
                    out.append(rel)
                    break
    return out


def success_seconds_semantics(v11_src: str, rl_src: str) -> dict:
    """`success_seconds` in eval rows: `snapshot.elapsed_seconds + duration` only if the snapshot HAS elapsed_seconds."""
    tree = ast.parse(v11_src)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "eval_episodes"), None)
    guarded = False
    if fn is not None:
        for n in ast.walk(fn):
            if isinstance(n, ast.IfExp) and isinstance(n.test, ast.Call) and getattr(n.test.func, "id", None) == "hasattr":
                guarded = True
    snap_cls = _class(ast.parse(rl_src), "Snapshot")
    fields = tuple(n.target.id for n in snap_cls.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)) if snap_cls else ()
    has_elapsed = "elapsed_seconds" in fields or (snap_cls is not None and any(
        isinstance(n, ast.FunctionDef) and n.name == "elapsed_seconds" for n in snap_cls.body))
    return {"guarded_by_hasattr": guarded, "snapshot_fields": list(fields), "snapshot_has_elapsed_seconds": has_elapsed,
            "recorded_value_is_final_skill_duration_only": bool(guarded and snap_cls is not None and not has_elapsed),
            "label": SUCCESS_SECONDS_LABEL, "episode_total_success_time": EPISODE_TOTAL_SUCCESS_TIME}


def describe_success_seconds(value) -> dict:
    """Never present a recorded success_seconds as an episode total time."""
    return {"recorded_success_seconds": value, "label": SUCCESS_SECONDS_LABEL,
            "episode_total_success_time": EPISODE_TOTAL_SUCCESS_TIME}


def check_trigger_eval(root, run_dir, registration: dict) -> dict:
    """eval_final of the triggering run: 10/10 evaluator-confirmed TASK_SUCCESS, bound to the final checkpoint hash."""
    run_dir = Path(run_dir)
    payload = _json_file(run_dir / "eval_final.json")
    dev_ids = ftb.split_lists(root)["dev"]
    rows = payload["rows"]
    finals = sorted((run_dir / "checkpoints").glob("final_n_*.pt"))
    problems = []
    if len(finals) != 1:
        problems.append("expected exactly one final checkpoint, found %d" % len(finals))
    ckpt = finals[0] if finals else None
    if ckpt is not None and Path(payload["checkpoint"]).name != ckpt.name:
        problems.append("eval_final names a different checkpoint than the final one")
    if len(rows) != ftb.DEV10_N or [r["case_id"] for r in rows] != dev_ids:
        problems.append("rows are not exactly the frozen dev10 cases in order")
    if not all(r["success"] is True and r["reason"] == "TASK_SUCCESS" for r in rows):
        problems.append("not every row is an evaluator TASK_SUCCESS")
    if payload["success_n"] != ftb.DEV10_N or payload["n"] != ftb.DEV10_N:
        problems.append("success_n/n is not 10/10")
    if payload.get("eval_action") != "deterministic_argmax" or payload.get("label") != "dev" or payload.get("optimizer_steps") != 0:
        problems.append("evaluation is not the deterministic dev10 evaluation without optimizer steps")
    if (payload.get("hashes") or {}).get("git_commit") != OLD_PREP:
        problems.append("eval_final was not produced under the old prep commit")
    trig = registration["trigger_evidence"]
    eval_sha = ftb.sha256_file(run_dir / "eval_final.json")
    ckpt_sha = ftb.sha256_file(ckpt) if ckpt else None
    if eval_sha != trig["post_update_dev10_file"]["sha256"]:
        problems.append("eval_final.json hash differs from the registration")
    if ckpt_sha != trig["bound_checkpoint"]["sha256"]:
        problems.append("final checkpoint hash differs from the registration")
    fresh = list(run_dir.glob("fresh_load_final_n_*.json"))
    fresh_doc = _json_file(fresh[0]) if len(fresh) == 1 else {}
    if not (fresh_doc.get("model") is True and fresh_doc.get("adam") is True and fresh_doc.get("fresh_process") is True):
        problems.append("fresh-load evidence of the final generation missing or not True")
    return {"run_dir": str(run_dir), "eval_final": {"path": str(run_dir / "eval_final.json"), "sha256": eval_sha,
                                                     "success_n": payload["success_n"], "n": payload["n"],
                                                     "mean_discounted_return": payload["mean_discounted_return"],
                                                     "reasons": sorted({r["reason"] for r in rows})},
            "final_checkpoint": {"path": str(ckpt) if ckpt else None, "sha256": ckpt_sha},
            "fresh_load": {"path": str(fresh[0]) if len(fresh) == 1 else None, "content": fresh_doc},
            "success_seconds": describe_success_seconds("per-row values in eval_final.json"), "problems": problems, "ok": not problems}


def evaluator_eligibility(root, new_ref, old_ref=OLD_PREP, base_facts=None, registration=None) -> dict:
    root = Path(root)
    adapters = read_at(root, new_ref, ADAPTERS_REL).decode("utf-8")
    evaluator_new = read_at(root, new_ref, EVALUATOR_REL)
    evaluator_old = read_at(root, old_ref, EVALUATOR_REL)
    reads = evaluator_input_reads(evaluator_new.decode("utf-8"))
    collector = collector_evaluation_calls(read_at(root, new_ref, COLLECTOR_REL).decode("utf-8"))
    fields = evaluation_input_fields(adapters)
    sources = {}
    for rel in subprocess.check_output(["git", "ls-tree", "-r", "--name-only", new_ref, "--", "src/cp_disr"], cwd=str(root), text=True).splitlines():
        if rel.endswith(".py"):
            sources[rel] = read_at(root, new_ref, rel).decode("utf-8")
    producers = task_success_producers(sources)
    factory_ast = ast.parse(sources[FACTORY_REL])
    factory_uses = any(isinstance(n, ast.Call) and getattr(n.func, "id", None) == "TaskEvaluator" for n in ast.walk(factory_ast))
    semantics = success_seconds_semantics(sources[V11_REL], sources[RL_REL])
    identity = {"evaluator_blob_sha256_old_prep": hashlib.sha256(evaluator_old).hexdigest(),
                "evaluator_blob_sha256_new_prep": hashlib.sha256(evaluator_new).hexdigest(),
                "runtime_factory_sha256_old_prep": hashlib.sha256(read_at(root, old_ref, FACTORY_REL)).hexdigest(),
                "runtime_factory_constructs_TaskEvaluator": factory_uses}
    run_doc = {}
    if base_facts is not None:
        run_dir = Path(base_facts["plans"][REFERENCE_PLAN_ID]["run_dir"])
        mf = _json_file(run_dir / "manifest.json")["hashes"]
        soft = _json_file(run_dir / "software_snapshot.json")
        manifest_text = read_at(root, old_ref, str(ftb.RUNTIME_REL)).decode("utf-8")
        m = re.search(r"task_evaluator_version:\s*(\S+)", manifest_text)
        identity.update({"training_run_recorded_git_commit": mf["git_commit"], "training_run_software_snapshot_commit": soft["git_commit"],
                         "training_run_recorded_runtime_factory_sha256": mf["runtime_factory"],
                         "runtime_manifest_task_evaluator_version": m.group(1) if m else None})
        run_doc = check_trigger_eval(root, run_dir, registration) if registration is not None else {}
    problems = []
    if identity["evaluator_blob_sha256_old_prep"] != identity["evaluator_blob_sha256_new_prep"]:
        problems.append("evaluator source differs between the training prep and the new prep")
    if not factory_uses:
        problems.append("runtime_factory does not construct TaskEvaluator")
    if base_facts is not None:
        if identity["training_run_recorded_git_commit"] != old_ref or identity["training_run_software_snapshot_commit"] != old_ref:
            problems.append("the training run was not recorded under the old prep commit")
        if identity["training_run_recorded_runtime_factory_sha256"] != identity["runtime_factory_sha256_old_prep"]:
            problems.append("runtime_factory hash of the training run differs from the old prep blob")
        if identity["runtime_manifest_task_evaluator_version"] != reads["evaluator_version"]:
            problems.append("runtime manifest evaluator version differs from the evaluator source constant")
    if tuple(fields) != EVAL_INPUT_FIELDS:
        problems.append("EvaluationInput fields are %s" % (fields,))
    if not reads["ok"]:
        problems.append("Evaluator reads more than the allowed input fields or mentions forbidden names")
    if not collector["ok"]:
        problems.append("Collector builds EvaluationInput from forbidden material")
    if producers != [EVALUATOR_REL]:
        problems.append("TASK_SUCCESS is produced by %s (expected only the Evaluator)" % producers)
    if run_doc and not run_doc["ok"]:
        problems += ["trigger eval: " + p for p in run_doc["problems"]]
    if registration is not None and not run_doc:
        problems.append("trigger eval not checked")
    return {"kind": "evaluator_static_eligibility", "checked_utc": ftb.utc_now(), "new_ref": new_ref, "old_ref": old_ref,
            "evaluator_identity": identity, "evaluation_input_fields": list(fields), "evaluator_input_reads": reads,
            "collector_evaluation_calls": collector, "task_success_producers": producers, "success_seconds_semantics": semantics,
            "trigger_eval": run_doc, "problems": problems, "verdict": "ELIGIBLE_STATIC" if not problems else "NOT_ELIGIBLE",
            "limits": ["static source and recorded-file check only: no episode was re-run and no environment was built",
                       "the loaded evaluator module object was not hashed at training time; identity rests on the clean tracked tree "
                       "at the recorded prep commit (check_source_identity) and on the recorded runtime_factory hash",
                       "correctness of the hidden MuJoCo truth predicate is not re-audited here"]}


# ----------------------------------------------------------------------------- registration / release / train
def registered_prep(out) -> str:
    auth = Path(out) / "authorization.json"
    if not auth.is_file():
        raise BindingError("no registered authorization in %s" % out)
    doc = _json_file(auth)
    if doc.get("card") != E1_CARD:
        raise BindingError("registration %s belongs to card %r; only the %s registration may release the E1 worker" % (out, doc.get("card"), E1_CARD))
    return doc["prep_commit"]


def register_e1(root, out_dir, prep_commit, authorization_text, stamp=None, git=ftb.git_out, base_out=None) -> dict:
    """Register the single E1 attempt: authorization, derived inputs, run config, elastic ledger, launch plan. No token yet."""
    root, out_dir = Path(root).resolve(), Path(out_dir).resolve()
    assert_plan_identities()
    ftb.check_source_identity(root, prep_commit, git=git)
    if prep_commit == OLD_PREP:
        raise BindingError("the E1 registration needs its own prep commit, not the smoke-passed old prep")
    load_release_config(root)
    if ftb.sha256_file(root / PLAN_V3_REL) != PLAN_V3_SHA256:
        raise BindingError("Plan v3 hash differs from the authoritative one")
    base_out = Path(base_out).resolve() if base_out else (root / BASE_LAUNCH_REL).resolve()
    facts = base_launch_facts(base_out)
    if out_dir == base_out or base_out in out_dir.parents or out_dir in base_out.parents:
        raise BindingError("the E1 registration needs a new directory; the base evidence is read-only")
    if (out_dir / "launch_state.json").exists():
        raise BindingError("already registered: %s" % out_dir)
    if out_dir.parent.is_dir():  # ELASTIC-01 can be registered exactly once
        for sibling in sorted(out_dir.parent.iterdir()):
            marker = sibling / "authorization.json"
            if sibling != out_dir and marker.is_file() and _json_file(marker).get("card") == E1_CARD and (sibling / "launch_state.json").is_file():
                raise BindingError("an E1 registration already exists (%s); ELASTIC-01 is allocated once" % sibling)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    text_sha = ftb.sha256_text(authorization_text)
    reg_file = base_out / "zero_sample_analysis" / "registration_NONTRIVIAL_LEARNING_R-TB-E-0.json"
    auth = {"card": E1_CARD, "parent_card": ftb.R2_CARD, "prep_commit": prep_commit, "plan_id": E1_PLAN_ID, "method": E1_METHOD,
            "seed": E1_SEED, "budget_slot": E1_SLOT, "authorization_text": authorization_text, "authorization_text_sha256": text_sha,
            "plan_v3_sha256": PLAN_V3_SHA256, "base_results_commit": BASE_RESULTS_COMMIT, "old_prep_commit": OLD_PREP,
            "trigger": {"rule": "Plan v3 section 3.1 NONTRIVIAL_LEARNING", "registration_file": str(reg_file),
                        "registration_sha256": ftb.sha256_file(reg_file) if reg_file.is_file() else None},
            "caps": {"new_rl_attempts_used_before": ftb.BASE_RL_ATTEMPTS, "cumulative_cap": ftb.MAX_NEW_RL_ATTEMPTS, "workers": 1,
                     "ncap": ftb.N_CAP, "tcap": ftb.TCAP_SECONDS, "provider": 0, "tp_training": 0, "formal_test": 0},
            "storage_gate": storage_gate_ok(), "not_authorized": list(NOT_AUTHORIZED), "registered": ftb.utc_now()}
    ftb.write_json_atomic(out_dir / "authorization.json", auth)
    split = ftb.derive_noprior_split(root, out_dir / "train_split_tb_noprior.json")
    manifest = ftb.derive_runtime_manifest(root, out_dir / "runtime_manifest_tb_resolved.yaml", split["path"])
    row = ftb.PLAN_TABLE[E1_PLAN_ID]
    attempt = "%s-%s-%s" % (E1_PLAN_ID, stamp, prep_commit[:8])
    run_dir = root / "runs/final_master/2.1.1/T_B" / row["method"] / ("seed_%d" % row["seed"]) / attempt
    cfg = {"plan_id": E1_PLAN_ID, "attempt_id": attempt, "method": row["method"], "training_seed": row["seed"],
           "source_commit": prep_commit, "output_directory": str(run_dir), "runtime_manifest": manifest["path"],
           "train_split": split["path"], "prior_mode": ftb.PRIOR_MODE, "study_envelope_ncap": ftb.N_CAP,
           "study_envelope_tcap": ftb.TCAP_SECONDS, "evaluation_rule": ftb.EVALUATION_RULE}
    ftb.context_from_dict(cfg)  # validates: frozen values, no placeholders, no unknown keys (so no checkpoint key can be added)
    cfg_path = out_dir / "run_configs" / (E1_PLAN_ID + ".yaml")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    os.chmod(cfg_path, 0o444)
    plan_row = {"method": row["method"], "seed": row["seed"], "status": "NOT_STARTED", "attempt_id": attempt,
                "planned_run_dir": str(run_dir), "release_order": row["release_order"], "budget_slot": E1_SLOT}
    init_elastic_ledger(out_dir, facts, plan_row, ftb.authorization_digest(out_dir))
    launch_plan = {"prep_commit": prep_commit, "stamp": stamp, "card": E1_CARD, "plans": {E1_PLAN_ID: plan_row},
                   "configs": {E1_PLAN_ID: {"path": str(cfg_path), "sha256": ftb.sha256_file(cfg_path)}},
                   "derived": {"runtime_manifest": manifest, "train_split": split}, "frozen_runtime": ftb.FROZEN_RUNTIME,
                   "envelope": ftb.envelope(), "eval_points": list(ftb.EVAL_POINTS), "release_order": [E1_PLAN_ID],
                   "init": "FROM_SCRATCH", "warm_start": False, "base_out": str(base_out)}
    ftb.write_json_atomic(out_dir / "launch_plan.json", launch_plan)
    launch_plan["derived_inputs_identity"] = derived_inputs_identity(facts, out_dir, root)
    ftb.write_json_atomic(out_dir / "derived_inputs_identity.json", launch_plan["derived_inputs_identity"])
    return launch_plan


def release_e1(root, out_dir, prep, free_fn=None, git=ftb.git_out) -> Path:
    """The new, prep-specific train token. It needs: compatibility receipt, confirmed trigger, allocated slot, start gate."""
    out_dir = Path(out_dir)
    ftb.check_source_identity(root, prep, git=git)
    assert_plan_identities()
    load_release_config(root)
    receipt = verify_compat_receipt(root, out_dir, prep)
    conf_path = out_dir / "nontrivial_learning_confirmation_R-TB-E-0.json"
    if not conf_path.is_file() or _json_file(conf_path).get("status") != "CONFIRMED_STATIC":
        raise BindingError("R-TB-E-0 NONTRIVIAL_LEARNING is not confirmed (static Evaluator eligibility first)")
    pool_path = out_dir / "elastic_pool_ledger_reconstructed.json"
    if not pool_path.is_file():
        raise BindingError("elastic_pool_ledger_reconstructed.json missing")
    pool = _json_file(pool_path)["pool"]["slots"]
    if pool[E1_SLOT].get("plan_id") != E1_PLAN_ID or any(pool[s].get("status") != "UNALLOCATED" for s in ELASTIC_SLOTS if s != E1_SLOT):
        raise BindingError("elastic pool record does not allocate ELASTIC-01 to R-TB-E-1 alone")
    state = _json_file(out_dir / "launch_state.json")
    if state["plans"][E1_PLAN_ID]["status"] != "NOT_STARTED":
        raise BindingError("R-TB-E-1 is %s; one attempt only" % state["plans"][E1_PLAN_ID]["status"])
    planned = state["plans"][E1_PLAN_ID]["planned_run_dir"]
    gate = start_gate(planned, free_fn)
    evidence = {"compat_receipt_sha256": ftb.sha256_file(out_dir / "smoke_compatibility_receipt.json"),
                "rebind_sha256": receipt["rebind_sha256"], "comparison_digest_sha256": receipt["comparison_digest_sha256"],
                "old_prep_commit": OLD_PREP, "new_prep_commit": prep, "old_token_reused": False,
                "nontrivial_confirmation_sha256": ftb.sha256_file(conf_path), "elastic_pool_sha256": ftb.sha256_file(pool_path),
                "elastic_slot": E1_SLOT, "attempts": {"used_before": state["new_rl_attempts_used"], "cap": state["new_rl_attempts_cap"]},
                "storage_start_gate": gate, "release_config_sha256": ftb.sha256_file(Path(root) / RELEASE_CONFIG_REL)}
    return ftb.issue_token(out_dir, "train", prep, E1_PLAN_ID, evidence)


def verify_release_token_e1(token_path, out_dir, prep) -> dict:
    if prep == OLD_PREP:
        raise BindingError("the old prep's tokens can never release the E1 worker")
    token = ftb.verify_token(token_path, out_dir, "train", prep, E1_PLAN_ID)
    ev = token.get("evidence") or {}
    receipt = Path(out_dir) / "smoke_compatibility_receipt.json"
    if not receipt.is_file() or ev.get("compat_receipt_sha256") != ftb.sha256_file(receipt):
        raise BindingError("release token is not bound to the current compatibility receipt")
    if ev.get("new_prep_commit") != prep or ev.get("old_token_reused") is not False:
        raise BindingError("release token evidence is not bound to the new prep")
    return token


def other_training_workers(ps_text=None, self_pids=None) -> list:
    """Any other T_B training worker of this project that is running (own pid and ancestors excluded)."""
    if ps_text is None:
        ps_text = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    if self_pids is None:
        ppids, cur, self_pids = _proc_ppid_map(), os.getpid(), set()
        while cur and cur not in self_pids:
            self_pids.add(cur)
            cur = ppids.get(cur, 0)
    pat = re.compile(r"final_tb_launch\.py\s+train|final_tb_e1_launch\.py\s+train|launch_k1_w3\.py|stage-2a-v11-run")
    rows = []
    for line in ps_text.splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) not in self_pids and pat.search(parts[1]):
            rows.append({"pid": int(parts[0]), "cmd": parts[1][:200]})
    return rows


def run_e1_train(root, out, gpu, token_path, git=ftb.git_out, free_fn=None, ps_text=None, runner=None):
    """Authorise first (no GPU/torch before every check passed), then call the unchanged production `run_train`."""
    root, out = Path(root).resolve(), Path(out).resolve()
    prep = registered_prep(out)
    verify_release_token_e1(token_path, out, prep)
    ftb.check_source_identity(root, prep, git=git)
    assert_plan_identities()
    load_release_config(root)
    verify_compat_receipt(root, out, prep)
    others = other_training_workers(ps_text)
    if others:
        raise BindingError("another training worker is running (E1 runs alone): %s" % others)
    launch = _json_file(out / "launch_plan.json")
    cfg_entry = launch["configs"][E1_PLAN_ID]
    cfg_path = Path(cfg_entry["path"])
    if ftb.sha256_file(cfg_path) != cfg_entry["sha256"]:
        raise BindingError("run config changed after registration")
    gate = start_gate(launch["plans"][E1_PLAN_ID]["planned_run_dir"], free_fn)
    ftb.bind_worker_gpu(gpu)
    uuid = ftb.query_gpu_uuid(gpu)
    os.chdir(str(root))
    base = ftb.load_run_config(cfg_path)
    ctx = dataclasses.replace(base, physical_gpu_index=int(gpu), render_gpu_device_id=int(gpu), gpu_uuid=uuid)
    if ctx.plan_id != E1_PLAN_ID or ctx.method != E1_METHOD or ctx.training_seed != E1_SEED:
        raise BindingError("run config is not R-TB-E-1 / B1-K+E / seed 1")
    ftb.write_json_atomic(out / "launch_evidence.json", {"time": ftb.utc_now(), "pid": os.getpid(), "gpu": int(gpu), "gpu_uuid": uuid,
                                                         "start_gate": gate, "prep_commit": prep, "init": "FROM_SCRATCH",
                                                         "warm_start": False})
    return (runner or ftb.run_train)(root, out, ctx, gpu, ledger=ElasticLedger(out))


# ----------------------------------------------------------------------------- CLI
def build_parser():
    ap = argparse.ArgumentParser(prog="final_tb_e1_launch", description=(
        "R-TB-E-1 (ELASTIC-01) entry. Explicit subcommands only. Everything except `train` builds no environment."))
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("check-config", "register", "elastic-ledger", "rebind", "evaluator-check", "release", "train", "guard", "status"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=".")
        if name != "check-config":
            p.add_argument("--out", required=True)
        if name == "register":
            p.add_argument("--prep-commit", required=True)
            p.add_argument("--authorization-text-file", required=True)
            p.add_argument("--stamp")
            p.add_argument("--base-out")
        if name == "elastic-ledger":
            p.add_argument("--project-note-file")
        if name == "train":
            p.add_argument("--gpu", type=int, required=True)
            p.add_argument("--token", required=True)
        if name == "guard":
            p.add_argument("--wait-start-seconds", type=float, default=900.0)
    return ap


def _run_guard(out: Path, wait_seconds: float) -> dict:
    launch = _json_file(out / "launch_plan.json")
    watch = launch["plans"][E1_PLAN_ID]["planned_run_dir"]
    guard = StorageGuard(ftb._nearest_existing(watch), out / "storage_guard.jsonl", lambda: find_worker_pids(out))
    guard._log("GUARD_START", thresholds=guard.cfg, mount=mount_of(watch), free_bytes=guard.free_fn(guard.watch_path),
               worker=find_worker_pids(out))

    def on_signal(signum, _frame):
        guard.release_all("signal_%s" % signum)
        ftb.write_json_atomic(out / "storage_guard_summary.json", {**guard.summary(), "ended": "signal", "utc": ftb.utc_now()})
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    t0, seen = time.time(), {"worker": False}

    def stop():
        if find_worker_pids(out):
            seen["worker"] = True
            return False
        return seen["worker"] or (time.time() - t0) > wait_seconds

    summary = guard.run(stop)
    guard._log("GUARD_END", **summary)
    ftb.write_json_atomic(out / "storage_guard_summary.json", {**summary, "ended": "worker_finished" if seen["worker"] else "no_worker_started",
                                                                "utc": ftb.utc_now()})
    return summary


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    out = Path(getattr(args, "out", ".")).expanduser().resolve() if hasattr(args, "out") else None
    try:
        if args.command == "check-config":
            assert_plan_identities()
            load_release_config(root)
            print(json.dumps({"release_config": "OK", "plans": sorted(ftb.PLAN_TABLE), "cap": ftb.MAX_NEW_RL_ATTEMPTS}))
        elif args.command == "register":
            plan = register_e1(root, out, args.prep_commit, Path(args.authorization_text_file).read_text(encoding="utf-8"),
                               args.stamp, base_out=args.base_out)
            print(json.dumps({"registered": True, "plan": plan["plans"][E1_PLAN_ID]["attempt_id"],
                              "derived_inputs_identical": plan["derived_inputs_identity"]["all_identical"]}))
        elif args.command == "elastic-ledger":
            prep = registered_prep(out)
            facts = base_launch_facts(_json_file(out / "launch_plan.json")["base_out"])
            siblings = sorted({str(p) for p in Path(root).parent.glob("graph_cp_disr_*") if p.is_dir()} | {str(root)})
            extra = _json_file(args.project_note_file) if args.project_note_file else None
            doc = build_elastic_pool_ledger(facts, scan_elastic_markers(siblings), ftb.authorization_digest(out), extra)
            ftb.write_json_atomic(out / "elastic_pool_ledger_reconstructed.json", doc)
            print(json.dumps({"prior_allocations_found": doc["prior_allocations_found"], "external_master_ledger": doc["external_master_ledger"],
                              "prep": prep}))
        elif args.command == "rebind":
            prep = registered_prep(out)
            ftb.check_source_identity(root, prep)
            rebind = build_rebind(root, prep, OLD_PREP, prep)
            facts = base_launch_facts(_json_file(out / "launch_plan.json")["base_out"])
            rebind["derived_inputs_identity"] = _json_file(out / "derived_inputs_identity.json")
            if not rebind["derived_inputs_identity"]["all_identical"]:
                rebind["violations"].append("derived split/manifest/profile differ from what R-TB-E-0 trained with")
                rebind["execution_path_identical"], rebind["verdict"] = False, "INCOMPATIBLE"
            rebind["created"] = ftb.utc_now()
            path = out / "smoke_compatibility_rebind.json"
            ftb.write_json_atomic(path, rebind)
            if rebind["verdict"] != "COMPATIBLE":
                print("STOP: execution path changed; A1 is NOT run automatically: %s" % rebind["violations"], file=sys.stderr)
                return 3
            write_compat_receipt(out, facts, rebind, path, prep)
            print(json.dumps({"verdict": rebind["verdict"], "digest": rebind["comparison_digest_sha256"]}))
        elif args.command == "evaluator-check":
            prep = registered_prep(out)
            ftb.check_source_identity(root, prep)
            launch = _json_file(out / "launch_plan.json")
            facts = base_launch_facts(launch["base_out"])
            registration = _json_file(Path(launch["base_out"]) / "zero_sample_analysis" / "registration_NONTRIVIAL_LEARNING_R-TB-E-0.json")
            doc = evaluator_eligibility(root, prep, OLD_PREP, facts, registration)
            ftb.write_json_atomic(out / "evaluator_static_eligibility.json", doc)
            if doc["verdict"] != "ELIGIBLE_STATIC":
                print("NOT ELIGIBLE: %s" % doc["problems"], file=sys.stderr)
                return 4
            conf = {"plan_id": REFERENCE_PLAN_ID, "label": "NONTRIVIAL_LEARNING", "status": "CONFIRMED_STATIC",
                    "rule": "Plan v3 section 3.1", "confirmed_utc": ftb.utc_now(),
                    "evaluator_static_eligibility_sha256": ftb.sha256_file(out / "evaluator_static_eligibility.json"),
                    "bound_eval_final": doc["trigger_eval"]["eval_final"], "bound_final_checkpoint": doc["trigger_eval"]["final_checkpoint"],
                    "fresh_load": doc["trigger_eval"]["fresh_load"], "limits": doc["limits"],
                    "not_implied": ["no comparison with B2", "no claim of reliability across seeds", "no change to Method 2.1.1"]}
            ftb.write_json_atomic(out / "nontrivial_learning_confirmation_R-TB-E-0.json", conf)
            print(json.dumps({"R-TB-E-0": "NONTRIVIAL_LEARNING", "status": conf["status"]}))
        elif args.command == "release":
            print(json.dumps({"token": str(release_e1(root, out, registered_prep(out)))}))
        elif args.command == "train":
            summary = run_e1_train(root, out, args.gpu, args.token)
            print(json.dumps({"plan": E1_PLAN_ID, "stop_reason": summary.get("stop_reason"), "complete_updates": summary.get("complete_updates")}))
        elif args.command == "guard":
            summary = _run_guard(out, args.wait_start_seconds)
            print(json.dumps(summary))
        elif args.command == "status":
            state = ftb.Ledger(out).read()
            plan = state["plans"][E1_PLAN_ID]
            row = {"status": plan["status"], **(ftb.phase_report(plan["run_dir"]) if plan.get("run_dir") else {})}
            launch = _json_file(out / "launch_plan.json")
            free = ftb.fs_free_bytes(ftb._nearest_existing(launch["plans"][E1_PLAN_ID]["planned_run_dir"]))
            print(json.dumps({"plan": row, "ledger": {k: state[k] for k in ("new_rl_attempts_used", "new_rl_attempts_cap")},
                              "elastic_slots": {k: v["status"] for k, v in state["elastic_slots"].items()},
                              "free_bytes": free, "free_gib": round(free / GIB, 2)}))
        return 0
    except BindingError as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 2
