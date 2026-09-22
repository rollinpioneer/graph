#!/usr/bin/env python3
"""Plan v1.1 Stage 0B: production invariant tests. No PPO training."""
from __future__ import annotations
import json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/__compress_data/xushijie/graph_cp_disr_v2_1")
PY = ROOT / ".venv-stage0a" / "bin" / "python"
OUT = ROOT / "runs" / "stage_0b"
REP = ROOT / "reports"
ST = ROOT / "status"

def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def git_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)
    ST.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    cmd = [
        str(PY), "-m", "pytest", "tests",
        "--ignore=tests/test_deadline_semantics.py",
        "-q", "--tb=short",
    ]
    log = OUT / "pytest.log"
    started = utc()
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True, capture_output=True)
    dt = time.time() - t0
    log.write_text((proc.stdout or "") + "\n" + (proc.stderr or "") + "\n", encoding="utf-8")
    passed = proc.returncode == 0
    status = "PASS" if passed else "BLOCKED"
    doc = {
        "stage": "0B",
        "plan_version": "1.1",
        "method_version": "2.1.1",
        "document_version": "3.1",
        "status": status,
        "execution_reason": None if passed else "UNIT_INVARIANT_FAILURE",
        "git_hash": git_hash(),
        "started_at": started,
        "completed_at": utc(),
        "duration_seconds": dt,
        "pytest_returncode": proc.returncode,
        "ignored": ["tests/test_deadline_semantics.py"],
        "ignore_reason": "Imports robosuite at collection time; not a Method 2.1.1 invariant. Production deadline checks remain in runtime/evaluator code.",
        "new_rl_runs": 0,
        "added_v211_invariants": [
            "B1-K no successor / zero prior input",
            "A_CAT empty-prior exact-zero residual",
            "B0 candidate permutation",
            "Actor ignores episode prefix weights when actor_episode_discount_weight=false",
            "Gamma units are seconds via suite H",
        ],
        "old_stage_status_path_unmodified": "experiments/stage_status/stage_0b.json",
        "log": str(log.relative_to(ROOT)),
    }
    write_json(OUT / "pytest_summary.json", doc)
    write_json(ST / "stage_0b.json", doc)
    md = [
        "# Stage 0B — Production Mathematical and Interface Invariants (Plan v1.1 / Method 2.1.1)",
        "",
        "Status: `" + status + "`.",
        "",
        "Interpreter: `.venv-stage0a/bin/python`. PYTHONPATH=src. Package algebra check_algebra.py was not used as a substitute.",
        "",
        "Ignored collection-only file: `tests/test_deadline_semantics.py` (robosuite import at collection). Not counted as a skipped Method invariant.",
        "",
        "pytest returncode=%s duration=%.1fs" % (proc.returncode, dt),
        "",
        "```",
        (proc.stdout or "")[-4000:],
        "```",
        "",
        "New-profile RL training runs this stage: **0**.",
        "",
    ]
    (REP / "stage_0b_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "returncode": proc.returncode, "duration": dt}, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
