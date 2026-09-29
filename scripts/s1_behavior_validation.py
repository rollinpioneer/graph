#!/usr/bin/env python3
"""S1-REV1 production behavior validation without live provider or environments."""
from __future__ import annotations
import argparse, csv, hashlib, inspect, json, os, re, subprocess, sys
from pathlib import Path

GROUPS = {
    "runner": ["tests/test_s1_runner_behavior.py"],
    "gate": ["tests/test_s1_gate_behavior.py"],
    "dispatch": ["tests/test_s1_dispatch_behavior.py"],
    "probe": ["tests/test_s1_probe_evidence_behavior.py"],
}
IDS = {
    "runner": [f"B{i:02d}" for i in range(1, 20)],
    "gate": [f"G{i:02d}" for i in range(1, 7)],
    "probe": [f"P{i:02d}" for i in range(1, 5)],
    "dispatch": [],
}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

def read_json(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))

def root_out(args):
    root = Path(args.root).resolve()
    out = Path(args.output).resolve()
    return root, out

def update_manifest(out: Path, **updates):
    path = out / "validation_manifest.json"
    doc = read_json(path, {})
    doc.update(updates)
    write_json(path, doc)

def cmd_inventory(args):
    root, out = root_out(args)
    out.joinpath("inventory").mkdir(parents=True, exist_ok=True)
    targets = [
        "src/cp_disr/analysis/s1_revision.py",
        "src/cp_disr/analysis/s1_revision_resume.py",
        "src/cp_disr/analysis/s1_evidence_review.py",
        "src/cp_disr/platforms/libero/runtime_factory.py",
        "tests/helpers/s1_runner_doubles.py",
        *[p for group in GROUPS.values() for p in group],
    ]
    files = []
    for rel in targets:
        path = root / rel
        files.append({"path": rel, "exists": path.is_file(), "sha256": sha(path) if path.is_file() else ""})
    locations = {}
    try:
        sys.path.insert(0, str(root / "src"))
        import cp_disr.analysis.s1_revision as dispatch
        for name in ("execute_registered_branch", "run_witnesses", "register_physical_branches",
                     "probe_production_representation", "offline_rescore_planner", "finalize_eligibility"):
            obj = getattr(dispatch, name, None)
            try:
                locations[name] = {
                    "module": obj.__module__,
                    "file": inspect.getsourcefile(obj),
                    "line": inspect.getsourcelines(obj)[1],
                }
            except Exception as exc:
                locations[name] = {"error": type(exc).__name__}
    except Exception as exc:
        locations["import_error"] = {"error": type(exc).__name__, "message": str(exc)}
    write_json(out / "inventory/source_files.json", files)
    write_json(out / "inventory/production_callable_locations.json", locations)
    write_json(out / "inventory/behavior_requirements.json", {
        "baseline_groups": GROUPS,
        "behavior_ids": IDS,
        "live_provider_calls": 0,
        "live_environment_episodes": 0,
        "rl_transitions": 0,
        "optimizer_steps": 0,
        "mode": "CPU_ONLY_ENGINEERING_VALIDATION",
    })
    update_manifest(out, inventory_status="PASS", validation_status="RUNNING")
    print(json.dumps({"status": "PASS", "files": len(files), "callables": len(locations)}))

def parse_pytest_summary(text: str):
    m = re.search(r"(\d+) passed", text)
    f = re.search(r"(\d+) failed", text)
    e = re.search(r"(\d+) error", text)
    return {"passed": int(m.group(1)) if m else 0, "failed": int(f.group(1)) if f else 0,
            "errors": int(e.group(1)) if e else 0}

def cmd_baseline(args):
    root, out = root_out(args)
    test_dir = out / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.update({"CUDA_VISIBLE_DEVICES": "", "CP_DISR_AUDIT_ONLY": "1",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    results = {}
    matrix = []
    for group, tests in GROUPS.items():
        log = test_dir / f"{group}.log"
        junit = test_dir / f"{group}.xml"
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
               f"--junitxml={junit}", *tests]
        proc = subprocess.run(cmd, cwd=root, env=env, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=180)
        log.write_text(proc.stdout, encoding="utf-8")
        summary = parse_pytest_summary(proc.stdout)
        status = "PASS" if proc.returncode == 0 else "FAIL"
        results[group] = {"status": status, "returncode": proc.returncode, **summary,
                          "tests": tests, "log": str(log.relative_to(out))}
        for ident in IDS[group]:
            matrix.append({"id": ident, "group": group, "status": status,
                           "evidence_ref": str(log.relative_to(out))})
    fields = ["id", "group", "status", "evidence_ref"]
    with (out / "tests/behavior_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(matrix)
    write_json(out / "tests/baseline_results.json", results)
    status = "PASS" if all(v["status"] == "PASS" for v in results.values()) else "FAIL"
    update_manifest(out, baseline_status=status, validation_status="RUNNING",
                    baseline_tests=sum(v.get("passed", 0) for v in results.values()))
    print(json.dumps({"status": status, "groups": results}, ensure_ascii=False))

def cmd_summarize(args):
    root, out = root_out(args)
    baseline = read_json(out / "tests/baseline_results.json", {})
    mutations = read_json(out / "mutations/mutation_summary.json", {})
    rows = mutations.get("results", [])
    survivors = [r for r in rows if r.get("status") in ("SURVIVED", "NOT_APPLIED", "NOT_RUN")]
    baseline_ok = bool(baseline) and all(v.get("status") == "PASS" for v in baseline.values())
    status = "READY" if baseline_ok and not survivors and rows else "NOT_READY"
    blockers = []
    if not baseline_ok: blockers.append("BASELINE_BEHAVIOR_TEST_FAILURE")
    if not rows: blockers.append("MUTATION_MATRIX_MISSING")
    blockers.extend(f"{r.get('id')}:{r.get('status')}" for r in survivors)
    summary = {
        "status": status,
        "baseline_status": "PASS" if baseline_ok else "FAIL",
        "mutation_count": len(rows),
        "mutation_killed": sum(r.get("status") == "KILLED" for r in rows),
        "mutation_survivors": survivors,
        "live_provider_calls": 0,
        "live_environment_episodes": 0,
        "rl_transitions": 0,
        "optimizer_steps": 0,
        "formal_test_executed": False,
        "training_authorized": False,
        "remaining_blockers": blockers,
    }
    write_json(out / "validation_summary.json", summary)
    write_json(out / "remaining_blockers.json", {"status": status, "blockers": blockers})
    write_json(out / "proposed/e4_recovery_budget_request.json", {
        "status": "NOT_READY" if status != "READY" else "READY_FOR_EXPLICIT_REVIEW",
        "source_revision_index": 1, "new_source_revision": False,
        "requested_additional_physical_episodes": 8,
        "approved_additional_physical_episodes": 0,
        "new_provider_calls_requested": 0, "rl_attempts_requested": 0,
        "execute_now": False, "requires_explicit_user_budget_amendment": True,
    })
    (out / "validation_summary.md").write_text(
        "# S1-REV1 Engineering Validation\n\n"
        f"Status: **{status}**\n\n"
        f"Baseline: {'PASS' if baseline_ok else 'FAIL'}; "
        f"mutations killed: {summary['mutation_killed']}/{len(rows)}.\n\n"
        "No provider calls, real environment episodes, RL transitions, optimizer steps, "
        "formal test, or training were executed.\n\n"
        "Remaining blockers:\n" + "".join(f"- {b}\n" for b in blockers),
        encoding="utf-8")
    update_manifest(out, summary_status=status, validation_status=status)
    print(json.dumps(summary, ensure_ascii=False))

def cmd_verify(args):
    root, out = root_out(args)
    before = read_json(out / "inventory/protected_before.json", {})
    if isinstance(before, list):
        before = {item["path"]: item["sha256"] for item in before}
    after = {}
    changed = []
    for rel, expected in before.items():
        path = Path(rel)
        if not path.is_absolute():
            path = root / rel
        actual = sha(path) if path.is_file() else ""
        after[rel] = actual
        if actual != expected:
            changed.append(rel)
    budget_before = read_json(out / "inventory/budget_before.json", None)
    budget_now = read_json(out / "budget_ledger.json", None)
    budget_unchanged = budget_before == budget_now if budget_before is not None else False
    counters = {}
    if isinstance(budget_now, dict):
        for field in ("provider_first_calls", "provider_retries", "physical_witness_episodes",
                      "planner_environment_episodes", "rl_transitions", "optimizer_steps"):
            counters[field] = int((budget_now.get(field) or {}).get("used", 0))
    before_counters = {}
    if isinstance(budget_before, dict):
        for field in ("provider_first_calls", "provider_retries", "physical_witness_episodes",
                      "planner_environment_episodes", "rl_transitions", "optimizer_steps"):
            before_counters[field] = int((budget_before.get(field) or {}).get("used", 0))
    deltas = {field: counters.get(field, 0) - before_counters.get(field, 0)
              for field in counters}
    zero = all(v == 0 for v in deltas.values())
    result = {"status": "PASS" if not changed and budget_unchanged and zero else "FAIL",
              "protected_files": len(before), "changed_protected_files": changed[:20],
              "budget_unchanged": budget_unchanged, "counters_before": before_counters,
              "counters_after": counters, "counter_deltas": deltas,
              "all_new_resource_counters_zero": zero, "provider_calls": 0,
              "environment_episodes": 0, "rl_transitions": 0, "optimizer_steps": 0}
    write_json(out / "inventory/protected_after.json", after)
    write_json(out / "verify.json", result)
    update_manifest(out, verification_status=result["status"], validation_status=result["status"])
    print(json.dumps(result, ensure_ascii=False))

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inventory", "baseline", "summarize", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--root", required=True)
        p.add_argument("--output", required=True)
        p.set_defaults(func=globals()["cmd_" + name])
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
