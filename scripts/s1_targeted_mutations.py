#!/usr/bin/env python3
"""Run targeted source mutations in isolated temporary copies only."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

TESTS = [
    "tests/test_s1_runner_behavior.py",
    "tests/test_s1_gate_behavior.py",
    "tests/test_s1_dispatch_behavior.py",
    "tests/test_s1_probe_evidence_behavior.py",
]
MUTATIONS = [
    ("M01", "src/cp_disr/analysis/s1_revision_resume.py",
     "refs = runtime.get(\"reference_skill_seconds_by_task\")",
     "refs = manifest.get(\"reference_skill_seconds_by_task\")", "nested manifest reference"),
    ("M02", "src/cp_disr/analysis/s1_revision_resume.py",
     "candidate = plan.plan[0]", "candidate = branch[\"candidate_id\"]", "drop first planned skill"),
    ("M03", "src/cp_disr/analysis/s1_revision_resume.py",
     "if task.terminated or task.truncated:",
     "if task.terminated or task.truncated or execution.controller_exit == \"NORMAL_TERMINATION\":",
     "controller exit treated as task success"),
    ("M04", "src/cp_disr/analysis/s1_revision_resume.py",
     "if candidate not in index or not bool(snap.mask[index[candidate]]):",
     "if candidate not in index:", "skip current action mask"),
    ("M05", "src/cp_disr/analysis/s1_revision_resume.py",
     "if elapsed >= deadline:", "if False:", "skip deadline before action"),
    ("M06", "src/cp_disr/analysis/s1_revision_resume.py",
     "elapsed = now - episode_start", "elapsed = 0.0", "reset elapsed at every decision"),
    ("M07", "src/cp_disr/analysis/s1_revision_resume.py",
     "if task.terminated or task.truncated:",
     "if task.terminated:", "ignore truncation boundary"),
    ("M08", "src/cp_disr/analysis/s1_revision_resume.py",
     "if decisions > 1024:", "if decisions > 6:", "restore hard-coded seven-action cap"),
    ("M09", "src/cp_disr/analysis/s1_revision_resume.py",
     'execution_status="SEARCH_TIMEOUT", termination_reason="SEARCH_TIMEOUT", protocol_complete=True, witness_validity="NOT_ESTABLISHED"',
     'execution_status="SEARCH_TIMEOUT", termination_reason="SEARCH_TIMEOUT", protocol_complete=True, witness_validity="NOT_ESTABLISHED", eligible_for_e4=True',
     "search timeout accepted as E4"),
    ("M10", "src/cp_disr/analysis/s1_revision_resume.py",
     'execution_status="SYMBOLIC_GOAL", termination_reason="SYMBOLIC_EVALUATOR_MISMATCH", protocol_complete=False',
     'execution_status="SYMBOLIC_GOAL", termination_reason="SYMBOLIC_EVALUATOR_MISMATCH", protocol_complete=True, task_success=True',
     "symbolic goal overrides evaluator"),
    ("M11", "src/cp_disr/analysis/s1_revision_resume.py",
     'bundle.start_case(branch["case_id"], restore_seed=int(seed))',
     'bundle.start_case(branch["case_id"])', "paired seed omitted at restore"),
    ("M12", "src/cp_disr/analysis/s1_revision_resume.py",
     'if int(item.get("used", 0)) >= int(item.get("cap", 0)):',
     'if False:', "budget checked after runtime creation"),
    ("M13", "src/cp_disr/analysis/s1_revision_resume.py",
     'if prior in ("STARTED", "COMPLETED", "UNKNOWN"):',
     'if False:', "duplicate attempt accepted"),
    ("M14", "src/cp_disr/analysis/s1_revision.py",
     "fcntl.flock(lock.fileno(), fcntl.LOCK_EX)",
     "fcntl.flock(lock.fileno(), fcntl.LOCK_SH)", "shared instead of exclusive budget lock"),
    ("M15", "src/cp_disr/analysis/s1_evidence_review.py",
     'and x.get("synthetic_unit_fixture") is not True and x.get("natural") is not False and x.get("provider") is not False',
     "and True", "synthetic/provider labels shortcut E6"),
    ("M16", "src/cp_disr/analysis/s1_evidence_review.py",
     "len(cases) >= 2 and len(effects) >= 1",
     "True", "incomplete E4 evidence accepted"),
    ("M17", "src/cp_disr/analysis/s1_evidence_review.py",
     'x.get("actual_comparison") is True and ',
     "", "alignment labels without actual comparison"),
    ("M18", "src/cp_disr/analysis/s1_evidence_review.py",
     'x.get("separate_sources") is True',
     "True", "gradient provenance omitted"),
    ("M19", "src/cp_disr/analysis/s1_revision_resume.py",
     '    except Exception as exc:\n        event("error", error_phase=phase',
     '    except KeyError as exc:\n        event("error", error_phase=phase',
     "bundle/runtime failures escape broad production boundary"),
    ("M20", "src/cp_disr/analysis/s1_revision_resume.py",
     '"runtime_constructed": False',
     '"runtime_constructed": True', "offline rescore claims runtime construction"),
]

def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

def run_one(root: Path, out: Path, item):
    ident, rel, old, new, rationale = item
    work = Path(tempfile.mkdtemp(prefix=f"s1_mut_{ident}_", dir="/tmp"))
    try:
        shutil.copytree(root / "src", work / "src")
        (work / "tests").mkdir()
        (work / "tests" / "__init__.py").write_text("", encoding="utf-8")
        shutil.copytree(root / "tests" / "helpers", work / "tests" / "helpers")
        for test in TESTS:
            source = root / test
            target = work / test
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        target = work / rel
        text = target.read_text(encoding="utf-8")
        occurrences = text.count(old)
        record = {"id": ident, "path": rel, "rationale": rationale,
                  "occurrences": occurrences, "isolated_copy": str(work)}
        if occurrences != 1:
            record.update({"status": "NOT_APPLIED", "returncode": None})
            write_json(out / "mutations" / ident / "result.json", record)
            return record
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(work / "src") + os.pathsep + str(work)
        env.update({"CUDA_VISIBLE_DEVICES": "", "CP_DISR_AUDIT_ONLY": "1",
                    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
                              cwd=work, env=env, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=180)
        log = out / "mutations" / ident / "pytest.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(proc.stdout, encoding="utf-8")
        record.update({"status": "KILLED" if proc.returncode != 0 else "SURVIVED",
                       "returncode": proc.returncode, "log": str(log.relative_to(out))})
        write_json(out / "mutations" / ident / "result.json", record)
        return record
    except subprocess.TimeoutExpired as exc:
        record = {"id": ident, "path": rel, "rationale": rationale,
                  "status": "KILLED", "returncode": "TIMEOUT",
                  "log": str((out / "mutations" / ident / "timeout.log").relative_to(out))}
        (out / "mutations" / ident).mkdir(parents=True, exist_ok=True)
        (out / "mutations" / ident / "timeout.log").write_text(str(exc), encoding="utf-8")
        write_json(out / "mutations" / ident / "result.json", record)
        return record
    except Exception as exc:
        record = {"id": ident, "path": rel, "rationale": rationale,
                  "status": "NOT_RUN", "error": type(exc).__name__, "message": str(exc)}
        write_json(out / "mutations" / ident / "result.json", record)
        return record
    finally:
        shutil.rmtree(work, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command != "run":
        return
    root, out = Path(args.root).resolve(), Path(args.output).resolve()
    results = [run_one(root, out, item) for item in MUTATIONS]
    summary = {
        "status": "PASS" if results and all(x["status"] == "KILLED" for x in results) else "NOT_READY",
        "mutation_count": len(results),
        "killed": sum(x["status"] == "KILLED" for x in results),
        "survived": sum(x["status"] == "SURVIVED" for x in results),
        "not_applied": sum(x["status"] == "NOT_APPLIED" for x in results),
        "not_run": sum(x["status"] == "NOT_RUN" for x in results),
        "results": results,
        "live_provider_calls": 0, "live_environment_episodes": 0,
        "rl_transitions": 0, "optimizer_steps": 0,
    }
    write_json(out / "mutations/mutation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    main()
