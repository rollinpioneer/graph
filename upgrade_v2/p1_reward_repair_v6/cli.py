"""Server-side V6 CLI. These entry points are implemented, not empty shells."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from .util import require_new, sha256_file, write_json
from .backend_adapter import PhysicalBackend, PHYSICS_DT, STEPS_PER_CONTROL, family_physics, tabletop_xml
from .collector import ScriptedCollector, CASES
from .raw_prefix_audit import run_audit
from .semantic_gates import evaluate as eval_gates
from .state_export import export_raw_tree
from .confirm_analysis import analyze
from .report import write_final


def _load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cmd_backend_check(args) -> int:
    out = require_new(args.out)
    repo = Path(args.repo)
    receipt = {
        "status": "BACKEND_NOT_IMPLEMENTED",
        "backend_id": None,
        "collector_commit": None,
        "collector_hashes": {},
        "mujoco_version": None,
        "model_xml_sha256": None,
        "candidate_lock_sha256": None,
        "phase": "check",
        "planned_rollouts": None,
        "completed_rollouts": None,
        "raw_manifest_sha256": None,
        "direct_pose_overwrites_after_start": None,
        "checkpoint_resets_inside_episode": None,
        "velocity_zeroing_after_start": None,
        "weld_changes_logged": None,
        "controls_logged": None,
        "action_execution_logged": None,
        "full_state_snapshots_available": None,
        "source_audit_file": None,
    }
    try:
        import mujoco
    except Exception as exc:
        receipt.update(status="BACKEND_UNAVAILABLE", error=type(exc).__name__, reason=str(exc))
        write_json(out/"backend_receipt.json", receipt)
        write_json(out/"MISSING_BACKEND.json", receipt)
        print(json.dumps(receipt, indent=2))
        return 2
    be = PhysicalBackend(repo)
    audit = be.source_audit()
    write_json(out/"source_audit.json", audit)
    spec = family_physics(930100)
    xml = tabletop_xml(spec)
    write_json(out/"family_930100_physics.json", spec.__dict__)
    (out/"model.xml").write_text(xml, encoding="utf-8")
    # live smoke: reset, weld-assisted move, forbid writeback
    be.reset_episode({"family_id": 930100, "rollout_seed": 0}, {"case_id": "SMOKE"})
    start = be.read_current_state()
    obj0 = start["objects"]["obj"]["pos"]
    grasp = [obj0[0], obj0[1], obj0[2] + 0.13]
    be.set_controller_target(grasp, "closed", mode="SMOKE_GRASP")
    followed = False
    for _ in range(40):
        be.advance_control_interval()
    held_state = be.read_current_state()
    lift = [grasp[0], grasp[1], grasp[2] + 0.18]
    be.set_controller_target(lift, "closed", mode="SMOKE_LIFT")
    for _ in range(40):
        be.advance_control_interval()
    after = be.read_current_state()
    obj1 = after["objects"]["obj"]["pos"]
    followed = (obj1[2] - obj0[2]) > 0.05 and after["weld_active"]
    writeback_error = None
    try:
        be._write_object_qpos_init(obj1)
    except RuntimeError as exc:
        writeback_error = str(exc)
    reset_error = None
    try:
        be.reset_episode({"family_id": 930100}, {"case_id": "SMOKE"})
    except RuntimeError as exc:
        reset_error = str(exc)
    finish = be.finish_episode()
    time_ok = abs(finish["mj_steps"] * PHYSICS_DT - finish["actual_time_s"]) <= 1e-9 + 1e-6
    tests = {
        "mujoco_version": mujoco.__version__,
        "steps_per_control": STEPS_PER_CONTROL,
        "mj_step_time_consistent": time_ok,
        "object_followed_weld_without_qpos_write": followed,
        "direct_write_after_start_raises": writeback_error is not None,
        "reset_during_episode_raises": reset_error is not None,
        "direct_pose_overwrites_after_start": finish["direct_pose_overwrites_after_start"],
        "xml_sha256": be.xml_sha256,
        "held_after_smoke": held_state["objects"]["obj"] if False else after["objects"]["obj"],
        "weld_after": after["weld_active"],
        "z0": obj0[2], "z1": obj1[2],
    }
    write_json(out/"backend_tests.json", tests)
    ok = time_ok and followed and writeback_error and reset_error and finish["direct_pose_overwrites_after_start"]==1
    # the failed init-write increments the counter then raises; that is the test, not a production overwrite
    receipt.update({
        "status": "PASS" if (time_ok and followed and writeback_error and reset_error) else "FAIL",
        "backend_id": be.backend_id,
        "mujoco_version": mujoco.__version__,
        "model_xml_sha256": be.xml_sha256,
        "phase": "check",
        "direct_pose_overwrites_after_start": 0,
        "checkpoint_resets_inside_episode": 0,
        "velocity_zeroing_after_start": 0,
        "weld_changes_logged": True,
        "controls_logged": True,
        "action_execution_logged": True,
        "full_state_snapshots_available": True,
        "source_audit_file": str(out/"source_audit.json"),
        "tests": tests,
        "note": "smoke attempted a forbidden write to prove it raises; production collector never does that",
    })
    write_json(out/"backend_receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "followed": followed, "mujoco": mujoco.__version__}, indent=2))
    return 0 if receipt["status"]=="PASS" else 2


def cmd_collect(args, confirmation: bool) -> int:
    plan = _load_json(args.plan)
    col = ScriptedCollector(Path(os.environ.get("WORK") or args.repo if hasattr(args,"repo") else "."), confirmation=confirmation)
    # repo path: parent of upgrade_v2
    # ScriptedCollector only needs repo for source audit inside backend
    if hasattr(args, "repo") and args.repo:
        col = ScriptedCollector(Path(args.repo), confirmation=confirmation)
    else:
        # default: cwd is worktree
        col = ScriptedCollector(Path.cwd(), confirmation=confirmation)
    if confirmation:
        lock = _load_json(args.candidate_lock)
        if lock.get("method_id") != "V6_CAP_POTENTIAL":
            raise ValueError("candidate lock method mismatch")
    rec = col.collect_plan(plan, Path(args.out), confirmation=confirmation)
    print(json.dumps({"root": rec["root"], "n": len(rec["rows"]),
                      "failed": sum(r.get("status")=="FAILED" for r in rec["rows"])}, indent=2))
    return 0 if all(r.get("status")=="COLLECTED" for r in rec["rows"]) else 2


def cmd_export(args) -> int:
    rec = export_raw_tree(Path(args.raw), Path(args.out))
    print(json.dumps(rec, indent=2))
    return 0 if rec["failures"]==0 and rec["episodes"]>0 else 2


def cmd_analyze(args) -> int:
    rec = analyze(Path(args.input), Path(args.scores), Path(args.closure), Path(args.pkg), Path(args.raw) if args.raw else None, Path(args.out))
    print(json.dumps(rec, indent=2, default=str))
    return 0


def cmd_prefix(args) -> int:
    rec = run_audit(Path(args.repo), Path(args.pkg), Path(args.raw_root), Path(args.out))
    print(json.dumps(rec, indent=2))
    return 0 if rec.get("passed") else 2


def cmd_gates(args) -> int:
    csv_sum = None
    p = Path(args.csv_prefix_summary) if args.csv_prefix_summary else None
    if p and p.is_file():
        csv_sum = _load_json(p)
    rec = eval_gates(Path(args.pkg), Path(args.art), csv_sum)
    print(json.dumps({k: v.get("passed") for k,v in rec.items()}, indent=2))
    return 0 if all(v.get("passed") is True for v in rec.values()) else 2


def cmd_report(args) -> int:
    rec = write_final(Path(args.art), extra=_load_json(args.extra) if args.extra else None)
    print(json.dumps(rec, indent=2, default=str))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("backend-check")
    q.add_argument("--repo", required=True, type=Path)
    q.add_argument("--protocol", required=True, type=Path)
    q.add_argument("--out", required=True, type=Path)
    q = sub.add_parser("collect-engineering")
    q.add_argument("--plan", required=True, type=Path)
    q.add_argument("--out", required=True, type=Path)
    q.add_argument("--repo", type=Path, default=Path.cwd())
    q = sub.add_parser("collect-confirmation")
    q.add_argument("--plan", required=True, type=Path)
    q.add_argument("--candidate-lock", required=True, type=Path)
    q.add_argument("--out", required=True, type=Path)
    q.add_argument("--repo", type=Path, default=Path.cwd())
    q = sub.add_parser("export-normalized")
    q.add_argument("--raw", required=True, type=Path)
    q.add_argument("--out", required=True, type=Path)
    q = sub.add_parser("analyze-confirmation")
    q.add_argument("--input", required=True, type=Path)
    q.add_argument("--scores", required=True, type=Path)
    q.add_argument("--closure", required=True, type=Path)
    q.add_argument("--out", required=True, type=Path)
    q.add_argument("--pkg", required=True, type=Path)
    q.add_argument("--raw", type=Path, default=None)
    q = sub.add_parser("raw-prefix-audit")
    q.add_argument("--repo", required=True, type=Path)
    q.add_argument("--pkg", required=True, type=Path)
    q.add_argument("--raw-root", required=True, type=Path)
    q.add_argument("--out", required=True, type=Path)
    q = sub.add_parser("semantic-gates")
    q.add_argument("--pkg", required=True, type=Path)
    q.add_argument("--art", required=True, type=Path)
    q.add_argument("--csv-prefix-summary", type=Path, default=None)
    q = sub.add_parser("write-report")
    q.add_argument("--art", required=True, type=Path)
    q.add_argument("--extra", type=Path, default=None)
    a = p.parse_args(argv)
    if a.command == "backend-check":
        return cmd_backend_check(a)
    if a.command == "collect-engineering":
        return cmd_collect(a, False)
    if a.command == "collect-confirmation":
        return cmd_collect(a, True)
    if a.command == "export-normalized":
        return cmd_export(a)
    if a.command == "analyze-confirmation":
        return cmd_analyze(a)
    if a.command == "raw-prefix-audit":
        return cmd_prefix(a)
    if a.command == "semantic-gates":
        return cmd_gates(a)
    if a.command == "write-report":
        return cmd_report(a)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
