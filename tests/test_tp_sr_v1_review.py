"""Tests for the zero-sample T_P_SR V1 review script (scripts/tp_sr_v1_review.py)."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tp_sr_v1_review.py"
spec = importlib.util.spec_from_file_location("tp_sr_v1_review", SCRIPT)
rv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rv)

SRC_REL = "runs/x/src_run"


def _w(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def make_fixture(tmp: Path):
    root = tmp / "repo"
    src = root / SRC_REL
    _w(root / "src/cp_disr/platforms/libero/skill_executor.py",
       "class E:\n    def _pick(self, trace, obj):\n        return 1\n\n    def _place(self, a):\n        return 2\n")
    _w(root / "src/cp_disr/platforms/libero/perception.py", "def _mask():\n    pass\n")
    _w(root / "src/cp_disr/platforms/libero/tp_sr_runtime.py", "def install_nearest_palette_mask():\n    perception._mask = _nearest_palette_mask\n")
    pool = ["case_id,occlusion_bin,axis,seed,approach_clearance_proxy,camera_projected_overlap,interferer_buffer_distance,"
            "target_container_distance,target_grasp_margin_proxy,target_interferer_xy_distance"]
    for i in range(8):
        pool.append(f"P{i},{i % 4},y,{i},{0.05 + i * 0.01},0.0,0.2,0.3,0.1,{0.11 + i * 0.01}")
    _w(src / "pool/pool_index.csv", "\n".join(pool) + "\n")
    opp = ["case_id,occlusion_bin,opportunity,reason,direct_success,reloc_success,direct_mean_time,reloc_mean_time,paired_restore",
           "P0,0,STRONG_HELPFUL,r,0/2,2/2,4.3,15.2,ok", "P1,1,HARMFUL,r,2/2,0/2,7,3.9,ok",
           "P2,2,HARMFUL,r,2/2,0/2,7,3.9,ok", "P3,3,UNKNOWN,r,2/2,0/2,7,3.9,ok"]
    _w(src / "opportunity/scene_opportunity.csv", "\n".join(opp) + "\n")
    brs = ["branch_id,case_id,route,repeat,occlusion_bin,attempt_state,execution_status,termination_reason,success,"
           "engineering_failure,controller_failures,sim_time,actions"]
    plan = {"P0": ("NO_PLAN", "NO_PLAN"), "P1": ("NO_PLAN", "NO_PLAN"), "P2": ("FAILED", "UNKNOWN_CONTROLLER_EXIT"),
            "P3": ("FAILED", "SYMBOLIC_EVALUATOR_MISMATCH")}
    n = 0
    for i, cid in enumerate(["P0", "P1", "P2", "P3"]):
        for route in ("direct", "relocation"):
            for rep in (0, 1):
                bid = f"b{n:03d}"
                n += 1
                if route == "direct" or cid == "P0" and route == "relocation":
                    brs.append(f"{bid},{cid},{route},{rep},{i},COMPLETED,TERMINATED,TASK_SUCCESS,True,False,0,5.0,a")
                    ev = {"phase": "evaluator_complete", "reason": "TASK_SUCCESS", "task_success": True, "terminated": True,
                          "relative_time": 5.0}
                else:
                    st = "COMPLETED" if plan[cid][0] == "NO_PLAN" else "FAILED"
                    brs.append(f"{bid},{cid},{route},{rep},{i},{st},{plan[cid][1]},{plan[cid][1]},False,False,0,3.9,a")
                    ev = {"phase": "evaluator_complete", "reason": "CONTINUE", "task_success": False, "terminated": False,
                          "relative_time": 3.9}
                j = [{"phase": "preflight_complete", "relative_time": None},
                     {"phase": "action_complete", "controller_exit": "NORMAL_TERMINATION", "relative_time": 3.9},
                     {"phase": "verifier_complete", "verified_fact_count": 10, "relative_time": 3.9}, ev,
                     {"phase": "planner_return", "status": "NO_PLAN", "expanded_nodes": 1, "relative_time": 3.9}]
                _w(src / f"physical/witnesses/branch_{bid}.jsonl", "\n".join(json.dumps(e) for e in j) + "\n")
                _w(src / f"physical/branch_results/{bid}.json", json.dumps(
                    {"branch_id": bid, "controller_exit": "NORMAL_TERMINATION",
                     "trace": [{"controller_exit": "TIMEOUT" if cid == "P2" else "NORMAL_TERMINATION"}]}))
    _w(src / "physical/branch_results.csv", "\n".join(brs) + "\n")
    _w(src / "static/rows/P0.json", json.dumps({"qa_blob_xyz": [0, 0, 0]}))
    out = tmp / "out"
    (out).mkdir()
    return root, src, out


def run_all(root, src, out):
    for c in ("geometry", "branches", "assemble"):
        assert rv.main([c, "--source", str(src), "--output", str(out), "--root", str(root)]) == 0


def rows(p):
    return list(csv.DictReader(open(p)))


def tree_hash(d: Path):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}


def test_join_complete_and_repeats_not_configs(tmp_path):
    root, src, out = make_fixture(tmp_path)
    run_all(root, src, out)
    j = rows(out / "geometry/geometry_outcome_join.csv")
    assert len(j) == 8
    assert sum(r["physical_scene"] == "yes" for r in j) == 4
    assert sum(int(r["n_branches"]) for r in j) == 16
    sem = json.loads((out / "geometry/bin_semantics.json").read_text())
    assert sem["physical_scene_count"] == 4 and sem["pool_count"] == 8
    assert all("distance_bin_originally_named_occlusion" in r for r in j)


def test_zero_overlap_not_no_interference(tmp_path):
    root, src, out = make_fixture(tmp_path)
    run_all(root, src, out)
    sem = json.loads((out / "geometry/bin_semantics.json").read_text())
    assert sem["nonzero_projected_overlap_pool_count"] == 0
    assert sem["overlap_zero_implies_no_mechanical_interference"] is False
    assert "not a real gripper collision" in sem["approach_clearance_proxy_definition"].lower() or "not a real gripper" in sem["approach_clearance_proxy_definition"].lower()


def test_no_plan_and_normal_exit_not_physical_failure(tmp_path):
    root, src, out = make_fixture(tmp_path)
    run_all(root, src, out)
    r = [x for x in rows(out / "branches/termination_review.csv") if x["case_id"] == "P1" and x["route"] == "relocation"]
    assert r and all(x["causal_interpretation_status"] == "OBSERVED_NON_SUCCESS_AT_PLANNER_STOP" for x in r)
    assert all("PHYSICAL" not in x["causal_interpretation_status"] for x in rows(out / "branches/termination_review.csv"))
    assert all(x["raw_controller_exit"] == "NORMAL_TERMINATION" for x in r)


def test_provider_not_run_is_not_observed_zero(tmp_path):
    root, src, out = make_fixture(tmp_path)
    run_all(root, src, out)
    a = json.loads((out / "assembly.json").read_text())
    assert a["provider_status"] == "NOT_RUN" and a["provider_calls_observed"] == "NOT_RUN"
    assert a["representation_status"] == "NOT_RUN"


def test_missing_facts_not_filled(tmp_path):
    root, src, out = make_fixture(tmp_path)
    run_all(root, src, out)
    for r in rows(out / "branches/termination_review.csv"):
        for k in ("post_action_facts_ref", "post_action_rgb_ref", "post_action_depth_ref",
                  "controller_subphase_trace_ref", "physical_contact_evidence_ref"):
            assert r[k] == "NOT_RECORDED"
    g = json.loads((out / "branches/recording_gaps.json").read_text())
    assert g["never_filled_from"] and g["normal_termination_is_not_grasp_success"] is True
    # legal-but-unmapped raw code is flagged for mapping review, not physical failure
    p2 = [x for x in rows(out / "branches/termination_review.csv") if x["case_id"] == "P2" and x["route"] == "relocation"]
    assert all(x["causal_interpretation_status"] == "CONTROLLER_STATUS_MAPPING_REVIEW_REQUIRED" for x in p2)


def test_no_environment_provider_imports():
    assert rv.scan_imports(SCRIPT) == []
    import ast
    called = {n.func.id for n in ast.walk(ast.parse(SCRIPT.read_text())) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not called & {"run_physical_probe", "static_screen", "classify_opportunity", "call_provider", "probe_representation"}


def test_source_hashes_unchanged(tmp_path):
    root, src, out = make_fixture(tmp_path)
    before = tree_hash(root)
    run_all(root, src, out)
    assert tree_hash(root) == before


def test_no_output_releases_part_b(tmp_path):
    root, src, out = make_fixture(tmp_path)
    run_all(root, src, out)
    b = json.loads((out / "proposal/budget_request.json").read_text())
    assert b["status"] == "PROPOSED_NOT_AUTHORIZED" and b["approved_branch_attempts"] == 0
    assert b["execute_now"] is False and b["part_b_authorized"] is False
    assert b["requested_branch_attempts_total"] == 20 and b["s2_s3_formal_test_authorized"] is False
    assert b["provider_calls_requested"] == 0 and b["standalone_capture_resets_requested"] == 0
    (out / "protected_before.json").write_text(json.dumps({"sha256": {p.relative_to(root).as_posix(): rv.sha(p) for p in (root / SRC_REL).rglob("*") if p.is_file()}}))
    assert rv.main(["verify", "--source", str(src), "--output", str(out), "--root", str(root)]) == 0
    assert json.loads((out / "verify.json").read_text())["status"] == "PASS"
