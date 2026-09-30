"""Derived Family B gates: fake artifacts exercise rejection, not evidence."""
from pathlib import Path

from cp_disr.analysis import family_b_pilot as p

ROOT = Path(__file__).resolve().parents[1]
CFG = p.config(ROOT, "configs/final_master/s4_family_b_staging.yaml")


def _technical_fixture(out):
    p.freeze(ROOT, CFG, out)
    p.write(out / "physical/budget_ledger.json",
            {"physical_witness_episodes": {"cap": 24, "used": 4}})
    for b in p._registered(out, "technical"):
        bid = b["branch_id"]
        suffix = (["a:PICK:obj_b:v1", "a:PLACE:obj_b:receiver:v1"]
                  if b["context"] == "B_PENDING" else
                  ["a:PICK:obj_c:v1", "a:PLACE:obj_c:receiver:v1"])
        actions = [{"action_id": aid} for aid in
                   b["prefix"]+[b["candidate_id"]]+suffix]
        p.write(out / "physical/branch_results" / f"{bid}.json", {
            "branch_id": bid, "status": "TASK_SUCCESS", "task_success": True,
            "actions": actions, "recorder_errors": [],
            "env_counts": {"reset_calls": 1}})
        base = out / "captures" / bid
        p.write(base / "boundary/public.json", {
            "fact_values": {"p:Held:carrier": "TRUE"},
            "candidate_ids": ["a:PLACE_BUFFER:carrier:pad_u:v1",
                              "a:PLACE_BUFFER:carrier:pad_v:v1"],
            "candidate_mask": [True, True]})
        p.write(base / "boundary/qa_state.json", {
            "qa_only": True, "qpos": [0., 1.], "qvel": [0., 0.]})
        for i in range(6):
            d = base / f"action_{i:02d}"
            d.mkdir(exist_ok=True)
            for name in ("before_rgb.png", "after_rgb.png", "before_depth.npy",
                         "after_depth.npy", "facts.json", "evaluator.json",
                         "perception.json"):
                (d / name).touch()
            if 3 <= i < 5:
                (d / "planner.json").touch()


def test_terminal_planner_mutation_fails_technical_gate(tmp_path):
    _technical_fixture(tmp_path)
    assert p.check_technical(ROOT, CFG, tmp_path)["status"] == "PASS"
    bid = p._registered(tmp_path, "technical")[0]["branch_id"]
    (tmp_path / "captures" / bid / "action_05/planner.json").touch()
    amended = p.check_technical(ROOT, CFG, tmp_path)
    assert amended["status"] == "FAIL"
    assert any("terminal planner" in x for x in amended["problems"])


def test_provider_fake_pipeline_is_validated_without_api(tmp_path):
    result = p.provider_preflight(ROOT, CFG, tmp_path)
    assert result["status"] == "PASS"
    assert result["live_requests"] == 0
    assert result["accepted_synthetic_relation"] == 1


def test_mechanism_gate_keeps_failure_time_null(tmp_path):
    p.freeze(ROOT, CFG, tmp_path)
    p.write(tmp_path / "physical/budget_ledger.json",
            {"physical_witness_episodes": {"cap": 24, "used": 24}})
    for b in p._registered(tmp_path):
        cost = 5.0 if b["candidate"] == "pad_u" else 4.0
        p.write(tmp_path / "physical/branch_results" / f'{b["branch_id"]}.json', {
            "task_success": True, "status": "TASK_SUCCESS",
            "time_to_task_success": cost, "actions": [
                {"stage": "setup", "duration_sim": 1.0}],
            "initial_sim_time": 0.0, "terminal_sim_time": 9.0})
    result = p.analyze_physical(ROOT, CFG, tmp_path)
    assert result["physical_mechanism_status"] == "UNRESOLVED"
    assert result["cost_reversal_pass"] is False
    bid = p._registered(tmp_path)[0]["branch_id"]
    row = p.read(tmp_path / "physical/branch_results" / f"{bid}.json")
    row.update(task_success=False, status="NO_PLAN", time_to_task_success=None)
    p.write(tmp_path / "physical/branch_results" / f"{bid}.json", row)
    result = p.analyze_physical(ROOT, CFG, tmp_path)
    assert result["physical_mechanism_status"] == "UNRESOLVED"
    assert result["complete_success_pairs"] == 11


def test_relation_case_real_payload_path_accepts_new_task(tmp_path):
    from PIL import Image
    from cp_disr.analysis import family_b_provider as provider
    from cp_disr.platforms.libero.family_b_runtime import VERIFIER_FACT_IDS
    from cp_disr.vlm_provider import validate_payload
    p.freeze(ROOT, CFG, tmp_path)
    branch = next(b for b in p._registered(tmp_path)
                  if b["layout"] == "layout_0" and b["repeat"] == 0
                  and b["context"] == "B_PENDING" and b["candidate"] == "pad_u")
    cap = tmp_path / "captures" / branch["branch_id"]
    cap.mkdir(parents=True)
    Image.new("RGB", (64, 64), (128, 128, 128)).save(cap / "initial_rgb.png")
    p.write(cap / "initial_public_facts.json", {"facts": {
        fid: {"value": "UNKNOWN", "reason": "synthetic-test"}
        for fid in VERIFIER_FACT_IDS}})
    case = provider.relation_case(ROOT, CFG, tmp_path, "layout_0", 0)
    validate_payload(case["payload"])
    assert case["template"].goals
    assert case["manifest"]["task_id"] == "T_P_FB"
    assert case["input_refs"]["hidden_truth_included"] is False


def test_offline_production_policy_accepts_family_b_candidate_patch():
    import torch
    from cp_disr.common import digest
    from cp_disr.facts import FactRecord, FactStore, Truth
    from cp_disr.neural import Policy
    from cp_disr.rl import Snapshot
    from cp_disr.platforms.libero.snapshot import SnapshotBuilder
    from cp_disr.platforms.libero.family_b_runtime import build_task_template, VERIFIER_FACT_IDS
    template = build_task_template(ROOT / CFG["runtime"]["contract_path"])
    values = {fid: Truth.FALSE for fid in VERIFIER_FACT_IDS}
    values.update({
        "p:Open:receiver": Truth.TRUE, "p:Held:carrier": Truth.TRUE,
        "p:OnTable:obj_b": Truth.TRUE, "p:Inside:obj_c:receiver": Truth.TRUE})
    facts = FactStore(tuple(FactRecord(fid, value, 0.0, 0.0)
                            for fid, value in values.items()))
    ids, mask, features = SnapshotBuilder(template, (), None, None, 60)._mask(facts)
    edges = (("a:PLACE_BUFFER:carrier:pad_u:v1", "a:PICK:obj_b:v1", "SOFT_SUPPORTS"),)
    snap = Snapshot(
        env_id="offline", episode_id="offline", decision_id=3,
        template=template, facts=facts, candidate_ids=ids, mask=mask,
        prior_edges=edges, prior_hash=digest(edges), base_input=(0.,)*48,
        candidate_features=features, observation_ref="offline", clock_seconds=0.)
    assert [ids[i] for i, m in enumerate(mask) if m] == list(CFG["candidate_ids"])
    from cp_disr.analysis.family_b_reference_bank import _plan_options, _q_r
    planned, options = _plan_options(snap, 3.649999999999709)
    assert planned.status == "PLAN_FOUND"
    assert len(options) == 2
    assert len({o["cost"] for o in options}) == 1
    relation = [("a:PLACE_BUFFER:carrier:pad_u:v1", "a:PICK:obj_b:v1",
                 "SOFT_SUPPORTS", "p:AtBuffer:carrier:pad_u")]
    scores = {o["candidate_id"]: _q_r(o, relation, {"p:Inside:obj_b:receiver"})
              for o in options}
    assert scores[CFG["candidate_ids"][0]] > scores[CFG["candidate_ids"][1]]
    actions = sorted({n.schema for n in template.nodes if n.kind == "ACTION"})
    predicates = sorted({n.schema for n in template.nodes if n.kind == "PROPOSITION"})
    types = sorted({t for n in template.nodes for t in n.argument_types})
    for method in ("Full", "A_STAT", "B2", "A_CAT", "A_Q"):
        torch.manual_seed(650001)
        model = Policy(actions, predicates, types, 48, 8, method=method).eval()
        out = model(snap)
        assert tuple(out.candidate_ids) == ids
        assert bool(out.mask[ids.index(CFG["candidate_ids"][0])])
        assert bool(out.mask[ids.index(CFG["candidate_ids"][1])])
        assert out.diagnostics["successor_used"] is True
        if method == "B2":
            assert all(float(torch.count_nonzero(x)) == 0
                       for x in out.diagnostics["prior_inputs"].values())



def test_stopped_closeout_assembles_not_run_evidence(tmp_path):
    from cp_disr.analysis.family_b_closeout import assemble, TABLES
    _technical_fixture(tmp_path)
    branches = p._registered(tmp_path, "technical")
    for b in branches:
        result_path = tmp_path / "physical/branch_results" / f'{b["branch_id"]}.json'
        row = p.read(result_path)
        row["time_to_task_success"] = 6.0
        row["initial_sim_time"] = 0.0
        row["terminal_sim_time"] = 6.0
        for index, action in enumerate(row["actions"]):
            action.update(stage="setup" if index < 3 else "continuation", duration_sim=1.0)
        p.write(result_path, row)
    p.write(tmp_path / "physical/attempt_registry.json",
            {b["branch_id"]: "COMPLETED" for b in branches})
    p.write(tmp_path / "budget_ledger.json", {
        "branch_attempts": {"cap": 24, "used": 4},
        "explicit_resets": {"cap": 24, "used": 4},
        "live_skill_calls": {"cap": 144, "used": 24},
        "rl_transitions": {"cap": 0, "used": 0},
        "optimizer_steps": {"cap": 0, "used": 0},
    })
    p.write(tmp_path / "physical/technical_gate.json",
            {"status": "FAIL", "attempts": 4})
    result = p.analyze_physical(ROOT, CFG, tmp_path)
    assert result["physical_mechanism_status"] == "UNRESOLVED"
    review = assemble(ROOT, CFG, tmp_path)
    assert review["tp_training_authorized"] is False
    assert review["recommended_next_action"] == "DO_NOT_PROMOTE_FAMILY_B_WITHOUT_NEW_REVIEW"
    assert all((tmp_path / "evidence" / name).is_file() for name in TABLES)
