import subprocess
import sys

import pytest

from cp_disr.analysis import tp_libero_audit as a

BDDL = """
(define (problem LIBERO_Test)
  (:domain robosuite)
  (:language put both the soup and the butter in the basket)
  (:regions
    (soup_region (:target main_table) (:ranges ((-0.1 -0.1 -0.05 -0.05))) (:yaw_rotation ((0.0 0.0))))
    (contain_region (:target basket_1))
  )
  (:fixtures main_table - table)
  (:objects soup_1 - alphabet_soup butter_1 - butter basket_1 - basket)
  (:obj_of_interest soup_1 basket_1)
  (:init (On soup_1 main_table_soup_region) (Open cab_1_top_region))
  (:goal (And (In soup_1 basket_1_contain_region) (In butter_1 basket_1_contain_region)))
)
"""


def test_registry_key_rule_matches_upstream():
    assert a.snake("Chefmate8Frypan") == "chefmate_8_frypan"
    assert a.snake("AkitaBlackBowl") == "akita_black_bowl"
    assert a.snake("FlatStove") == "flat_stove"


def test_parse_bddl_sections():
    p = a.parse_bddl(BDDL)
    assert p["language"].startswith("put both")
    assert p["fixtures"] == [("main_table", "table")]
    assert ("soup_1", "alphabet_soup") in p["objects"] and ("basket_1", "basket") in p["objects"]
    assert [x[0] for x in p["goal_atoms"]] == ["In", "In"] and p["goal_ops"] == ["and"]
    assert p["regions"][("main_table", "soup_region")]["ranges"] == [[-0.1, -0.1, -0.05, -0.05]]
    assert p["regions"][("main_table", "soup_region")]["yaw"] == [[0.0, 0.0]]
    assert ("basket_1", "contain_region") in p["regions"]
    inst, sect = a.inst_maps(p)
    res = a.resolve("basket_1_contain_region", p, inst, sect)
    assert res["instance"] == "basket_1" and res["region"] == "contain_region" and res["type"] == "basket"


def test_guard_refuses_demos_scores_and_binary_suffixes(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("guard_dir")
    g = a.Guard()
    ok = tmp_path / "x.bddl"
    ok.write_text("(a)")
    assert g.read(ok) == "(a)"
    for name in ("demo_01.hdf5", "benchmark_score.json", "task_results.json", "init.pruned_init"):
        with pytest.raises(PermissionError):
            g.read(tmp_path / name)
    assert len(g.refused) == 4


def asset(ext, geoms=1):
    return {"resolved": True, "collision_extent_xyz": list(ext), "collision_geoms_used": geoms}


def test_grasp_class_rules():
    assert a.grasp_class(asset((0.04, 0.04, 0.04)))["class"] == "COMPACT"
    assert a.grasp_class(asset((0.04, 0.04, 0.04)))["strict_binding"] is True
    bowl = a.grasp_class(asset((0.107, 0.107, 0.051), 40))
    assert bowl["class"] == "NOT_PINCHABLE" and bowl["optimistic_class"] == "NOT_PINCHABLE" and "standing" not in bowl["hypotheses"]
    milk = a.grasp_class(asset((0.053, 0.131, 0.053), 4))
    assert milk["class"] == "YAW_DEPENDENT" and milk["strict_binding"] is False and milk["binding_if_orientation_verified"] is False
    bottle = a.grasp_class(asset((0.043, 0.044, 0.157), 21))
    assert bottle["class"] == "YAW_DEPENDENT" and bottle["strict_binding"] is False
    assert bottle["optimistic_class"] == "COMPACT" and bottle["binding_if_orientation_verified"] is True and bottle["optimistic_hypothesis"] == "standing"
    flat_box = a.grasp_class(asset((0.017, 0.04, 0.076), 1))
    assert flat_box["strict_binding"] is False and flat_box["class"] == "YAW_DEPENDENT"


def test_open_action_joint_rules():
    assets = {"drawer": {"joints": [{"type": "slide"}]}, "door": {"joints": [{"type": "hinge"}]}, "none": {"joints": []}}
    assert a.open_action({"type": "drawer", "ref": "d"}, assets, "x")["class"] == "NEW_SCRIPTED_SKILL_REQUIRED"
    assert a.open_action({"type": "door", "ref": "m"}, assets, "x", close=True)["class"] == "NEW_LOW_LEVEL_CONTROLLER_REQUIRED"
    assert a.open_action({"type": "none", "ref": "n"}, assets, "x")["class"] == "NOT_MAPPABLE"


def test_hard_precondition_ignores_initially_open_container():
    closed = [{"skill": "PICK", "source": {"support_kind": "DRAWER_INTERIOR", "container_open_initially": False}}]
    opened = [{"skill": "PICK", "source": {"support_kind": "DRAWER_INTERIOR", "container_open_initially": True}}]
    assert a.hard_precondition(closed, None, None)
    assert not a.hard_precondition(opened, None, None)


def test_rect_gap_ranges():
    (dxm, dxM), (dym, dyM) = a.rect_gap([0, 0, 0.02, 0.02], [0.1, 0, 0.12, 0.02])
    assert abs(dxm - 0.08) < 1e-9 and abs(dxM - 0.12) < 1e-9 and dym == 0.0 and abs(dyM - 0.02) < 1e-9


def facts(**kw):
    f = {"n_legal_first_actions": 2, "legal_first_actions_note": "x", "hard_precondition": False, "hard_precondition_note": "", "contract_gives_winner": False, "contract_note": "tie",
         "persistent_effect": {"material": True, "note": "n"}, "downstream_dependence": {"value": True, "note": "n"}, "path_only": False, "not_path_note": "n", "relation_expressible": True, "relation_note": "r",
         "relation_is_contract_renaming": False, "renaming_note": "n", "covers": {"classes": ["helpful", "neutral", "harmful"], "established": True, "note": "n"}, "fixed_rule_solves": False, "fixed_rule_note": "n",
         "skills": {"worst_class": "CURRENT_SKILL_WITH_NEW_BINDING", "platform_ok": True, "note": "n"}, "verifier": {"status": "PASS", "note": "n"}, "evaluator": {"status": "PASS", "note": "n"},
         "restore": {"status": "PASS", "note": "n"}, "canary_episodes": 4}
    f.update(kw)
    return f


def test_gates_all_pass_only_for_a_clean_card():
    g = a.evaluate_gates(facts())
    assert all(v["status"] == "PASS" for v in g.values()) and set(g) == set(a.GATES)


@pytest.mark.parametrize("override,gate_id", [({"hard_precondition": True, "hard_precondition_note": "x"}, "G2"), ({"path_only": True}, "G5"), ({"fixed_rule_solves": True}, "G9"),
                                              ({"skills": {"worst_class": "NEW_SCRIPTED_SKILL_REQUIRED", "platform_ok": True, "note": "n"}}, "G10"),
                                              ({"skills": {"worst_class": "CURRENT_SKILL_WITH_NEW_BINDING", "platform_ok": False, "note": "n"}}, "G10"),
                                              ({"n_legal_first_actions": 1}, "G1"), ({"canary_episodes": 5}, "G14"),
                                              ({"persistent_effect": {"material": False, "note": "n"}}, "G3"), ({"covers": {"classes": ["neutral"], "established": True, "note": "n"}}, "G8")])
def test_each_hard_gate_blocks(override, gate_id):
    assert a.evaluate_gates(facts(**override))[gate_id]["status"] == "FAIL"


def test_unverifiable_effect_blocks_selection_but_is_not_a_fail():
    g = a.evaluate_gates(facts(persistent_effect={"material": None, "note": "n"}, downstream_dependence={"value": None, "note": "n"}))
    assert g["G3"]["status"] == g["G4"]["status"] == "UNVERIFIABLE_STATICALLY"
    assert not all(v["status"] == "PASS" for v in g.values())


def test_attacks_flag_fixed_rule_and_contract():
    c = facts(fixed_rule_solves=True, contract_gives_winner=True, origin="UNMODIFIED_PUBLIC_TASK")
    r = a.attacks(c)
    assert r["A1_fixed_rule"]["status"] == "FIXED_HEURISTIC_SOLVABLE" and r["A2_contract_planner"]["status"] == "CONTRACT_SOLVABLE"
    assert r["A5_memorisation"]["status"] == "MEMORISABLE_FIXED_SCENES"


def test_module_import_pulls_no_simulator_stack():
    code = "import sys; import cp_disr.analysis.tp_libero_audit as m; bad=[x for x in ('robosuite','mujoco','libero','torch','gym') if x in sys.modules]; print(bad)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr


def test_public_validation_rates():
    tasks = [{"suite": "libero_10", "index": i, "name": "t%d" % i, "coverage": c, "coverage_if_orientation_verified": c2, "actions": [{"skill": "PICK", "class": c, "reasons": ["r"]}]}
             for i, (c, c2) in enumerate([("CURRENT_SKILL_WITH_NEW_BINDING",) * 2, ("NEW_SCRIPTED_SKILL_REQUIRED", "CURRENT_SKILL_WITH_NEW_BINDING"), ("NEW_LOW_LEVEL_CONTROLLER_REQUIRED",) * 2])]
    rows, s = a.public_validation(tasks, {})
    assert s["total_tasks"] == 3 and s["binding_supported"] == 1 and s["exactly_supported"] == 0 and s["unsupported"] == 2
    assert s["label_if_future_run"] == "LIBERO_COMPATIBLE_SUBSET" and abs(s["upper_bound_if_orientation_verified"] - 2 / 3) < 1e-3


def test_hard_precondition_needs_a_gated_placement():
    pure_open = [{"skill": "OPEN", "object": "cab_middle"}]
    assert a.hard_precondition(pure_open, None, None) == []
    gated = [{"skill": "PICK", "source": None}, {"skill": "PLACE", "dest": "cab_top", "dest_kind": "DRAWER_INTERIOR"}, {"skill": "OPEN", "object": "cab_top"}]
    flags = a.hard_precondition(gated, None, None)
    assert any("gates a placement" in f for f in flags) and any("destination inside a closed articulated container" in f for f in flags)


def test_baseline_identity_uses_branch_ref(tmp_path_factory):
    repo = tmp_path_factory.mktemp("gitrepo")

    def git(*args):
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    git("init", "-q")
    (repo / "f.txt").write_text("1")
    git("add", "f.txt")
    git("commit", "-q", "-m", "base")
    git("branch", "-M", "codex/cp-disr-tp-vis-preflight")
    base = git("rev-parse", "HEAD")
    git("checkout", "-q", "-b", "codex/cp-disr-tp-libero-opportunity-audit")
    (repo / "g.txt").write_text("2")
    git("add", "g.txt")
    git("commit", "-q", "-m", "engineering")
    head = git("rev-parse", "HEAD")
    ident = a.baseline_identity(repo, head)
    assert ident["baseline_head_full"] == base != head and len(base) == 40
    assert ident["run_head_full"] == head and ident["run_head_descends_from_baseline"] is True
    assert ident["baseline_full_matches_expected"] is False


def test_protected_hash_exclusion_matches_output_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("proot")
    own = root / "runs/final_master/S4/tp_libero_opportunity_audit/run1"
    other = root / "runs/final_master/S4/tp_vis_preflight/run1"
    own.mkdir(parents=True)
    other.mkdir(parents=True)
    (own / "a.json").write_text("{}")
    (other / "b.json").write_text("{}")
    assert a.OWN_OUTPUT.search("runs/final_master/S4/tp_libero_opportunity_audit/x/a.json")
    assert not a.OWN_OUTPUT.search("runs/final_master/S4/tp_vis_preflight/x/a.json")
    hashed = a.protected_hashes(root)
    assert list(hashed) == ["runs/final_master/S4/tp_vis_preflight/run1/b.json"]


def test_capacity_counts_duplicate_item_types():
    assets = {"flat_stove": {"sites": {"cook_region": {"pos": [0, 0, 0], "size": [0.075, 0.075, 0.0025]}}}, "moka_pot": {"resolved": True, "collision_extent_xyz": [0.08, 0.15, 0.15]}}
    one = a.capacity("flat_stove", "cook_region", ["moka_pot"], assets)
    two = a.capacity("flat_stove", "cook_region", ["moka_pot", "moka_pot"], assets)
    assert abs(two["fill_ratio_max"] - 2 * one["fill_ratio_max"]) < 1e-6 and two["fill_ratio_max"] > 1.0
    assert a.capacity("flat_stove", "cook_region", [], assets)["fill_ratio_max"] == 0.0

def test_fill_template_cases_are_labels_and_decisions_claim_nothing_unaudited():
    card = dict(facts(), card_id="X", origin="UNMODIFIED_PUBLIC_TASK", tasks=["t1"], covers={"classes": ["neutral"], "established": False, "note": "some note"})
    tp = a.fill_template(card)
    assert tp["neutral_case"] == "present" and tp["helpful_case"] == "absent" and tp["reversed_or_harmful_case"] == "absent"
    assert "some note" in tp["case_coverage_note"] and "not established statically" in tp["case_coverage_note"]
    joined = " ".join(a.DECISIONS)
    assert "does not help" not in joined and "not audited here" in joined