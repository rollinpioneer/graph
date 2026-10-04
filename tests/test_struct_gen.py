"""CP-DISR-TB-STRUCT-GEN-V1: symbolic suite, audits, mutations, evaluator equivalence, representation parity (offline)."""
import copy
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def suite():
    from cp_disr import struct_gen as G
    manifest, contracts, cache = G.build_manifest(ROOT)
    return G, manifest, contracts, cache


# ----------------------------------------------------------------------------- depth / signatures
def test_depth_is_the_shortest_legal_dependency_chain(suite):
    G, manifest, contracts, cache = suite
    got = {k: v["depth"] for k, v in manifest["solutions"].items()}
    assert got == G.EXPECTED_DEPTH
    facts0 = G.initial_facts(contracts)
    for key, atoms in G.GOAL_SETS.items():
        assert G.reference_depth(contracts, facts0, atoms) == G.EXPECTED_DEPTH[key]


def test_depth_levels_are_strictly_ordered_and_present(suite):
    G, manifest, contracts, _ = suite
    audit = G.dependency_depth_audit(manifest, contracts)
    assert audit["verdict"] == "PASS" and audit["checks"]["levels_strictly_ordered"]
    assert {c["depth_level"] for c in manifest["cases"] if c["split"] == "test"} == {1, 2, 3}
    assert {c["depth_level"] for c in manifest["cases"] if c["split"] == "train"} == {1, 3}


def test_effect_composition_signature_is_stable_and_order_free(suite):
    G, manifest, contracts, cache = suite
    again, _, _ = G.build_manifest(ROOT)
    assert [c["effect_composition_signature"] for c in manifest["cases"]] == [c["effect_composition_signature"] for c in again["cases"]]
    plans = cache["IN_T+BUF_S"]["plans"]
    assert len(plans) > 1 and len({G.plan_multiset(p) for p in plans}) == 1
    assert G.effect_composition_signature(plans) == G.effect_composition_signature(list(reversed(plans)))
    assert G.effect_composition_signature(cache["IN_T"]["plans"]) != G.effect_composition_signature(cache["IN_S"]["plans"])


def test_binding_novelty_is_correct(suite):
    G, manifest, contracts, _ = suite
    audit = G.binding_novelty_audit(manifest, contracts)
    assert audit["verdict"] == "PASS"
    train = {tuple(b) for b in audit["train_atomic_bindings"]}
    assert ("target", "container") in train and ("second_object", "buffer") in train
    assert ("second_object", "container") not in train and ("target", "buffer") not in train
    for row in audit["cases"]:
        assert row["unseen_atomic_bindings"] and row["schemas_all_seen_in_train"]


def test_splits_have_no_duplicates_and_test_is_disjoint(suite):
    G, manifest, _, _ = suite
    cases = manifest["cases"]
    assert len({c["case_id"] for c in cases}) == len(cases) == 102
    assert len({c["seed"] for c in cases}) == len(cases)
    poses = {(tuple(c["geometry"]["target_xy"]), tuple(c["geometry"]["second_xy"])) for c in cases}
    assert len(poses) == len(cases)
    train_cells = {c["cell"] for c in cases if c["split"] in ("train", "dev")}
    test_cells = {c["cell"] for c in cases if c["split"] == "test"}
    assert not (train_cells & test_cells)
    assert [sum(1 for c in cases if c["split"] == s) for s in ("train", "dev", "test")] == [60, 12, 30]


def test_runtime_rows_use_the_frozen_t_b_layout_and_poses_are_legal(suite):
    G, manifest, _, _ = suite
    rows = G.generate_rows()
    for r in rows:
        assert r["container_xy"] == G.CONTAINER and r["buffer_xy"] == G.BUFFER and r["lid_closed"] is True and r["second_role"] == "second_object"
        assert G.legal_pose(r["target_xy"], r["second_xy"]) and not any(k.startswith("cache") for k in r)


def test_generator_constants_equal_the_frozen_t_b_generator():
    from cp_disr import struct_gen as G
    from cp_disr import stage2a_v11 as v11
    assert (G.CONTAINER, G.BUFFER, G.TARGET_BOX, G.SECOND_BOX, G.MIN_DIST) == (v11.CONTAINER, v11.BUFFER, v11.TARGET_BOX, v11.SECOND_BOX, v11.MIN_DIST)


# ----------------------------------------------------------------------------- shortcut gate + mutations
def test_shortcut_gate_passes_on_the_real_suite(suite):
    G, manifest, contracts, cache = suite
    audit = G.shortcut_audit(manifest, contracts, cache)
    assert audit["verdict"] == "PASS"
    assert audit["S1"]["optimal_on_all_cases"] == 0 and audit["S2"]["solving_all_cases"] == 0
    assert max(audit["feature_lookup_accuracy"].values()) <= 0.50 and audit["state_and_goal_lookup_accuracy"] == 1.0


def test_M1_candidate_zero_always_optimal_fails_the_shortcut_gate(suite):
    G, _, _, _ = suite
    degenerate = {"train": ("BUF_S",), "dev": ("BUF_S",), "test": ("BUF_S",)}  # one cell: a fixed candidate priority list is then optimal everywhere
    rows = G.generate_rows(per_cell={"train": 6, "dev": 2, "test": 4}, split_cells=degenerate)
    manifest, contracts, cache = G.build_manifest(ROOT, rows)
    audit = G.shortcut_audit(manifest, contracts, cache)
    assert audit["verdict"] == "FAIL" and not audit["checks"]["S1_no_fixed_candidate_id_priority_is_optimal_on_all_cases"]


def test_object_id_leakage_is_caught(suite):
    G, manifest, contracts, cache = suite
    leaky = copy.deepcopy(manifest)
    for c in leaky["cases"]:
        c["objects"] = {"case_object_%s" % c["cell"]: "object"}
    audit = G.shortcut_audit(leaky, contracts, cache)
    assert not audit["checks"]["S3_object_ids_do_not_determine_the_optimal_candidate"] and audit["verdict"] == "FAIL"


def test_M2_test_binding_in_train_fails_the_novelty_gate(suite):
    G, _, _, _ = suite
    leaky = {"train": G.TRAIN_CELLS + ("IN_S",), "dev": G.TRAIN_CELLS, "test": G.TEST_CELLS}
    rows = G.generate_rows(per_cell={"train": 4, "dev": 2, "test": 2}, split_cells=leaky)
    manifest, contracts, _ = G.build_manifest(ROOT, rows)
    audit = G.binding_novelty_audit(manifest, contracts)
    assert audit["verdict"] == "FAIL" and not audit["checks"]["test_goal_sets_absent_from_train"]


def test_M3_corrupted_depth_label_fails_the_depth_audit(suite):
    G, manifest, contracts, _ = suite
    bad = copy.deepcopy(manifest)
    for c in bad["cases"]:
        if c["cell"] == "IN_S+BUF_T":
            c["dependency_depth"] = 4
            c["depth_level"] = 2
    audit = G.dependency_depth_audit(bad, contracts)
    assert audit["verdict"] == "FAIL" and not audit["checks"]["all_depth_labels_match_independent_search"]


def test_split_files_on_disk_match_the_generator_and_keep_test_apart():
    from cp_disr import struct_gen as G
    rows = G.generate_rows()
    train_dev = json.loads((ROOT / "configs/splits/struct_gen_v1_train_dev.json").read_text())
    test = json.loads((ROOT / "configs/splits/struct_gen_v1_test.json").read_text())
    assert train_dev["train"] == [r for r in rows if r["split"] == "train"] and train_dev["dev"] == [r for r in rows if r["split"] == "dev"]
    assert train_dev["test"] == [] and test["test"] == [r for r in rows if r["split"] == "test"] and test["train"] == [] and test["dev"] == []
    ids_test = {r["case_id"] for r in test["test"]}
    assert not (ids_test & {r["case_id"] for r in train_dev["train"] + train_dev["dev"]})


def test_test_split_file_is_not_referenced_by_any_training_code_path():
    hits = []
    for path in (ROOT / "src").rglob("*.py"):
        if "struct_gen_v1_test" in path.read_text(errors="ignore"):
            hits.append(str(path.relative_to(ROOT)))
    assert hits == [], "src files may not open the structural test split: %s" % hits


# ----------------------------------------------------------------------------- frozen components unchanged
def test_reward_evaluator_controller_files_unchanged_against_the_seed_balance_result_commit():
    base = "b602dfd1bc6f1241ebed1a474d51e6f6875d52a2"
    protected = ["src/cp_disr/platforms/libero/task_evaluator.py", "src/cp_disr/platforms/libero/skill_executor.py", "src/cp_disr/platforms/libero/verifier.py",
                 "src/cp_disr/platforms/libero/perception.py", "src/cp_disr/platforms/libero/safety.py", "src/cp_disr/platforms/libero/clock.py", "src/cp_disr/platforms/libero/d0_env.py",
                 "src/cp_disr/platforms/libero/runtime_factory.py", "src/cp_disr/platforms/libero/snapshot.py", "src/cp_disr/platforms/libero/observations.py",
                 "src/cp_disr/rl.py", "src/cp_disr/collector.py", "src/cp_disr/torch_rl.py", "src/cp_disr/adapters.py", "src/cp_disr/neural.py", "src/cp_disr/repctl_policy.py",
                 "src/cp_disr/stage2a_v11.py", "src/cp_disr/phase_a_v12.py", "src/cp_disr/final_tb.py", "src/cp_disr/contracts.py", "src/cp_disr/graph.py",
                 "configs/runtime/stage_2a_contract_registry.yaml", "experiments/manifests/runtime_manifest_v211.yaml"]
    changed = subprocess.check_output(["git", "diff", "--name-only", "--diff-filter=MD", base, "--", *protected], cwd=str(ROOT), text=True).split()
    assert changed == [], changed


class _FakeEnv:
    second_role = "second_object"

    def __init__(self, truth):
        self._truth = truth

    def hidden_truth(self):
        return self._truth


def _truth(rng, *_ignored):
    import numpy as np
    z = 0.825 + 0.021
    c, b = np.array([0.18, 0.12, z]), np.array([-0.18, 0.12, z])

    def pos():
        where = rng.integers(0, 3)  # 0 inside container, 1 at buffer, 2 elsewhere
        base = [c, b, np.array([0.0, -0.1, z])][where]
        spread = [0.045, 0.075, 0.08][where]
        return base + np.array([rng.uniform(-spread, spread), rng.uniform(-spread, spread), rng.uniform(-0.02, 0.2) if where == 2 else rng.uniform(0.0, 0.14)])
    lid_far = rng.random() < 0.8
    return {"target": pos(), "second_object": pos(), "container": c, "buffer": b, "lid": c + (np.array([0.2, 0, 0]) if lid_far else np.array([0.0, 0, 0])), "table_top_z": 0.825}


def test_evaluator_wrapper_equals_task_evaluator_on_t_b_t_a_and_single_goal_modes():
    import numpy as np
    from cp_disr import struct_gen as G
    from cp_disr.struct_gen_runtime import StructGenEvaluator
    from cp_disr.platforms.libero.task_evaluator import TaskEvaluator
    rng = np.random.default_rng(0)
    agree = {"T_B": 0, "T_A": 0, "D0": 0}
    for _ in range(6000):
        truth = _truth(rng)
        env = _FakeEnv(truth)
        for mode, atoms in (("T_B", G.GOAL_SETS["IN_T+BUF_S"]), ("T_A", (G.IN_T, G.IN_S)), ("D0", G.GOAL_SETS["IN_T"])):
            ref = TaskEvaluator(env, 60.0, task_id=mode).goal_true()
            got = StructGenEvaluator(env, 60.0, atoms).goal_true()
            assert ref == got, (mode, truth)
            agree[mode] += int(ref)
    assert all(v > 0 for v in agree.values()), "the random truths must exercise both outcomes: %s" % agree


def test_evaluator_wrapper_new_goals_use_the_frozen_atomic_checks():
    import numpy as np
    from cp_disr import struct_gen as G
    from cp_disr.struct_gen_runtime import StructGenEvaluator
    from cp_disr.platforms.libero.task_evaluator import TaskEvaluator
    rng = np.random.default_rng(1)
    for _ in range(1500):
        truth = _truth(rng)
        env = _FakeEnv(truth)
        ref = TaskEvaluator(env, 60.0, task_id="T_B")
        for key, atoms in G.GOAL_SETS.items():
            expect = all((ref._inside(truth, "target" if ":target:" in a else "second_object") if ":Inside:" in a else ref._at_buffer(truth, "target" if ":target:" in a else "second_object")) for a in atoms)
            assert StructGenEvaluator(env, 60.0, atoms).goal_true() == expect


@pytest.mark.torch_runtime
def test_templates_differ_only_in_goals_and_keep_candidates_nodes_and_mask_rule():
    from cp_disr import tb_repctl_checks as C
    from cp_disr import struct_gen as G
    from cp_disr.graph import Goal, build_template
    from cp_disr.platforms.libero.runtime_factory import PREDICATES
    base = C.tb_template(ROOT)
    templates = {k: build_template(base.contracts, tuple(Goal(a, 1) for a in atoms), PREDICATES, G.OBJECTS) for k, atoms in G.GOAL_SETS.items()}
    ref = templates["IN_T+BUF_S"]
    assert ref.goals == base.goals and ref.nodes == base.nodes and ref.edges == base.edges
    for key, t in templates.items():
        assert t.nodes == ref.nodes and t.edges == ref.edges and t.contracts == ref.contracts and len(t.nodes) == 17
        assert {g.fact_id for g in t.goals} == set(G.GOAL_SETS[key])


@pytest.mark.torch_runtime
def test_representation_semantics_hold_on_every_goal_template():
    import torch
    from cp_disr import tb_repctl_checks as C
    from cp_disr import struct_gen as G
    from cp_disr.graph import Goal, build_template
    from cp_disr.platforms.libero.runtime_factory import PREDICATES
    torch.set_num_threads(2)
    base = C.tb_template(ROOT)
    for key, atoms in G.GOAL_SETS.items():
        template = build_template(base.contracts, tuple(Goal(a, 1) for a in atoms), PREDICATES, G.OBJECTS)
        states = C.state_suite(template, count=2)
        receipts = C.all_semantic_checks(template, states)
        assert receipts["A1_f_after_identical_to_b2"]["candidate_states_compared"] > 0 and receipts["N1_source_scan"]["hits"] == []


@pytest.mark.torch_runtime
def test_parameter_counts_do_not_depend_on_the_goal_template():
    from cp_disr import tb_repctl_checks as C
    from cp_disr import struct_gen as G
    from cp_disr.graph import Goal, build_template
    from cp_disr.platforms.libero.runtime_factory import PREDICATES
    base = C.tb_template(ROOT)
    counts = {}
    for key in ("IN_T", "IN_S+BUF_T"):
        template = build_template(base.contracts, tuple(Goal(a, 1) for a in G.GOAL_SETS[key]), PREDICATES, G.OBJECTS)
        seed, values = C.state_suite(template, count=1)[0]
        snap = C.make_snapshot(template, values, seed)
        counts[key] = {m: C.effective_parameters(C.make_policy(template, m, 0), snap)[0] for m in ("B1-K+E", "B2", "B2-ABS", "B1-K+NC")}
    assert counts["IN_T"] == counts["IN_S+BUF_T"]
    assert counts["IN_T"]["B2"] == counts["IN_T"]["B2-ABS"] == 849987 and counts["IN_T"]["B1-K+NC"] == 884291


def test_new_modules_import_no_provider_vlm_or_test_cache():
    import ast
    forbidden = {"vlm_provider", "vlm_cache_pipeline", "vlm", "prior", "stage2a_runner", "phase_a_v13_r1", "phase_a_v13_r3"}
    for rel in ("src/cp_disr/struct_gen.py", "src/cp_disr/struct_gen_runtime.py"):
        mods = set()
        for n in ast.walk(ast.parse((ROOT / rel).read_text())):
            if isinstance(n, ast.ImportFrom):
                mods.add((n.module or "").split(".")[-1])
                mods.update(a.name for a in n.names)
            elif isinstance(n, ast.Import):
                mods.update(a.name.split(".")[-1] for a in n.names)
        assert not (mods & forbidden), (rel, mods & forbidden)
    # the pure symbolic module must not pull in torch or the simulator
    src = (ROOT / "src/cp_disr/struct_gen.py").read_text()
    assert not re.search(r"^\s*(import|from)\s+(torch|robosuite|mujoco)", src, re.M)
