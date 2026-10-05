"""CP-DISR-C1-MECH-CONFIRM-V1: launch binding, Fresh Confirm isolation and frozen-suite checks (offline, no torch / simulator needed except where noted)."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_plan_table_is_three_qmark_attempts_seed_major_and_one_worker():
    from cp_disr import final_tb_c1 as C1
    assert sorted(C1.PLAN_TABLE) == ["R-TB-C1-QMARK-0", "R-TB-C1-QMARK-1", "R-TB-C1-QMARK-2"]
    assert {v["method"] for v in C1.PLAN_TABLE.values()} == {"B1-K+QMARK"}
    assert [C1.PLAN_TABLE["R-TB-C1-QMARK-%d" % s]["seed"] for s in (0, 1, 2)] == [0, 1, 2]
    assert C1.MAX_ATTEMPTS == 3 and C1.MAX_WORKERS == 1 and C1.INITIAL_RELEASE == ("R-TB-C1-QMARK-0",)


def test_assignment_validation_rejects_wrong_method_seed_or_plan():
    from cp_disr import final_tb_c1 as C1
    from cp_disr.common import BindingError
    assert C1.validate_assignment("R-TB-C1-QMARK-1", "B1-K+QMARK", 1)["seed"] == 1
    for args in (("R-TB-C1-QMARK-0", "B2", 0), ("R-TB-C1-QMARK-0", "B1-K+QMARK", 1), ("R-TB-C1-QMARK-0", "B1-K+QMARK", True), ("R-TB-C1-QMARK-3", "B1-K+QMARK", 3)):
        with pytest.raises(BindingError):
            C1.validate_assignment(*args)


def test_training_split_is_the_unchanged_struct_gen_file_and_never_a_fresh_confirm_file():
    from cp_disr import final_tb_c1 as C1
    assert str(C1.TRAIN_DEV_REL) == "configs/splits/struct_gen_v1_train_dev.json"
    assert not any(f.startswith("configs/splits/c1_") for f in C1.RELEASE_GATE_FILES)
    tree = ast.parse((ROOT / "src/cp_disr/final_tb_c1.py").read_text(encoding="utf-8"))
    consts = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not any("c1_fresh_confirm_v1_test" in c for c in consts)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "TEST_REL" not in names and "QUAL_REL" not in names


def test_context_refuses_a_fresh_confirm_split_and_old_methods():
    from cp_disr import final_tb as ftb
    from cp_disr import final_tb_c1 as C1
    from cp_disr.common import BindingError
    cfg = {k: "x" for k in ftb.REQUIRED_CONFIG_KEYS}
    cfg.update(plan_id="R-TB-C1-QMARK-0", attempt_id="a", method="B1-K+QMARK", training_seed=0, source_commit="a" * 40, output_directory="/o", runtime_manifest="/m", train_split="/s/c1_fresh_confirm_v1_test.json",
               prior_mode=ftb.PRIOR_MODE, study_envelope_ncap=ftb.N_CAP, study_envelope_tcap=ftb.TCAP_SECONDS, evaluation_rule=ftb.EVALUATION_RULE)
    with pytest.raises(BindingError):
        C1.context_from_dict(cfg)
    cfg["train_split"] = "/s/struct_gen_v1_train_dev.json"
    assert C1.context_from_dict(cfg).method == "B1-K+QMARK"
    cfg["method"] = "B2"
    with pytest.raises(BindingError):
        C1.context_from_dict(cfg)


def test_ledger_caps_three_attempts_one_worker_and_refuses_duplicates(tmp_path):
    from cp_disr import final_tb_c1 as C1
    from cp_disr.common import BindingError
    ledger = C1.C1Ledger(tmp_path)
    ledger.init({p: {"method": "B1-K+QMARK", "family": "QMARK", "seed": v["seed"], "status": "NOT_STARTED", "queue_order": v["queue_order"]} for p, v in C1.PLAN_TABLE.items()})
    ledger.reserve("R-TB-C1-QMARK-0", "a0", tmp_path / "r0", 1, 0)
    with pytest.raises(BindingError):
        ledger.reserve("R-TB-C1-QMARK-0", "a0", tmp_path / "r0b", 1, 0)           # duplicate start
    with pytest.raises(BindingError):
        ledger.reserve("R-TB-C1-QMARK-1", "a1", tmp_path / "r1", 2, 1)            # one worker at a time


def test_qmark_registration_is_idempotent_and_does_not_touch_other_methods():
    from cp_disr import c1_qmark_policy as Q
    calls = []
    v11 = SimpleNamespace(make_policy=lambda template, method, device: calls.append(method) or "legacy", EMPTY_PRIOR_METHODS={"B2"})
    Q.register(v11)
    first = v11.make_policy
    Q.register(v11)
    assert v11.make_policy is first and Q.QMARK_METHOD in v11.EMPTY_PRIOR_METHODS and "B2" in v11.EMPTY_PRIOR_METHODS
    assert v11.make_policy(None, "B2", None) == "legacy" and calls == ["B2"]


# ----------------------------------------------------------------------------- the frozen Fresh Confirm suite
def _suite():
    from cp_disr import c1_fresh_confirm as F
    return json.loads((ROOT / F.QUAL_REL).read_text()), json.loads((ROOT / F.TEST_REL).read_text())


def test_fresh_confirm_counts_cells_and_isolation():
    from cp_disr import c1_fresh_confirm as F
    qual, test = _suite()
    q, t = qual["qualification"], test["test"]
    assert len(q) == 8 and len(t) == 32 and not qual["test"] and not test["qualification"]
    for cell in F.CELLS:
        assert sum(r["cell"] == cell for r in q) == 2 and sum(r["cell"] == cell for r in t) == 8
    assert {r["case_id"] for r in q}.isdisjoint({r["case_id"] for r in t})
    old = F.old_rows(ROOT)
    old_poses = {F._pose(r) for r in old}
    assert not ({F._pose(r) for r in q + t} & old_poses)
    assert len({F._pose(r) for r in q + t}) == 40 and not ({F._pose(r) for r in q} & {F._pose(r) for r in t})
    assert not ({r["seed"] for r in q + t} & F.old_seeds_anywhere(ROOT) - {r["seed"] for r in q + t})
    assert all(r["case_id"].startswith("C1_") and r["lid_closed"] and r["second_role"] == "second_object" for r in q + t)


@pytest.fixture
def fresh_goal_sets():
    from cp_disr import c1_fresh_confirm as F
    F.install_goal_sets()
    yield
    F.uninstall_goal_sets()          # leave the old structural-generalization tables exactly as the frozen module defines them


def test_importing_the_fresh_confirm_module_does_not_change_the_old_suite_tables():
    from cp_disr import c1_fresh_confirm  # noqa: F401
    from cp_disr import struct_gen as G
    assert "BUF_T" not in G.GOAL_SETS and "BUF_T" not in G.EXPECTED_DEPTH


def test_fresh_confirm_goal_sets_depths_and_unseen_bindings(fresh_goal_sets):
    from cp_disr import c1_fresh_confirm as F
    from cp_disr import struct_gen as G
    assert G.GOAL_SETS["BUF_T"] == (G.BUF_T,) and G.EXPECTED_DEPTH["BUF_T"] == 2
    assert {c: G.EXPECTED_DEPTH[c] for c in F.CELLS} == {"IN_S": 3, "BUF_T": 2, "BUF_T+BUF_S": 4, "IN_S+BUF_T": 5}
    train_bindings = {("target", "container"), ("second_object", "buffer")}
    for cell in F.CELLS:
        assert {tuple(b) for b in G.atomic_bindings(G.GOAL_SETS[cell])} - train_bindings


def test_frozen_hashes_match_the_manifest():
    from cp_disr import c1_fresh_confirm as F
    manifest = json.loads((ROOT / "runs/final_master/c1_route_b/mech_confirm_v1/prep/split_manifest.json").read_text())
    for rel, digest in manifest["file_sha256"].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel
    assert set(manifest["file_sha256"]) == {str(F.QUAL_REL), str(F.TEST_REL)}


def test_old_split_files_are_unmodified():
    import subprocess
    out = subprocess.run(["git", "diff", "--stat", "5a2b3d18d21cbab19b9d09c3c430b3203e873730", "--", "configs/splits/struct_gen_v1_train_dev.json", "configs/splits/struct_gen_v1_test.json"],
                         cwd=str(ROOT), capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == ""
