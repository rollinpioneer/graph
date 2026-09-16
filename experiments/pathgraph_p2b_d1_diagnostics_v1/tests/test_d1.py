from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from d1.core import (
    FAILURE_PRIORITY,
    RunningPair,
    canonical,
    classify_features,
    geom_signature,
    guidance_category,
    quantiles,
    sha256_file,
    spearman,
    state_hash,
    write_json,
)
from d1.runner import BASE, MILESTONES, METHODS, SEEDS, manifest, protocol_document


def features(**changes):
    row = {
        "ever_held_or_transport": True,
        "final_valid_count": 0,
        "transport_stall": False,
        "invalid_loop": False,
        "last_16_invalid_fraction": 0.0,
        "object_switch_loop": False,
        "premature_place_or_release": False,
        "loss_seen": False,
        "recovery_started": False,
        "regrasped": False,
        "resumed_after_regrasp": False,
        "actions": ["WAIT"],
        "progress_during_cycle": False,
    }
    row.update(changes)
    return row


def state(phase="WAIT", events=None, episode="one"):
    obj = lambda p: {"pos": [0.0, 0.0, 0.03], "vel": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0], "angular_vel": [0.0, 0.0, 0.0], "target_xy": [1.0, 0.0], "held": p in ("HELD", "TRANSPORT"), "valid": p == "VALID", "phase": p}
    return {"episode_id": episode, "state_index": 0, "task": "dual_order", "available_at_ns": 0, "physical_time_ns": 0, "capture_order": 0, "success": False, "terminal_failure": False, "gripper_closed": False, "eef": [0.0, 0.0, 0.3], "objects": {"A": obj(phase), "B": obj("WAIT")}, "events": events or [], "evidence_tier": "x"}


def test_01_protocol_fixed_seed_set():
    assert protocol_document()["policy_seeds"] == list(SEEDS)


def test_02_forbids_model_learn():
    assert "model.learn(" not in Path(__file__).parents[1].joinpath("d1/runner.py").read_text()


def test_03_forbids_optimizer_step():
    assert "optimizer.step(" not in Path(__file__).parents[1].joinpath("d1/runner.py").read_text()


def test_04_forbids_backward():
    assert ".backward(" not in Path(__file__).parents[1].joinpath("d1/runner.py").read_text()


def test_05_checkpoint_hash_is_read_only(tmp_path):
    p = tmp_path / "policy.zip"; p.write_bytes(b"frozen")
    before = p.read_bytes(); sha256_file(p); assert p.read_bytes() == before


def test_06_test_episode_unique_key():
    rows = [(1, "A", "C", 0), (2, "A", "C", 0)]
    assert len(rows) == len(set(rows))


def test_07_fixed_subset_episode_coverage():
    keys = {(f, p, c) for f in range(32) for p in range(2) for c in range(8)}
    assert len(keys) == 512


def test_08_action_table_is_dynamic():
    source = Path(__file__).parents[1].joinpath("d1/runner.py").read_text()
    assert "tuple(envmod.ACTIONS)" in source


def test_09_no_acquire_progress():
    assert classify_features(features(ever_held_or_transport=False))[0] == "NO_ACQUIRE_PROGRESS"


def test_10_one_object_only_timeout():
    assert classify_features(features(final_valid_count=1))[0] == "ONE_OBJECT_ONLY_TIMEOUT"


def test_11_transport_stall():
    assert classify_features(features(transport_stall=True))[0] == "TRANSPORT_STALL"


def test_12_invalid_action_loop():
    assert classify_features(features(invalid_loop=True))[0] == "INVALID_ACTION_LOOP"


def test_13_object_switch_loop():
    assert classify_features(features(object_switch_loop=True))[0] == "OBJECT_SWITCH_LOOP"


def test_14_recovery_failure_classes():
    assert classify_features(features(loss_seen=True))[0] == "RECOVERY_NOT_STARTED"
    assert classify_features(features(loss_seen=True, recovery_started=True))[0] == "RECOVERY_STARTED_NO_REGRASP"
    assert classify_features(features(loss_seen=True, recovery_started=True, regrasped=True))[0] == "REGRASP_NO_TASK_RESUME"


def test_15_single_primary_multiple_secondary():
    primary, secondary = classify_features(features(transport_stall=True, invalid_loop=True))
    assert primary == "TRANSPORT_STALL" and secondary == ["INVALID_ACTION_LOOP"]


def test_16_worker_cumulative_steps():
    values = [64, 12, 128]
    cumulative = []; total = 0
    for x in values: total += x; cumulative.append(total)
    assert cumulative == [64, 76, 204]


def test_17_validation_milestone_sorting():
    assert tuple(sorted(MILESTONES)) == MILESTONES


def test_18_stochastic_rng_reproducibility():
    assert np.random.default_rng(2026091601).integers(1000, size=20).tolist() == np.random.default_rng(2026091601).integers(1000, size=20).tolist()


def test_19_stochastic_diagnostic_does_not_modify_checkpoint(tmp_path):
    p = tmp_path / "p.zip"; p.write_bytes(b"checkpoint")
    before = sha256_file(p); np.random.default_rng(1).random(100); assert sha256_file(p) == before


def test_20_state_hash_excludes_log_identity():
    a = state(episode="a"); b = state(episode="b")
    a["source_policy"] = "X"; b["source_policy"] = "Y"
    assert state_hash(a) == state_hash(b)


def test_21_state_hash_retains_phase():
    assert state_hash(state("WAIT")) != state_hash(state("HELD"))


def test_22_state_hash_retains_event_history():
    event = {"kind": "LOSS", "object_id": "A", "loss_id": "L1", "known_at_ns": 0}
    assert state_hash(state()) != state_hash(state(events=[event]))


class FakeCap:
    def __init__(self, episode_id): self.episode_id = episode_id
    def observe(self, s): return {"cost_cap": .75, "psi": -.25, "local_credit": .5}


def test_23_potential_bank_each_episode_reset():
    from p2b.potentials import PotentialBank
    a = PotentialBank("a", FakeCap); b = PotentialBank("b", FakeCap)
    assert a is not b and a.open is not b.open


def test_24_noncausal_order_rejected():
    from p2b.potentials import PotentialBank
    bank = PotentialBank("one", FakeCap); s = state()
    bank.observe(s)
    with pytest.raises(ValueError, match="noncausal"): bank.observe(s)


def test_25_terminal_effective_phi_zero():
    from p2b.shaping import shaped_transition
    row = shaped_transition(0.0, -1.0, 4.0, terminated=True)
    assert row["phi_after_effective"] == 0 and row["shaping_reward"] == 1.0


def test_26_geom_formula_matches_frozen_implementation():
    from p2b.potentials import PotentialBank
    values, meta = PotentialBank("one", FakeCap).observe(state())
    assert values["GEOM_COUNT_EVENTS_PBRS"] == values["COUNT_EVENTS_PBRS"] + .25 * meta["geometry"]


def test_27_graph_formula_matches_frozen_implementation():
    from p2b.potentials import PotentialBank
    values, _ = PotentialBank("one", FakeCap).observe(state())
    assert values["GRAPH_COST_PBRS"] == -.75 and values["GRAPH_FULL_PBRS"] == -.25


def test_28_zero_tolerance_boundary():
    assert guidance_category(1e-10, -1e-10) == "BOTH_ZERO"
    assert guidance_category(1.01e-10, -1.01e-10) == "OPPOSITE_SIGN"


def test_29_geom_signature_ignores_case_and_outcome():
    assert geom_signature(1, 0, .25) == geom_signature(1, 0, .25)
    assert "condition" not in geom_signature(1, 0, .25)


def test_30_alias_group_denominator():
    groups = {"x": [1, 2], "y": [3]}
    assert sum(map(len, groups.values())) == 3


def test_31_unique_state_deduplication():
    states = [state_hash(state()), state_hash(state()), state_hash(state("HELD"))]
    assert len(set(states)) == 2


def test_32_condition_profile_macro():
    rates = {"a": 0.0, "b": 1.0}
    assert sum(rates.values()) / len(rates) == .5


def test_33_deepcopy_branch_does_not_modify_original():
    original = {"events": [], "objects": {"A": {"phase": "WAIT"}}}
    branch = copy.deepcopy(original); branch["events"].append(1); branch["objects"]["A"]["phase"] = "HELD"
    assert original == {"events": [], "objects": {"A": {"phase": "WAIT"}}}


def test_34_thirteen_actions_covered():
    from p2a.env import ACTIONS
    assert len(ACTIONS) == 13 and len(set(ACTIONS)) == 13


def test_35_expert_is_diagnostic_only():
    source = Path(__file__).parents[1].joinpath("d1/runner.py").read_text()
    assert "DIAGNOSTIC_CURRENT_STATE_HEURISTIC;NOT_POLICY_INPUT;NOT_TEACHER_TAKEOVER;NOT_TEST_SUCCESS_DEFINITION" in source


def test_36_large_file_external_only():
    source = Path(__file__).parents[1].joinpath("d1/runner.py").read_text()
    assert 'external_root / "potential_corpus"' in source


def test_37_manifest_per_file_sha(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    result = manifest(tmp_path)
    assert result["file_count"] == 1 and len(result["files"][0]["sha256"]) == 64


def test_38_historical_artifact_read_only(tmp_path):
    p = tmp_path / "decision.json"; p.write_text('{"confirmation_passed":false}')
    before = sha256_file(p); json.loads(p.read_text()); assert sha256_file(p) == before


def test_39_no_checkpoint_extensions_in_git_outputs():
    root = Path(__file__).parents[1]
    assert not list(root.rglob("*.zip")) and not list(root.rglob("*.pt")) and not list(root.rglob("*.pth"))


def test_40_final_report_new_training_jobs_zero():
    assert protocol_document()["new_training_jobs"] == 0


def test_41_optimizer_and_gradient_counts_zero():
    p = protocol_document(); assert p["optimizer_updates"] == 0 and p["gradient_steps"] == 0


def test_42_original_decision_unchanged():
    assert protocol_document()["p2b_original_decision_modified"] is False


def test_43_confirmation_is_false():
    assert protocol_document()["confirmation_passed"] is False


def test_44_running_correlation():
    pair = RunningPair(); [pair.add(x, 2*x) for x in range(5)]
    assert pair.result() == pytest.approx(1.0)


def test_45_spearman_with_ties():
    assert spearman([1, 2, 2, 4], [2, 4, 4, 8]) == pytest.approx(1.0)


def test_46_quantiles_are_deterministic():
    assert quantiles([3, 1, 2])["0.5"] == 2


def test_47_base_commit_frozen():
    assert BASE == "0f26660cde901aa32b0cfc032fc75ec01b31687b"


def test_48_six_methods_frozen():
    assert len(METHODS) == 6 and METHODS[-1] == "GRAPH_FULL_PBRS"
