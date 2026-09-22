"""Stage 2A-P0 unit gates that do not require robosuite or API."""
from pathlib import Path
import pytest
import yaml

from cp_disr.stage2a_runner import planned_jobs, Stage2ARunner, validate_runner


@pytest.mark.pure
def test_move_rejected_in_stage_2a_registry():
    root = Path(__file__).resolve().parents[1]
    actions = yaml.safe_load((root / "configs/runtime/stage_2a_action_registry.yaml").read_text(encoding="utf-8"))
    contracts = yaml.safe_load((root / "configs/runtime/stage_2a_contract_registry.yaml").read_text(encoding="utf-8"))
    assert "MOVE" not in actions["allowed_schemas"]
    assert "MOVE" in actions["rejected_schemas"]
    assert actions["move_runtime_bound"] is False
    assert actions["move_alias_of_place_buffer"] is False
    names = [c["name"] for c in contracts["contracts"]]
    assert "MOVE" not in names
    assert set(names) == {"OPEN", "PICK", "PLACE", "PLACE_BUFFER"}
    for task_id, ids in actions["grounded_action_ids"].items():
        assert all(not s.startswith("a:MOVE:") for s in ids), task_id


@pytest.mark.pure
def test_stage_2a_jobs_unique_and_complete():
    jobs = planned_jobs()
    assert len(jobs) == 24
    assert len({j["run_id"] for j in jobs}) == 24
    assert len({j["output_dir"] for j in jobs}) == 24
    assert len({j["config_hash"] for j in jobs}) == 24
    combos = {(j["task_id"], j["method"], j["seed"]) for j in jobs}
    expected = {(t, m, s) for t in ("T_A", "T_C") for m in ("B0", "B1", "B2", "Full") for s in (0, 1, 2)}
    assert combos == expected
    assert all(j["planned_transitions"] == 65536 for j in jobs)
    assert all(j["planned_updates"] == 64 for j in jobs)
    assert all(j["test_access"] is False for j in jobs)
    assert all(j["share_optimizer"] is False for j in jobs)


@pytest.mark.pure
def test_stage_2a_resume_and_no_learning_probe(tmp_path):
    jobs = planned_jobs()
    runner = Stage2ARunner(tmp_path, jobs, learning_enabled=False)
    probe = runner.no_learning_probe(jobs[0]["run_id"])
    assert probe["resume_ok"] is True
    assert probe["optimizer_steps"] == 0
    assert probe["ppo_update_blocked"] is True
    with pytest.raises(Exception):
        runner.ppo_update(jobs[0]["run_id"])


@pytest.mark.pure
def test_validate_runner_writes_artifacts(tmp_path):
    (tmp_path / "experiments/part_2_exploration/stage_2a_p0").mkdir(parents=True)
    report = validate_runner(tmp_path)
    assert report["passed"] is True
    assert (tmp_path / "experiments/part_2_exploration/stage_2a_p0/planned_run_registry.csv").exists()
    assert (tmp_path / "experiments/part_2_exploration/stage_2a_p0/resource_plan.yaml").exists()
@pytest.mark.pure
def test_validate_runner_twice_does_not_overwrite_attempt(tmp_path):
    (tmp_path / "experiments/part_2_exploration/stage_2a_p0").mkdir(parents=True)
    first = validate_runner(tmp_path)
    second = validate_runner(tmp_path)
    assert first["passed"] is True
    assert second["passed"] is True
    assert first["probe"]["attempt_dir"] != second["probe"]["attempt_dir"]
