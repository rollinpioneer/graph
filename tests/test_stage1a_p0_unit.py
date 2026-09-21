"""Stage 1A-P0 unit gates that do not require robosuite."""
from pathlib import Path
import pytest
import yaml

@pytest.mark.pure
def test_move_rejected_in_runtime_registry():
    root = Path(__file__).resolve().parents[1]
    doc = yaml.safe_load((root / "configs/contracts/d0_runtime_skills.yaml").read_text())
    names = [c["name"] for c in doc["contracts"]]
    assert "MOVE" not in names
    assert set(names) == {"OPEN", "PICK", "PLACE", "PLACE_BUFFER"}
    assert doc.get("move_decision") == "REJECTED_FOR_D0"

@pytest.mark.pure
def test_runtime_time_unit_is_seconds():
    root = Path(__file__).resolve().parents[1]
    doc = yaml.safe_load((root / "experiments/manifests/runtime_manifest.yaml").read_text(encoding="utf-8"))
    assert doc["runtime"]["actual_interaction_time_unit"] == "seconds"
    assert doc["runtime"].get("actual_clock_source") == "mujoco_sim_data_time"

