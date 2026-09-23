from pathlib import Path
import json
import pytest
import yaml
from cp_disr.stage1a_v11 import resolve_profile, verify_gamma_half_life
from cp_disr.rl import set_suite_half_life, clear_suite_half_life, gamma


@pytest.mark.pure
def test_stage1a_v11_profile_reads_calibrated_H_and_dref():
    root = Path(__file__).resolve().parents[1]
    prof = resolve_profile(root)
    assert prof["H"] == pytest.approx(23.09999999999752)
    assert prof["d_ref"] == pytest.approx(4.5500000000015195)
    assert prof["Ncap"] == 16384
    assert prof["Tcap"] == pytest.approx(16384 * prof["d_ref"])
    assert prof["Tcap"] != pytest.approx(16384 * 3.55)
    assert prof["actor_episode_discount_weight"] is False
    assert prof["runtime_path"].endswith("runtime_manifest_v211.yaml")
    assert prof["task_deadline_seconds"] == pytest.approx(43.0)
    man = yaml.safe_load((root / "experiments/manifests/runtime_manifest_v211.yaml").read_text(encoding="utf-8"))
    ref = json.loads((root / "runs/stage_0a/reference_execution_manifest.json").read_text(encoding="utf-8"))
    assert ref["H"] == prof["H"]
    assert man["runtime"]["suite_H_seconds"] == prof["H"]
    set_suite_half_life(prof["H"])
    try:
        verify_gamma_half_life(prof["H"])
        assert gamma(prof["H"]) == pytest.approx(0.5, abs=1e-12, rel=1e-12)
    finally:
        clear_suite_half_life()
