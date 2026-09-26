import json

import pytest

from cp_disr import phase_a_v13_r1 as r1


pytestmark = pytest.mark.pure


def test_r1_allowlist_and_budget():
    assert r1.TASK == "T_B"
    assert r1.METHODS == ("B1-K", "B2")
    assert r1.AUTHORIZED == {
        "B1-K": "v13_R1_T_B_B1K_s0_E16",
        "B2": "v13_R1_T_B_B2_s0_E16",
    }
    assert r1.N_CAP == 16384
    assert r1.MAX_UPDATES == 16
    assert r1.ROLLOUT_N == 1024
    assert r1.EVAL_POINTS == (0, 4096, 8192, 16384)


def test_configure_routes_only_r1(monkeypatch):
    monkeypatch.setattr(r1.v12, "_install_collector_profiler", lambda: None)
    r1.configure()
    assert r1.v11.TASKS == ("T_B",)
    assert r1.v11.METHODS == ("B1-K", "B2")
    assert r1.v11.N_CAP == 16384
    assert r1.v11.DEV_EPISODES == 10
    assert r1.v11.ENABLED_SPLITS["T_B"] == r1.DEV10_REL
    assert set(r1.v11.PLANNED.values()) == set(r1.AUTHORIZED.values())


def test_source_hashes_bind_runner_and_base(tmp_path, monkeypatch):
    runner = tmp_path / "src/cp_disr/phase_a_v13_r1.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("runner", encoding="utf-8")
    monkeypatch.setattr(r1.subprocess, "check_output", lambda *args, **kwargs: "source-commit\n")
    hashes = r1._source_hashes(tmp_path)
    assert hashes["r1_runner"] == r1.sha(runner)
    assert hashes["git_commit"] == "source-commit"
    assert hashes["base_commit"] == "66614acb443dcacee2b16c5472d4ecbd697bf561"


def test_fixed_mechanism_scope_is_written(tmp_path):
    checkpoint = tmp_path / "job/generations/g000001/model.pt"
    checkpoint.parent.mkdir(parents=True)
    r1._fixed_mechanism_diagnostic(tmp_path, "B2", checkpoint, "n_000000", None)
    row = json.loads((tmp_path / "job/generations/fixed_mechanism_scope.jsonl").read_text())
    assert row["effective_prior"] == 0
    assert row["test_id_used"] is False
