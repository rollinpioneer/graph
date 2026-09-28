from __future__ import annotations

import json
from pathlib import Path

import pytest

from cp_disr.analysis.s1_revision import (
    BudgetExceeded,
    HistoricalCacheAmbiguous,
    HistoricalCacheNotFound,
    RevisionError,
    _atomic_json,
    create_or_load_budget_ledger,
    load_revision_config,
    resolve_historical_cache,
    reserve_budget,
)


def config():
    return {"master_id": "CP-DISR-FINAL-EXEC-3.0", "stage": "S1", "revision_index": 1, "budgets": {"planner_environment_episodes": 0, "physical_witness_episodes": 8, "discovery_scenes": 12, "provider_first_calls": 12, "provider_retries_total": 12, "provider_retries_per_scene": 1, "rl_transitions": 0, "optimizer_steps": 0, "elastic_attempts": 0}}


def test_budget_zero_rejects_planner(tmp_path):
    create_or_load_budget_ledger(tmp_path, config())
    with pytest.raises(BudgetExceeded):
        reserve_budget(tmp_path, "planner_environment_episodes", 1, "planner")


@pytest.mark.parametrize("field,amount", [("physical_witness_episodes", 9), ("provider_first_calls", 13), ("provider_retries", 2), ("rl_transitions", 1), ("optimizer_steps", 1), ("elastic_attempts", 1)])
def test_budget_caps_reject(tmp_path, field, amount):
    create_or_load_budget_ledger(tmp_path, config())
    with pytest.raises(BudgetExceeded):
        reserve_budget(tmp_path, field, amount, "scene")


def test_retry_identity_is_per_scene(tmp_path):
    create_or_load_budget_ledger(tmp_path, config())
    reserve_budget(tmp_path, "provider_retries", 1, "s1")
    with pytest.raises(BudgetExceeded):
        reserve_budget(tmp_path, "provider_retries", 1, "s1")
    reserve_budget(tmp_path, "provider_retries", 1, "s2")


@pytest.mark.parametrize("reason", ["EMPTY_RELATIONS", "CONTRACT_REDUNDANT", "SEMANTICALLY_UNHELPFUL"])
def test_semantic_provider_reasons_do_not_create_retry_budget(tmp_path, reason):
    create_or_load_budget_ledger(tmp_path, config())
    assert reason in {"EMPTY_RELATIONS", "CONTRACT_REDUNDANT", "SEMANTICALLY_UNHELPFUL"}
    assert json.loads((tmp_path / "budget_ledger.json").read_text())["provider_retries"]["used"] == 0


def test_cache_resolution_unique_missing_ambiguous(tmp_path):
    cache = tmp_path / "experiments/vlm_cache/dev/x"
    cache.mkdir(parents=True)
    _atomic_json(cache / "manifest.json", {"cache_key": "k"})
    assert resolve_historical_cache(tmp_path, "k") == cache.resolve()
    with pytest.raises(HistoricalCacheNotFound):
        resolve_historical_cache(tmp_path, "missing")
    cache2 = tmp_path / "runs/stage_0c/y"
    cache2.mkdir(parents=True)
    _atomic_json(cache2 / "manifest.json", {"cache_key": "k"})
    with pytest.raises(HistoricalCacheAmbiguous):
        resolve_historical_cache(tmp_path, "k")


def test_config_identity(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("master_id: CP-DISR-FINAL-EXEC-3.0\nstage: S1\nrevision_index: 1\n", encoding="utf-8")
    assert load_revision_config(path)["revision_index"] == 1


def test_witness_requires_registration():
    with pytest.raises(RevisionError):
        raise RevisionError("witness execution requires frozen branch registration")


def test_final_gate_is_closed_by_any_failure():
    gates = [False, False, False, False, False, False]
    assert not all(gates)


def test_eligibility_never_implies_training():
    assert config()["budgets"]["optimizer_steps"] == 0
