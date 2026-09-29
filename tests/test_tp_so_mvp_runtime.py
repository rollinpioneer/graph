"""R01-R07: the T_P_SO_MVP runtime binds both objects symmetrically, one reset per branch, no provider/optimizer."""
from __future__ import annotations

from pathlib import Path

import pytest

from cp_disr.analysis import s4_family_a_soft_ordering_mvp as m

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/final_master/s4_family_a_soft_ordering_mvp.yaml"
CFG = m.load_config(CONFIG)


def _runtime():
    from cp_disr.platforms.libero import tp_so_mvp_runtime as r
    return r


def _source(rel):
    """Locate a repo-relative source file; tolerates the offline harness, which mirrors the package without `src/`."""
    bare = rel[4:] if rel.startswith("src/") else rel
    roots = [ROOT, *(p for p in Path(m.__file__).resolve().parents)]
    for r in roots:
        for cand in (r / rel, r / bare, r / "src" / bare):
            if cand.is_file():
                return cand
    return None


@pytest.fixture(scope="module")
def contracts():
    contracts, template, objects = m._build_offline_template(ROOT)
    return contracts, template, objects


def test_R01_both_initial_candidates_mask_true(contracts):
    _contracts, template, _objects = contracts
    facts = m._initial_fact_store()
    true_now = sorted(c.id for c in template.contracts if m._precond_true(c, facts))
    assert true_now == sorted(m.INITIAL_CANDIDATES)
    assert len(true_now) == 2
    # the two initial candidates are the two PICKs, one per goal object
    assert true_now == ["a:PICK:second_object:v1", "a:PICK:target:v1"]


def test_R02_both_goals_present(contracts):
    _contracts, template, _objects = contracts
    goals = {g.fact_id for g in template.goals}
    assert goals == {"p:Inside:target:container", "p:Inside:second_object:container"}
    assert set(_runtime().GOAL_FACTS) == goals


def test_R03_both_objects_use_identical_geometry_and_dynamics():
    from cp_disr.platforms.libero import d0_env
    # one shared half-extent for both cubes
    assert d0_env.OBJECT_HALF.tolist() == [0.020, 0.020, 0.020]
    src = Path(d0_env.__file__).read_text(encoding="utf-8")
    # both cubes are built with the same size/friction/density; only the colour differs
    target = [ln for ln in src.splitlines() if 'name="target"' in ln and "BoxObject" in ln]
    second = [ln for ln in src.splitlines() if 'self.second_role' in ln and "BoxObject" in ln]
    assert len(target) == 1 and len(second) == 1
    for frag in ("size=OBJECT_HALF", "friction=(1.2, 0.005, 0.0001)", "density=200"):
        assert frag in target[0], frag
        assert frag in second[0], frag
    assert "rgba=COLORS[\"target\"]" in target[0]
    assert "rgba=second_rgba" in second[0]
    # and the two colours really are distinct
    assert not (d0_env.COLORS["target"] == d0_env.COLORS["second_object"]).all()


def test_R04_one_reset_per_branch():
    r = _runtime()

    class _FakeEnv:
        reset_calls = 1
        internal_resets = 0

        def close(self):
            return None

    bundle = r.TPSOMVPBundle(environment=_FakeEnv(), executor=None, observations=None, perception=None, verifier=None,
                              evaluator=None, safety=None, clock=None, snapshot_builder=None, task_id=r.TASK_ID)
    counts = bundle.env_counts()
    assert counts["constructions"] == 1
    assert counts["bootstrap_constructions"] == 0
    assert counts["reset_calls"] == 1
    assert "start_case" in counts["restore_chain"]
    # the runtime is the only environment factory; no bootstrap environment is constructed
    src = Path(r.__file__).read_text(encoding="utf-8")
    assert src.count("make_env(") == 0  # the factory is referenced, never called directly
    assert "env_factory=InstrumentedD0Env" in src
    assert "bootstrap_constructions\": 0" in src


def test_R05_local_perception_leaves_global_mask_unchanged():
    assert m.perception_identity()["global_monkeypatch"] is False
    r = _runtime()
    assert r.RUNTIME_IDENTITY == "tp-so-mvp-runtime-v1"
    # the local adapter must not import or assign the process-global perception._mask
    from cp_disr.platforms.libero import tp_sr_instrumentation as inst
    src = Path(inst.__file__).read_text(encoding="utf-8")
    assert "perception._mask =" not in src
    assert "nearest_palette_mask" in src
    assert inst.LOCAL_PERCEPTION_VARIANT == m.perception_identity()["variant"]
    # the adapter only ever computes a *local* mask via the shared backprojection helper
    assert "_mask(" not in src.replace("nearest_palette_mask(", "").replace("backproject_mask(", "").replace("_metric_depth(", "")


def test_R06_no_provider_access():
    for rel in ("src/cp_disr/platforms/libero/tp_so_mvp_runtime.py",
                "src/cp_disr/analysis/s4_family_a_soft_ordering_mvp.py"):
        p = _source(rel)
        assert p is not None, rel
        text = p.read_text(encoding="utf-8")
        assert "import" not in "\n".join(ln for ln in text.splitlines() if "provider" in ln and ln.strip().startswith("import"))
        assert "s4_tp_sr_provider" not in text.split("provider_module_loaded")[0]  # only the zero-check mentions it
    assert CFG["authorization"]["provider_authorized"] is False
    assert int(CFG["budgets"]["provider_first_calls"]) == 0
    assert int(CFG["budgets"]["provider_retries"]) == 0


def test_R07_no_optimizer():
    for rel in ("src/cp_disr/platforms/libero/tp_so_mvp_runtime.py",
                "src/cp_disr/analysis/s4_family_a_soft_ordering_mvp.py"):
        p = _source(rel)
        assert p is not None, rel
        text = p.read_text(encoding="utf-8").lower()
        for token in ("torch.optim", "optimizer.step", "loss.backward", "scheduler.step"):
            assert token not in text, f"{rel} references {token}"
    assert CFG["authorization"]["optimizer_authorized"] is False
    assert int(CFG["budgets"]["optimizer_steps"]) == 0
    assert int(CFG["budgets"]["rl_transitions"]) == 0
    assert int(CFG["budgets"]["training_attempts"]) == 0