"""I01-I09: passive recorders never change control, never leak hidden QA, keep the global perception untouched."""
from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from cp_disr.platforms.libero import perception as P
from cp_disr.platforms.libero import tp_sr_instrumentation as I
from cp_disr.platforms.libero.d0_env import COLORS
from cp_disr.platforms.libero.skill_executor import SkillExecutor

CAM = {"pos": [0.5, 0.0, 1.35], "fovy": 45.0, "width": 128, "height": 128,
       "mat": [[0.0, -0.706147844353306, 0.7080644193257978], [1.0, 0.0, 0.0], [0.0, 0.7080644193257978, 0.706147844353306]]}


class FakeData:
    def __init__(self):
        self.time = 0.0
        self.cam_xpos = np.array([CAM["pos"]])
        self.cam_xmat = np.array([np.array(CAM["mat"]).reshape(-1)])
        self.qpos = np.zeros(4)
        self.qvel = np.zeros(4)


class FakeModel:
    cam_fovy = np.array([CAM["fovy"]])

    def camera_name2id(self, name):
        return 0


class FakeEnv:
    table_top_z = 0.825

    def __init__(self, rgb=None):
        self.sim = SimpleNamespace(data=FakeData(), model=FakeModel())
        self.eef = np.array([0.0, 0.0, 1.0])
        self.calls = []
        self.recorder = None
        self.last_perception = {}
        self.second_role = "interferer"
        self._rgb = rgb if rgb is not None else np.zeros((128, 128, 3), dtype=np.uint8)

    def public_observation(self):
        return {"rgb": self._rgb, "depth": np.full((128, 128), 1.05, dtype=np.float32), "proprio": np.zeros(3), "eef_pos": self.eef.copy(),
                "eef_quat": np.array([0, 0, 0, 1.0]), "gripper_qpos": np.array([0.04, -0.04]), "sim_time": self.sim.data.time}

    def public_layout(self):
        return {"container": np.array([0.03, 0.15, 0.829]), "buffer": np.array([-0.17, 0.12, 0.829]), "table_top_z": 0.825, "container_rim_z": 0.873}

    def hidden_truth(self):
        return {"target": np.zeros(3), "interferer": np.ones(3), "table_top_z": 0.825, "eef_pos": self.eef.copy(), "gripper_qpos": np.array([0.04, -0.04]),
                "container": np.array([0.03, 0.15, 0.829])}

    def step_osc(self, dpos, dgrip, n=1):
        self.calls.append((tuple(np.round(dpos, 9)), float(dgrip)))
        self.eef = self.eef + 0.05 * np.clip(np.asarray(dpos, dtype=float), -1, 1)
        self.sim.data.time += 0.05
        return []


def _exec(cls, env):
    return cls(env, safety=None, clock=None)


def _trace():
    from cp_disr.platforms.libero.skill_executor import SkillTrace
    return SkillTrace(skill="PICK", arguments=("target",))


def test_I01_I02_instrumentation_does_not_alter_or_add_steps(tmp_path):
    target = np.array([0.1, 0.05, 0.9])
    base_env, ins_env = FakeEnv(), FakeEnv()
    _exec(SkillExecutor, base_env)._move_to(_trace(), target, -1.0, 9.0, 0.0, tol=0.01, max_steps=60)
    ex = _exec(I.InstrumentedSkillExecutor, ins_env)
    ex.recorder = I.RunRecorder(tmp_path, "b0")
    ex.recorder.skill = "PICK"
    ex._move_to(_trace(), target, -1.0, 9.0, 0.0, tol=0.01, max_steps=60)
    assert base_env.calls == ins_env.calls and len(base_env.calls) >= 3          # I01 identical commands, I02 identical number of steps
    ev = [json.loads(x) for x in (tmp_path / "captures/b0/pre_action/controller_trace.jsonl").read_text().splitlines()]
    assert {e["event"] for e in ev if e["kind"] == "event"} == {"phase_begin", "phase_end"}


def test_I03_full_factrecord_values_recorded(tmp_path):
    from cp_disr.platforms.libero.verifier import FactVerifier
    env = FakeEnv()
    meas = SimpleNamespace(measurements={"blobs": {"target": {"xyz": [0.0, 0.0, 0.85]}, "interferer": {"xyz": [0.1, 0.1, 0.85]}, "container": {"xyz": [0.03, 0.15, 0.83]},
                                                   "buffer": {"xyz": [-0.17, 0.12, 0.83]}, "lid": {"xyz": [0.29, 0.15, 0.85]}},
                           "eef_pos": [0, 0, 1.0], "gripper_qpos": [0.04, -0.04]})
    v = FactVerifier(env)
    v.__class__ = I.RecordingVerifier
    v.recorder = I.RunRecorder(tmp_path, "b1")
    recs = v.verify(meas)
    rows = json.loads((tmp_path / "captures/b1/pre_action/facts.json").read_text())
    assert len(rows) == len(recs) == 10
    assert all(set(r) >= {"fact_id", "value", "reason", "evidence_ids", "capture_time", "available_time", "last_confirmed_value", "last_confirmed_time"} for r in rows)


def test_I04_rgb_depth_written_before_and_after(tmp_path):
    env = FakeEnv()
    rec = I.RunRecorder(tmp_path, "b2")
    rec.begin_action(env, "a:PICK:target:v1")
    rec.end_action(env, {"controller_exit": "NORMAL_TERMINATION", "steps": 3, "states": [], "sim_duration": 1.0, "execution_id": "x"})
    d = tmp_path / "captures/b2/action_00"
    for n in ("before_rgb.png", "before_depth.npy", "after_rgb.png", "after_depth.npy", "qa_state.jsonl", "controller_trace.jsonl"):
        assert (d / n).is_file() and (d / n).stat().st_size > 0, n
    assert not rec.errors, rec.errors


def test_I05_hidden_qa_is_flagged_and_not_reachable_from_policy_paths(tmp_path):
    env = FakeEnv()
    rec = I.RunRecorder(tmp_path, "b3")
    rec.begin_action(env, "a:PICK:target:v1")
    q = json.loads((tmp_path / "captures/b3/action_00/qa_state.jsonl").read_text().splitlines()[0])
    assert q["qa_only"] is True and not any(q[k] for k in ("used_by_policy", "used_by_planner", "used_by_verifier", "used_by_provider"))
    for obj in (I.RecordingVerifier, I.RecordingPlanner, I.TPSRNearestPalettePerceptionAdapter, I.RecordingSnapshotBuilderProxy):
        assert "hidden_truth" not in inspect.getsource(obj)


def test_I06_contact_geom_names_preserved_with_real_mujoco():
    mujoco = pytest.importorskip("mujoco")
    xml = """<mujoco><option gravity="0 0 0"/><worldbody>
      <body name="gripper0_leftfinger" pos="0 0 0"><freejoint/><geom name="gripper0_finger1_pad_collision" type="box" size="0.02 0.02 0.02"/></body>
      <body name="interferer_main" pos="0.0399 0 0"><freejoint/><geom name="interferer_g0" type="box" size="0.02 0.02 0.02"/></body>
    </worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d)
    env = FakeEnv()
    env.sim = SimpleNamespace(model=SimpleNamespace(_model=m), data=SimpleNamespace(_data=d, time=0.0))
    snap = I.RecordingEnvMixin._contact_snapshot(env)
    assert snap["count"] >= 1 and snap["pairs"]
    p = snap["pairs"][0]
    assert {p["geom1"], p["geom2"]} == {"gripper0_finger1_pad_collision", "interferer_g0"}
    assert {p["cat1"], p["cat2"]} == {"finger", "interferer"}
    assert isinstance(p["normal_force"], float)


def test_I07_planner_input_facts_saved(tmp_path):
    from cp_disr.facts import Truth
    inner = SimpleNamespace(plan=lambda facts, template, remaining: SimpleNamespace(status="NO_PLAN", plan=[], expanded_nodes=1, cpu_seconds=0.0))
    rec = I.RunRecorder(tmp_path, "b4")
    pl = I.RecordingPlanner(inner, rec)
    facts = SimpleNamespace(values={"p:Held:target": Truth.TRUE, "p:GripperEmpty": Truth.FALSE})
    template = SimpleNamespace(contracts=[SimpleNamespace(id="a:PLACE:target:container:v1")], goals=("p:Inside:target:container",))
    assert pl.plan(facts, template, 50.0).status == "NO_PLAN"
    doc = json.loads((tmp_path / "captures/b4/pre_action/planner.json").read_text())
    assert doc["input_facts"] == {"p:Held:target": "TRUE", "p:GripperEmpty": "FALSE"} and doc["candidate_ids"] == ["a:PLACE:target:container:v1"] and doc["status"] == "NO_PLAN"


def _cross_talk_frame():
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    rgb[60:70, 60:70] = (np.asarray(COLORS["interferer"][:3]) * 0.75 * 255).astype(np.uint8)   # shaded cyan, as in the V1 finding
    return rgb


def test_I08_online_adapter_leaves_global_mask_untouched(tmp_path):
    before = P._mask
    env = FakeEnv(_cross_talk_frame())
    ad = I.TPSRNearestPalettePerceptionAdapter(env)
    ad.recorder = I.RunRecorder(tmp_path, "b5")
    res = ad.infer(env.public_observation())
    assert P._mask is before and before.__module__ == P.__name__
    assert res.measurements["blobs"]["container"] is not None                    # static-layout fallback, not cyan cross-talk
    assert (tmp_path / "captures/b5/pre_action/perception.json").is_file()


def test_I09_offline_comparison_same_frame_and_local_equals_v1_and_original():
    from cp_disr.analysis.s4_tp_sr_pilot_v2_diag import compare_variants_on_frame
    from cp_disr.platforms.libero import tp_sr_runtime as V1
    rng = np.random.default_rng(0)
    for _ in range(3):
        img = rng.random((32, 32, 3)).astype(np.float32)
        for c in COLORS.values():
            assert np.array_equal(I.nearest_palette_mask(img, c, 0.32), V1._nearest_palette_mask(img, c, 0.32))
    rgb = _cross_talk_frame()
    env = FakeEnv(rgb)
    depth = np.full((128, 128), 1.05, dtype=np.float32)
    calib = P.camera_calibration(env)
    layout = env.public_layout()
    orig_infer = P.PerceptionAdapter(env).infer(env.public_observation()).measurements["blobs"]
    b, _t, _r = I.blobs_from_frame(rgb, depth, calib, layout, P._mask)
    assert {k: (None if v is None else v["pixels"]) for k, v in b.items()} == {k: (None if v is None else v["pixels"]) for k, v in orig_infer.items()}
    cmp_ = compare_variants_on_frame(rgb, depth, calib, {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in layout.items()}, [0, 0, 1.0], [0.04, -0.04])
    assert set(cmp_) == {"ORIGINAL_TOLERANCE_MASK", "NEAREST_PALETTE_MASK", "fact_differences"}
    assert cmp_["ORIGINAL_TOLERANCE_MASK"]["blobs"]["container"]["pixels"] > 0 and cmp_["NEAREST_PALETTE_MASK"]["blobs"]["container"]["pixels"] == 0
