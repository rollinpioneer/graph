"""Offline tests for the post-OPEN capture card: state dump/restore round trip, tolerances, accounting. No simulator."""
import json
import random

import numpy as np

from cp_disr.analysis import tp_ef_post_open_capture as c


class Buf:
    def __init__(self):
        self.buf = np.arange(6, dtype=float).reshape(3, 2)
        self.cur = 1.5


class Gripper:
    def __init__(self):
        self.current_action = np.array([-1.0, 1.0])
        self.speed = 0.2


class Ctrl:
    def __init__(self):
        self.goal_pos = np.array([0.1, 0.2, 0.3])
        self.goal_ori = np.eye(3)
        self.new_update = True
        self.sim = object()  # must be skipped
        self.torques = np.zeros(7)


class Env:
    def __init__(self):
        self.timestep = 7
        self.cur_time = 0.35
        self.done = False
        self._obs_cache = {"a": np.ones(3), "b": np.zeros(2)}
        self.sim = object()
        self.rng = np.random.RandomState(3)
        self.name = "env"
        self.callback = lambda: 1


def test_dump_roundtrip_and_hash_stable():
    e = Env()
    d = c.dump_state(e)
    assert any(c.path_str(p).endswith("timestep") for p in d)
    assert not any("sim" in c.path_str(p) for p in d) and not any("callback" in c.path_str(p) for p in d)
    meta, arrays = c.serialize_state(d)
    back = c.deserialize_state(json.loads(json.dumps(meta)), arrays)
    assert c.state_hash(d) == c.state_hash(back) and c.diff_states(d, back) == []


def test_load_state_restores_values_in_place():
    e = Env()
    d = c.dump_state(e)
    e.timestep, e.cur_time, e.done = 99, 9.9, True
    e._obs_cache["a"][:] = -5
    e.rng.randint(0, 100, 10)
    missing = c.load_state(e, d)
    assert missing == [] and e.timestep == 7 and e.cur_time == 0.35 and e.done is False
    assert np.array_equal(e._obs_cache["a"], np.ones(3)) and c.diff_states(d, c.dump_state(e)) == []


def test_diff_detects_changes_and_named_paths():
    g = Gripper()
    a = c.dump_state(g)
    g.current_action = np.array([0.0, 0.0])
    diffs = c.diff_states(a, c.dump_state(g))
    assert diffs and diffs[0]["path"].split("/")[-1].lstrip(".") in c.NAMED_GATING


def test_ctrl_skips_sim_and_keeps_goal_arrays():
    k = Ctrl()
    d = c.dump_state(k)
    names = {c.path_str(p) for p in d}
    assert ".goal_pos" in names and ".goal_ori" in names and ".new_update" in names and ".sim" not in names


def test_rng_roundtrip():
    np.random.seed(5)
    random.seed(5)
    snap = c.rng_snapshot()
    h = c.rng_hash(snap)
    np.random.rand(10)
    random.random()
    assert c.rng_hash(c.rng_snapshot()) != h
    c.apply_rng(snap)
    assert c.rng_hash(c.rng_snapshot()) == h


def test_obs_and_sim_tolerances():
    a = {"rgb": np.zeros((4, 4, 3), dtype=np.uint8), "depth": np.ones((4, 4), dtype=np.float32), "proprio": np.zeros(5)}
    b = {k: v.copy() for k, v in a.items()}
    assert c.compare_obs(a, b)["pass"]
    b["rgb"][0, 0, 0] = 1
    assert not c.compare_obs(a, b)["pass"]
    b = {k: v.copy() for k, v in a.items()}
    b["depth"][0, 0] += 1e-5
    assert not c.compare_obs(a, b)["pass"]
    s = {"qpos": np.zeros(3), "time": np.zeros(1)}
    assert c.compare_sim(s, {k: v.copy() for k, v in s.items()})["pass"]
    t = {k: v.copy() for k, v in s.items()}
    t["qpos"][1] = 1e-9
    assert not c.compare_sim(s, t)["pass"]


def test_budget_caps_fit_design():
    assert c.CAPS["environment_constructions"] == 4 and c.CAPS["start_case_calls"] == 4 and c.CASES == ("T_B_dev_03", "T_B_dev_05")
    assert c.CAPS["provider_requests"] == 0 and c.CAPS["continuation_skills"] == 0


def test_identity_excludes_episode_counter_fields():
    src = open(c.__file__).read()
    assert "episode_id" not in src.split("state_identity_sha256\"] = digest(")[1].split("\n")[0]
