"""A-D checks for FAMILY_B_PUBLIC_OBS_V2: CPU only, no simulator, no provider."""
from __future__ import annotations

import ast
import copy
import json
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest

from cp_disr.analysis import family_b_obs_v2 as m
from cp_disr.facts import Truth
from cp_disr.platforms.libero import family_b_obs_v2 as obs
from cp_disr.platforms.libero.family_b_adapters import FamilyBVerifier
from cp_disr.platforms.libero.family_b_env import PALETTE, FamilyBEnv
from cp_disr.platforms.libero.perception import THRESHOLDS, backproject_mask

ROOT = Path(__file__).resolve().parents[1]
PROFILE = obs.set_profile(ROOT / m.PROFILE_PATH)
CAL = {c: {"pos": PROFILE["camera_manifest"][c]["pos"], "mat": PROFILE["camera_manifest"][c]["mat"],
           "fovy": PROFILE["camera_manifest"][c]["fovy"], "width": 128, "height": 128} for c in obs.CAMERAS}
TABLE_Z = 0.825


def render(view, cubes, hidden=None):
    calib, size = CAL[view], 128
    f = 0.5 * size / np.tan(np.deg2rad(calib["fovy"]) / 2)
    R, t = np.array(calib["mat"]), np.array(calib["pos"])
    depth = np.full((size, size), 3.0, np.float32)
    rgb = np.full((size, size, 3), 0.5, np.float32)
    h, g = 0.02, np.linspace(-0.02, 0.02, 41)
    for name, c in cubes.items():
        a, b = np.meshgrid(g, g)
        faces = [np.stack([a, b, np.full_like(a, h)], -1).reshape(-1, 3)]
        for s in (-h, h):
            faces.append(np.stack([a, np.full_like(a, s), b], -1).reshape(-1, 3))
            faces.append(np.stack([np.full_like(a, s), a, b], -1).reshape(-1, 3))
        pts = np.concatenate(faces) + np.asarray(c)
        pc = (pts - t) @ R
        z = -pc[:, 2]
        u = np.round((size - 1) / 2 + pc[:, 0] * f / z).astype(int)
        v = np.round((size - 1) / 2 - pc[:, 1] * f / z).astype(int)
        for ui, vi, zi in zip(u, v, z):
            if 0 <= ui < size and 0 <= vi < size and zi < depth[vi, ui]:
                depth[vi, ui] = zi
                rgb[vi, ui] = PALETTE[name][:3]
    return rgb, depth


def fake_env():
    model = types.SimpleNamespace(camera_name2id=lambda n: obs.CAMERAS.index(n),
                                  cam_fovy=np.array([CAL[c]["fovy"] for c in obs.CAMERAS]))
    data = types.SimpleNamespace(cam_xpos=np.array([CAL[c]["pos"] for c in obs.CAMERAS]),
                                 cam_xmat=np.array([np.array(CAL[c]["mat"]).reshape(-1) for c in obs.CAMERAS]))
    return types.SimpleNamespace(sim=types.SimpleNamespace(model=model, data=data))


def perception(cls=obs.FamilyBPerceptionV2):
    p = object.__new__(cls)
    p.env, p.reference, p.profile_hash = fake_env(), {}, PROFILE["profile_sha256"]
    p.calibration_hash = "cal"
    p.checkpoint_hash = "ckpt"
    return p


def observation(cubes, **extra):
    views = {}
    for cam in obs.CAMERAS:
        rgb, depth = render(cam, cubes)
        views[cam] = {"rgb": rgb, "depth": depth}
    return {"views": views, **extra}


CUBES = {"carrier": [0.03, -0.08, TABLE_Z + 0.021], "obj_b": [-0.17, -0.026, TABLE_Z + 0.021],
         "obj_c": [0.13, -0.12, TABLE_Z + 0.021]}


# ---------------------------------------------------------------- A: v1 protection
def test_A1_inventory_detects_changed_removed_added(tmp_path):
    (tmp_path / "e").mkdir()
    (tmp_path / "e/a.bin").write_bytes(b"abc")
    (tmp_path / "e/b.bin").write_bytes(b"def")
    before = m.inventory([tmp_path / "e"])
    (tmp_path / "e/a.bin").write_bytes(b"abd")
    (tmp_path / "e/b.bin").unlink()
    (tmp_path / "e/c.bin").write_bytes(b"x")
    diff = m.diff_inventory(before, m.inventory([tmp_path / "e"]))
    assert diff["changed"] and diff["removed"] and diff["added"]
    assert m.diff_inventory(before, before) == {"changed": [], "removed": [], "added": []}


def test_A2_old_branch_and_base_pinned():
    assert m.BASE == "fffb2b073ab28fa4dc55c82fb2ae58c5aa38dc73"
    try:
        head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "codex/cp-disr-s4-family-b-staging-v1"], text=True).strip()
    except subprocess.CalledProcessError:
        pytest.skip("old branch not present")
    assert head == m.BASE


def test_A3_branch_ids_bind_everything_and_never_reuse_v1():
    logical = {"layout": "layout_0", "context": "B_PENDING", "repeat": 0, "candidate": "pad_u"}
    base = m.branch_identity(logical, "p", "c", "s", "a")
    for variant in (m.branch_identity(logical, "p2", "c", "s", "a"), m.branch_identity(logical, "p", "c2", "s", "a"),
                    m.branch_identity(logical, "p", "c", "s2", "a"), m.branch_identity(logical, "p", "c", "s", "a2"),
                    m.branch_identity({**logical, "candidate": "pad_v"}, "p", "c", "s", "a")):
        assert variant != base
    old = json.loads((ROOT / m.OLD_EVIDENCE / "physical/registration.json").read_text())["branches"]
    assert base not in {b["branch_id"] for b in old}


# ------------------------------------------------------------ B: observation-v2 config
def test_B1_profile_frozen_and_regenerates_identically():
    assert m.check_profile(ROOT)["profile_sha256"] == PROFILE["profile_sha256"]
    for name in ("camera_manifest", "fusion_contract", "pair_equivalence_contract"):
        file = json.loads((ROOT / m.CFG_DIR / f"{name if name != 'camera_manifest' else 'camera_manifest'}.json").read_text())
        assert file["profile_sha256"] == PROFILE["profile_sha256"]
    assert tuple(PROFILE["cameras"]) == obs.CAMERAS and PROFILE["pair_contract"] == m.PAIR_CONTRACT
    assert PROFILE["fusion_contract"]["camera_motion"] is False
    assert PROFILE["fusion_contract"]["candidate_dependent_views"] is False
    assert all(PROFILE["camera_manifest"][c]["fixed_in_world"] and PROFILE["camera_manifest"][c]["mode"] == 0 for c in obs.CAMERAS)


def _registration():
    branches = []
    for layout, context, repeat in m.TECHNICAL:
        for pad in m.PAD_PAIRS:
            branches.append({"layout": layout, "context": context, "repeat": repeat, "candidate": pad,
                             "observation_profile_sha256": "P", "cameras": list(obs.CAMERAS),
                             "profile_version": obs.PROFILE_VERSION, "restore_seed": 7 + (layout == "layout_1")})
    return {"branches": branches}


def test_B2_registration_uniform_across_candidates_layouts_methods():
    assert m.check_registration_uniform(_registration()) == []


def test_B3_queues_keep_a_pair_on_one_gpu_sequentially():
    reg = _registration()
    for i, b in enumerate(reg["branches"]):
        b["branch_id"] = f"b{i}"
    two = m.queues(reg["branches"], [3, 4])
    assert len(two) == 2 and all([b["candidate"] for b in q] == ["pad_u", "pad_v"] for q in two.values())
    assert all(len({(b["layout"], b["context"]) for b in q}) == 1 for q in two.values())
    one = m.queues(reg["branches"], [3])
    assert len(one[3]) == 4


def test_B4_missing_blob_is_unknown_never_carried_over():
    p = perception()
    full = p.analyze(observation(CUBES))[0]
    gone = p.analyze(observation({k: v for k, v in CUBES.items() if k != "obj_b"}))[0]
    p.reference = {v: {n: full[v][n]["depth_support"] for n in PALETTE} for v in obs.CAMERAS}
    assert obs.fuse(full, p.reference)["obj_b"]["blob"] is not None
    fused = obs.fuse(gone, p.reference)["obj_b"]
    assert fused["blob"] is None and fused["selected_view"] is None


def _verifier():
    env = types.SimpleNamespace(
        public_layout=lambda: {"table_top_z": TABLE_Z, "receiver": [0.18, 0.12, 0.83], "pad_u": [-0.2, 0.12, 0.83],
                               "pad_v": [0.2, -0.02, 0.83]},
        task_manifest=types.SimpleNamespace(task_id="T_P_FB"), task_contracts=())
    return FamilyBVerifier(env)


def _measurement(blobs):
    base = {n: {"xyz": c, "pixels": 40} for n, c in blobs.items()}
    return types.SimpleNamespace(measurements={"blobs": base, "eef_pos": [0.0, 0.0, 1.0], "gripper_qpos": [0.04, -0.04]})


def assert_no_carry_over(verifier):
    seen = {n: [c[0], c[1], TABLE_Z + 0.025] for n, c in CUBES.items()}
    first = {r.fact_id: r.value for r in verifier.verify(_measurement(seen))}
    assert first["p:OnTable:obj_b"] == Truth.TRUE
    again = {r.fact_id: r.value for r in verifier.verify(_measurement({k: v for k, v in seen.items() if k != "obj_b"}))}
    for fid in [f for f in again if f.endswith(":obj_b") or ":obj_b:" in f]:
        assert again[fid] == Truth.UNKNOWN, fid


def test_B5_real_verifier_does_not_fill_from_previous_frame():
    assert_no_carry_over(_verifier())


def test_B6_nominal_effect_cannot_fill_dynamic_facts():
    seen = {n: [c[0], c[1], TABLE_Z + 0.025] for n, c in CUBES.items()}
    nominal = types.SimpleNamespace(success=True, controller_exit="NORMAL_TERMINATION", effects="OnTable:obj_b=TRUE")
    a = {r.fact_id: r.value for r in _verifier().verify(_measurement(seen), None)}
    b = {r.fact_id: r.value for r in _verifier().verify(_measurement(seen), nominal)}
    assert a == b
    gone = {k: v for k, v in seen.items() if k != "obj_b"}
    c = {r.fact_id: r.value for r in _verifier().verify(_measurement(gone), nominal)}
    assert c["p:OnTable:obj_b"] == Truth.UNKNOWN


def assert_hidden_invariant(analyze):
    clean = analyze(observation(CUBES))
    poisoned = analyze(observation(CUBES, hidden_truth={"obj_b": [9, 9, 9]}, qpos=[1.0] * 30, qvel=[2.0] * 30))
    a, b = clean[0], poisoned[0]
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def test_B7_hidden_qa_cannot_enter_formal_perception():
    assert_hidden_invariant(perception().analyze)
    assert m._scan_forbidden() == []
    tree = ast.parse((ROOT / "src/cp_disr/platforms/libero/family_b_obs_v2.py").read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & {"hidden_truth", "candidate", "candidate_id", "context", "branch", "layout_id", "qpos", "qvel"}


def test_B8_fusion_rule_support_ratio_tie_and_none():
    pv = {v: {n: {"depth_support": 0, "blob_present": False, "xyz": None, "reason": "x"} for n in PALETTE} for v in obs.CAMERAS}

    def put(view, name, support, xyz):
        pv[view][name] = {"depth_support": support, "blob_present": True, "xyz": xyz, "std": [0, 0, 0], "reason": None}

    put("agentview", "obj_b", 40, [1, 0, 0])
    put("sideview", "obj_b", 40, [2, 0, 0])
    ref = {"agentview": {"obj_b": 40}, "sideview": {"obj_b": 40}}
    assert obs.fuse(pv, ref)["obj_b"]["selected_view"] == "agentview"                 # exact tie -> earlier camera
    put("agentview", "obj_b", 12, [1, 0, 0])                                          # occluded: ratio .3 < 1
    assert obs.fuse(pv, ref)["obj_b"]["selected_view"] == "sideview"
    put("agentview", "obj_b", 4, [1, 0, 0])                                           # below min_pixels
    put("sideview", "obj_b", 4, [2, 0, 0])
    assert obs.fuse(pv, ref)["obj_b"]["blob"] is None


def test_B9_estimators_agentview_is_v1_sideview_unbiased():
    p = perception()
    per_view, _, _ = p.analyze(observation(CUBES))
    rgb, depth = render("agentview", CUBES)
    mask = obs._palette_mask(rgb, "obj_b")
    v1 = backproject_mask(mask, depth, CAL["agentview"])
    assert np.allclose(per_view["agentview"]["obj_b"]["xyz"], v1["xyz"], atol=1e-9)
    side = per_view["sideview"]["obj_b"]
    assert side["blob_present"] and side["depth_support"] >= THRESHOLDS["min_pixels"]
    assert np.allclose(side["xyz"][:2], CUBES["obj_b"][:2], atol=0.012)


def test_B10_occluded_in_one_view_recovered_from_the_other():
    p = perception()
    hidden_in_agent = {k: v for k, v in CUBES.items()}
    views = observation(CUBES)["views"]
    rgb, depth = views["agentview"]["rgb"].copy(), views["agentview"]["depth"].copy()
    mask = obs._palette_mask(rgb, "obj_b")
    rgb[mask] = 0.5                      # a hand covers obj_b in agentview only
    depth[mask] = 0.9
    views["agentview"] = {"rgb": rgb, "depth": depth}
    per_view, _, _ = p.analyze({"views": views})
    ref = {v: {n: 40 for n in PALETTE} for v in obs.CAMERAS}
    fused = obs.fuse(per_view, ref)
    assert fused["obj_b"]["selected_view"] == "sideview"
    assert np.allclose(fused["obj_b"]["blob"]["xyz"][:2], hidden_in_agent["obj_b"][:2], atol=0.012)


def _write_boundary(root, bid, rgb_edit=0, qpos_edit=0.0, cam_edit=None):
    d = Path(root) / "captures" / bid / "boundary"
    d.mkdir(parents=True)
    views, selfrep, fused = {}, {}, {}
    for cam in obs.CAMERAS:
        rgb, depth = render(cam, CUBES)
        rgb8 = (rgb * 255).astype(np.uint8)
        if rgb_edit and cam == (cam_edit or "agentview"):
            rgb8[10, 10, 0] = np.uint8(min(255, int(rgb8[10, 10, 0]) + rgb_edit))
        from PIL import Image
        Image.fromarray(rgb8).save(d / f"{cam}_rgb.png")
        np.save(d / f"{cam}_depth_metric.npy", depth)
        views[cam] = {"rgb_sha256": obs.array_sha(rgb8), "rgb_shape": list(rgb8.shape), "rgb_dtype": "uint8",
                      "depth_sha256": obs.array_sha(depth), "depth_shape": list(depth.shape), "depth_dtype": "float32"}
        selfrep[cam] = {"rgb_equal": True, "depth_equal": True}
    for n in CUBES:
        fused[n] = {"selected_view": "agentview", "xyz": CUBES[n]}
    (d / "views_boundary.json").write_text(json.dumps({
        "profile_sha256": "P", "cameras": list(obs.CAMERAS), "views": views, "self_repeat": selfrep,
        "perception": {"fused": fused}}))
    (d / "public.json").write_text(json.dumps({"fact_values": {"a": "TRUE"}, "candidate_ids": ["x", "y"], "candidate_mask": [True, True]}))
    (d / "qa_state.json").write_text(json.dumps({"qpos": [0.0, 1.0 + qpos_edit], "qvel": [0.0, 0.0]}))


def _pair_out(tmp_path, **edit):
    reg = _registration()
    for i, b in enumerate(reg["branches"]):
        b["branch_id"] = f"b{i}"
        b["observation_profile_sha256"] = "P"
    (tmp_path / "physical").mkdir(parents=True)
    (tmp_path / "physical/registration.json").write_text(json.dumps(reg))
    for i in range(4):
        _write_boundary(tmp_path, f"b{i}", **(edit if i == 1 else {}))
    return tmp_path


def test_B11_pair_equivalence_exact_within_contract_and_out_of_contract(tmp_path):
    assert m.paired_restore(_pair_out(tmp_path / "exact"))["status"] == "PASS"
    near = m.paired_restore(_pair_out(tmp_path / "near", rgb_edit=2))
    assert near["status"] == "PASS" and near["pairs"]["layout_0/B_PENDING/0"]["pixel_equivalence"] == "NON_BITWISE_WITHIN_CONTRACT"
    assert near["pairs"]["layout_0/B_PENDING/0"]["non_bitwise_cause"].startswith("CROSS_PROCESS")
    far = m.paired_restore(_pair_out(tmp_path / "far", rgb_edit=40))
    assert far["status"] == "FAIL"
    assert m.paired_restore(_pair_out(tmp_path / "qpos", qpos_edit=1e-3))["status"] == "FAIL"


# ------------------------------------------------------------------ C: runtime binding
def _manifest():
    return {"runtime": {"active_task_id": "T_P_FB", "repository_path": str(ROOT),
                        "observation_profile_path": m.PROFILE_PATH, "contract_path": "configs/runtime/tp_fb_contract_registry.yaml",
                        "layouts_path": "configs/final_master/family_b_layouts.json", "task_deadlines": {"T_P_FB": 60.0}}}


def test_C1_bundle_binds_family_b_components_without_constructing_env():
    from cp_disr.platforms.libero.family_b_adapters import FamilyBEvaluator
    from cp_disr.platforms.libero.family_b_runtime_v2 import create_family_b_obs_v2_runtime
    bundle = create_family_b_obs_v2_runtime(_manifest())
    assert bundle.env_factory is obs.make_family_b_obs_v2_env and bundle.perception_cls is obs.FamilyBPerceptionV2
    assert bundle.verifier_factory is FamilyBVerifier and bundle.evaluator_factory is FamilyBEvaluator
    assert bundle.environment.__class__.__name__ == "_NullEnv"
    assert issubclass(obs.FamilyBObsV2Env, FamilyBEnv)


def test_C2_two_pads_are_distinct_real_destinations():
    layouts = json.loads((ROOT / "configs/final_master/family_b_layouts.json").read_text())["layouts"]
    for lay in layouts:
        assert np.linalg.norm(np.array(lay["pad_u_xy"]) - np.array(lay["pad_v_xy"])) > 0.15
    stub = types.SimpleNamespace(pad_centers={"pad_u": np.array([-0.2, 0.12, 0.829]), "pad_v": np.array([0.2, -0.02, 0.829])},
                                 receiver_slots={})
    du = FamilyBEnv.resolve_skill_destination(stub, "PLACE_BUFFER", "carrier", "pad_u")
    dv = FamilyBEnv.resolve_skill_destination(stub, "PLACE_BUFFER", "carrier", "pad_v")
    assert not np.allclose(du, dv)


def test_C3_three_dynamic_objects_in_every_view_and_fusion():
    p = perception()
    per_view, _, _ = p.analyze(observation(CUBES))
    fused = obs.fuse(per_view, {})
    for view in obs.CAMERAS:
        assert {"carrier", "obj_b", "obj_c"} <= set(per_view[view])
    assert {"carrier", "obj_b", "obj_c"} <= set(fused)
    assert all(fused[o]["blob"] is not None for o in ("carrier", "obj_b", "obj_c"))


def test_C4_no_provider_rl_or_optimizer_in_validation_modules():
    for rel in ("src/cp_disr/analysis/family_b_obs_v2.py", "scripts/family_b_obs_v2.py",
                "src/cp_disr/platforms/libero/family_b_obs_v2.py", "src/cp_disr/platforms/libero/family_b_runtime_v2.py"):
        tree = ast.parse((ROOT / rel).read_text())
        imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | \
                   {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not any("provider" in i or "optim" in i or "torch" in i or "neural" in i for i in imported), (rel, imported)


# ------------------------------------------------------------- D: reverse regressions
def test_D1_carry_over_regression_is_detected():
    class CarryOver(FamilyBVerifier):
        def verify(self, measurement, execution=None):
            recs = list(super().verify(measurement, execution))
            if "obj_b" not in measurement.measurements["blobs"]:
                prev = getattr(self, "_last", {})
                recs = [r.__class__(**{**r.__dict__, "value": prev.get(r.fact_id, r.value)}) if
                        (":obj_b" in r.fact_id and prev.get(r.fact_id) == Truth.TRUE) else r for r in recs]
            self._last = {r.fact_id: r.value for r in recs}
            return tuple(recs)

    with pytest.raises(AssertionError):
        assert_no_carry_over(CarryOver(_verifier().env))


def test_D2_candidate_dependent_camera_is_detected():
    reg = _registration()
    for b in reg["branches"]:
        if b["candidate"] == "pad_v":
            b["cameras"] = list(reversed(b["cameras"]))
    assert m.check_registration_uniform(reg)
    reg = _registration()
    for b in reg["branches"]:
        if b["candidate"] == "pad_v":
            b["cameras"] = ["agentview"]
    assert m.check_registration_uniform(reg)


def test_D3_hidden_pose_in_formal_perception_is_detected():
    class Leaky(obs.FamilyBPerceptionV2):
        def analyze(self, observation):
            per_view, calibs, depths = super().analyze(observation)
            hidden = observation.get("hidden_truth")
            if hidden:
                per_view["agentview"]["obj_b"]["xyz"] = list(hidden["obj_b"])
            return per_view, calibs, depths

    with pytest.raises(AssertionError):
        assert_hidden_invariant(perception(Leaky).analyze)


def test_D4_different_observation_config_for_u_and_v_is_detected():
    reg = _registration()
    for b in reg["branches"]:
        if b["candidate"] == "pad_v":
            b["observation_profile_sha256"] = "OTHER"
    assert any("different observation configs" in p or "differs" in p for p in m.check_registration_uniform(reg))
    reg = _registration()
    reg["branches"][1]["restore_seed"] = 999
    assert m.check_registration_uniform(reg)


# ------------------------------------------------- recorder / evidence-chain smoke tests
def _fake_full_env(cubes):
    env = fake_env()
    env.sim.data.time = 1.25
    env.public_observation = lambda: {**observation(cubes), "eef_pos": np.array([0.0, 0.0, 1.0]),
                                      "gripper_qpos": np.array([0.04, -0.04])}
    env.public_layout = lambda: {"receiver": np.array([0.18, 0.12, 0.83]), "pad_u": np.array([-0.2, 0.12, 0.829]),
                                 "pad_v": np.array([0.2, -0.02, 0.829]), "table_top_z": TABLE_Z, "container_rim_z": 0.9}
    return env


def test_C5_infer_writes_full_public_evidence_chain_without_recorder_errors(tmp_path):
    from cp_disr.platforms.libero.family_b_runtime_v2 import FamilyBRecorderV2
    p = perception()
    p.env = _fake_full_env(CUBES)
    p.checkpoint_hash = "ckpt"
    rec = FamilyBRecorderV2(tmp_path, "bid", {})
    rec.env = p.env
    rec.adir = tmp_path / "bid" / "action_03"
    rec.adir.mkdir(parents=True)
    p.recorder = rec
    result = p.infer(None)
    assert rec.errors == []
    frames = json.loads((rec.adir / "views_perception.json").read_text())
    f = frames[0]
    assert f["profile_sha256"] == PROFILE["profile_sha256"] and f["hidden_truth_used"] is False and f["reference_initial"] is True
    assert set(f["cameras"]) == set(obs.CAMERAS) and set(f["fusion"]) >= {"carrier", "obj_b", "obj_c"}
    for cam in obs.CAMERAS:
        for name in (f"vframe_001_{cam}_rgb.png", f"vframe_001_{cam}_depth_metric.npy"):
            assert (rec.adir / name).is_file()
        assert {"calibration", "rgb_sha256", "depth_sha256", "objects"} <= set(f["cameras"][cam])
    assert (rec.adir / "perception.json").is_file()
    for o in ("carrier", "obj_b", "obj_c"):
        assert result.measurements["blobs"][o] is not None
    assert result.measurements["blobs"]["receiver"]["source"] == "static_layout"


def test_C6_planner_semantics_and_remaining_support_chain(tmp_path):
    cap = tmp_path / "cap"
    acts = [{"action_id": f"a{i}"} for i in range(6)]
    for i in range(6):
        (cap / f"action_{i:02d}").mkdir(parents=True)

    def ev(i, success):
        return {"terminated": success, "task_success": success, "reason": "TASK_SUCCESS" if success else "CONTINUE"}

    evs = {i: ev(i, i == 5) for i in range(6)}
    for i in (3, 4):
        (cap / f"action_{i:02d}/planner.json").write_text(json.dumps({"status": "PLAN_FOUND", "plan": [acts[i + 1]["action_id"]]}))
    assert m.planner_semantics(cap, acts, evs) == []
    (cap / "action_05/planner.json").write_text(json.dumps({"status": "PLAN_FOUND", "plan": []}))
    assert any("after TASK_SUCCESS" in x for x in m.planner_semantics(cap, acts, evs))
    (cap / "action_05/planner.json").unlink()
    (cap / "action_04/planner.json").unlink()
    assert any("without planner record" in x for x in m.planner_semantics(cap, acts, evs))
    # remaining target chain: pixels -> blob -> fact -> PICK mask
    d = cap / "action_03"
    inputs = {v: {"support": 30, "reference": 36, "ratio": 0.8, "valid": True} for v in obs.CAMERAS}
    d.joinpath("views_perception.json").write_text(json.dumps([{"frame_id": "vframe_002", "fusion": {
        "obj_b": {"selected_view": "sideview", "inputs": inputs, "blob": {"xyz": [0, 0, 0]}}}}]))
    d.joinpath("snapshot.json").write_text(json.dumps({"candidate_ids": ["a:PICK:obj_b:v1"], "candidate_mask": [True],
                                                       "fact_values": {"p:OnTable:obj_b": "TRUE", "p:Held:obj_b": "FALSE"}}))
    assert m.remaining_support(cap, "B_PENDING", 8)["ok"] is True
    d.joinpath("snapshot.json").write_text(json.dumps({"candidate_ids": ["a:PICK:obj_b:v1"], "candidate_mask": [False],
                                                       "fact_values": {"p:OnTable:obj_b": "UNKNOWN", "p:Held:obj_b": "UNKNOWN"}}))
    assert m.remaining_support(cap, "B_PENDING", 8)["ok"] is False


def test_C7_timeline_rows_follow_pixels_blob_fact_mask_planner(tmp_path):
    from cp_disr.platforms.libero.family_b_runtime_v2 import FamilyBRecorderV2
    out = tmp_path
    bid = "bid1"
    reg = {"branches": [{"branch_id": bid, "layout": "layout_0", "context": "B_PENDING", "repeat": 0, "candidate": "pad_v",
                         "v1_logical_branch_id_reference_only": "old"}]}
    (out / "physical/branch_results").mkdir(parents=True)
    (out / "physical/registration.json").write_text(json.dumps(reg))
    p = perception()
    p.env, p.checkpoint_hash = _fake_full_env(CUBES), "c"
    rec = FamilyBRecorderV2(out, bid, {})
    rec.env = p.env
    rec.adir = out / "captures" / bid / "action_00"
    rec.adir.mkdir(parents=True)
    p.recorder = rec
    p.infer(None)
    (rec.adir / "facts.json").write_text(json.dumps([{"fact_id": "p:OnTable:obj_b", "value": "TRUE"}]))
    (rec.adir / "snapshot.json").write_text(json.dumps({"candidate_ids": [f"a:PICK:{o}:v1" for o in ("carrier", "obj_b", "obj_c")],
                                                       "candidate_mask": [True, True, True], "fact_values": {}}))
    (rec.adir / "evaluator.json").write_text(json.dumps({"reason": "CONTINUE", "task_success": False}))
    (out / "physical/branch_results" / f"{bid}.json").write_text(json.dumps({"actions": [{"action_id": "a:PICK:carrier:v1"}]}))
    rows = m.timeline(out)
    assert rows == 3 * len(obs.CAMERAS)
    import csv
    data = list(csv.DictReader((out / "observation_support_timeline.csv").open()))
    row = next(r for r in data if r["object_id"] == "obj_b" and r["view"] == "sideview")
    assert row["fused_present"] == "True" and row["final_OnTable"] == "TRUE" and row["pick_mask_for_object"] == "True"
    assert row["selected_view"] == "agentview" and row["planner_status"] == "NOT_CALLED"
