"""Production RuntimeBundle binding for T_P_FB with FAMILY_B_PUBLIC_OBS_V2."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cp_disr.common import BindingError

from .family_b_adapters import FamilyBEvaluator, FamilyBVerifier
from .family_b_obs_v2 import (CAMERAS, PROFILE_VERSION, FamilyBObsV2Env,
                              FamilyBPerceptionV2, array_sha, calibration,
                              live_camera, make_family_b_obs_v2_env, set_profile)
from .family_b_runtime import (ACTION_IDS, TASK_ID, FamilyBBundle, FamilyBRecorder,
                               build_task_template, load_task_contracts)
from .family_b_env import FamilyBCaseSpec
from .perception import _metric_depth
from .tp_so_mvp_runtime import _NullEnv
from .tp_sr_instrumentation import _safe, jsonable


class FamilyBRecorderV2(FamilyBRecorder):
    """Adds per-camera raw frames, metric depth, calibration and fusion trace."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._vframe, self._vframes = 0, []

    def _save_views(self, env, d, stem):
        obs = env.public_observation()
        for cam in CAMERAS:
            v = obs["views"][cam]
            self._png(d / f"{stem}_{cam}_rgb.png", v["rgb"])
            np.save(d / f"{stem}_{cam}_depth.npy", np.asarray(_metric_depth(env, v["depth"]), dtype=np.float32))

    @_safe
    def write_initial(self, env, bundle):
        super().write_initial(env, bundle)
        self._save_views(env, self.root, "initial")
        from .family_b_obs_v2 import fuse
        per_view, calibs, _ = bundle.perception.analyze(env.public_observation())
        fused = fuse(per_view, bundle.perception.reference)
        self._wj("initial_view_analysis.json", {
            "per_view": per_view, "reference": bundle.perception.reference,
            "fused": {n: {"selected_view": f["selected_view"], "inputs": f["inputs"],
                          "xyz": None if f["blob"] is None else f["blob"]["xyz"]} for n, f in fused.items()}},
            self.root)
        self._wj("observation_profile_live.json", {
            "profile_version": PROFILE_VERSION,
            "cameras": {c: {"live": live_camera(env, c), "calibration": calibration(env, c)} for c in CAMERAS},
        }, self.root)

    @_safe
    def begin_action(self, env, candidate_id):
        super().begin_action(env, candidate_id)
        self._vframes = []
        self._save_views(env, self.adir, "before")

    @_safe
    def end_action(self, env, result):
        super().end_action(env, result)
        self._save_views(env, self.adir, "after")

    @_safe
    def perception_views(self, p):
        d = self._dir()
        self._vframe += 1
        fid = f"vframe_{self._vframe:03d}"
        cams = {}
        for cam in CAMERAS:
            rgb = np.asarray(p["views"][cam]["rgb"])
            depth = np.asarray(p["depths"][cam], dtype=np.float32)
            self._png(d / f"{fid}_{cam}_rgb.png", rgb)
            np.save(d / f"{fid}_{cam}_depth_metric.npy", depth)
            cams[cam] = {
                "rgb_sha256": array_sha(rgb), "rgb_shape": list(rgb.shape), "rgb_dtype": str(rgb.dtype),
                "depth_sha256": array_sha(depth), "depth_shape": list(depth.shape), "depth_dtype": str(depth.dtype),
                "calibration": p["calibs"][cam], "objects": p["per_view"][cam],
            }
        meas = p["meas"]
        rec = {
            "frame_id": fid, "sim_time": self._sim_time(self.env) if self.env is not None else None,
            "profile_version": PROFILE_VERSION, "profile_sha256": p["profile_sha256"],
            "calibration_sha256": p["calibration_sha256"], "reference_initial": bool(p["reference_initial"]),
            "cameras": cams, "fusion": p["fused"],
            "final_blobs": {k: (None if v is None else {"xyz": v.get("xyz"), "pixels": v.get("pixels"),
                                                         "source": v.get("source"), "view": v.get("view")})
                            for k, v in meas["blobs"].items()},
            "unknown_reasons": meas["unknown_reasons"],
            "hidden_truth_used": False, "qpos_qvel_used": False,
        }
        self._vframes.append(rec)
        self._wj("views_perception.json", self._vframes, d)
        # keep the v1-shaped agentview record as well
        self.perception_frame(p["views"]["agentview"]["rgb"], p["depths"]["agentview"],
                              p["calibs"]["agentview"], p["layout"], meas, PROFILE_VERSION,
                              p["calibration_sha256"])


class FamilyBObsV2Bundle(FamilyBBundle):
    def configure(self, out_root, branch):
        self.out_root, self.branch = Path(out_root), dict(branch)
        self.recorder = FamilyBRecorderV2(out_root, branch["branch_id"], branch)
        return self.recorder

    def start_case(self, case_id, restore_seed=None):
        snap = super().start_case(case_id, restore_seed)
        if not isinstance(self.environment, FamilyBObsV2Env):
            raise BindingError("Family B observation-v2 environment binding lost")
        if not isinstance(self.perception, FamilyBPerceptionV2):
            raise BindingError("Family B observation-v2 perception binding lost")
        return snap


def create_family_b_obs_v2_runtime(manifest):
    runtime = manifest["runtime"]
    if runtime.get("active_task_id") != TASK_ID:
        raise BindingError("Family B runtime active_task_id mismatch")
    root = Path(runtime["repository_path"])
    profile = set_profile(root / runtime["observation_profile_path"])
    if profile["profile_version"] != PROFILE_VERSION:
        raise BindingError("observation profile version mismatch")
    template = build_task_template(root / runtime["contract_path"])
    deadline = float(runtime["task_deadlines"][TASK_ID])
    layouts = json.loads((root / runtime["layouts_path"]).read_text())
    cases = {}
    for layout in layouts["layouts"]:
        case_id = layout["layout_id"]
        cases[case_id] = FamilyBCaseSpec(
            case_id=case_id, split="dev", seed=0,
            target_xy=tuple(layout["carrier_xy"]), second_xy=tuple(layout["obj_b_xy"]),
            obj_c_xy=tuple(layout["obj_c_xy"]), container_xy=tuple(layout["receiver_xy"]),
            buffer_xy=tuple(layout["pad_u_xy"]), pad_v_xy=tuple(layout["pad_v_xy"]),
            lid_closed=False, task_id=TASK_ID, deadline=deadline,
        )
    null = _NullEnv()
    return FamilyBObsV2Bundle(
        environment=null, executor=None, observations=None, perception=None,
        verifier=FamilyBVerifier(null), evaluator=FamilyBEvaluator(null, deadline, TASK_ID),
        safety=None, clock=None, snapshot_builder=None, task_id=TASK_ID,
        template=template, cases=cases, caches={},
        env_factory=make_family_b_obs_v2_env, perception_cls=FamilyBPerceptionV2,
        verifier_factory=FamilyBVerifier, evaluator_factory=FamilyBEvaluator,
    )


def create(manifest):
    return create_family_b_obs_v2_runtime(manifest)
