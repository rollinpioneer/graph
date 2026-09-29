"""Passive observation recorders for the T_P_SR V2 pilot (CP-DISR-S4-TP-SR-PILOT-V2-1).

Nothing here changes an action, a timeout, a fact, a mask, a reward, a planner input or a random seed, and no extra
control step is added. Every recorder operation is guarded: a recorder fault is logged (and later blocks the Wave D
technical gate) but never alters or aborts the physical run. Hidden MuJoCo state is written QA-only and is never handed
to the policy, planner, verifier or provider.
"""
from __future__ import annotations

import functools
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np

from .d0_env import COLORS
from .perception import PerceptionAdapter, PerceptionResult, THRESHOLDS, VERSION as PERCEPTION_VERSION, _metric_depth, backproject_mask, camera_calibration, sha
from .skill_executor import SkillExecutor
from .verifier import FactVerifier

LOCAL_PERCEPTION_VARIANT = "TP_SR_V1_NEAREST_PALETTE_COMPAT_LOCAL"
INSTRUMENTATION_VERSION = "tp-sr-instrumentation-v1"
PICK_PHASES = ("HOVER", "DESCEND", "PRESS", "CLOSE", "LIFT_SHOW", "LIFT_HOVER_FALLBACK")
PLACE_PHASES = ("HOVER", "DROP_DESCEND", "RELEASE", "RETREAT")
QA_FLAGS = {"qa_only": True, "used_by_policy": False, "used_by_planner": False, "used_by_verifier": False, "used_by_provider": False}


def jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if hasattr(v, "value") and not isinstance(v, (str, int, float)):
        return getattr(v, "value")
    return v


def hsha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=float)).tobytes()).hexdigest()


def _safe(fn):
    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        try:
            return fn(self, *a, **k)
        except Exception as exc:  # noqa: BLE001 - a recorder fault must never alter the physical run
            try:
                self.errors.append({"where": fn.__name__, "error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()[-800:]})
                with (self.root / "recorder_errors.jsonl").open("a") as h:
                    h.write(json.dumps(self.errors[-1]) + "\n")
            except Exception:
                pass
            return None
    return wrapper


# ------------------------------------------------------------------ local nearest-palette perception (no global patch)
def nearest_palette_mask(rgb, color, tol):
    """Numerically identical to tp_sr_runtime._nearest_palette_mask (V1) but only ever called from the local adapter."""
    img = np.asarray(rgb).astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    mine = np.linalg.norm(img - np.asarray(color[:3])[None, None, :], axis=2)
    ok = mine < tol
    for other in COLORS.values():
        if other is color or np.array_equal(np.asarray(other[:3]), np.asarray(color[:3])):
            continue
        ok &= mine <= np.linalg.norm(img - np.asarray(other[:3])[None, None, :], axis=2)
    return ok


def blobs_from_frame(rgb, depth, calib, layout, mask_fn):
    """Pure re-statement of PerceptionAdapter.infer's blob stage with an injectable mask function (used online and offline)."""
    blobs, table_xyz, reasons = {}, {}, {}
    for name, color in COLORS.items():
        m = mask_fn(rgb, color, THRESHOLDS["color_tol"])
        rec = backproject_mask(m, depth, calib)
        if rec is None:
            blobs[name] = None
            reasons[name] = "insufficient_color_depth_support"
        else:
            blobs[name] = {"xyz": rec["xyz"].tolist(), "pixels": rec["pixels"], "std": rec["std"].tolist()}
            table_xyz[name] = rec["xyz"]
    for key in ("container", "buffer"):
        if key not in table_xyz:
            table_xyz[key] = np.array(layout[key], dtype=float)
            blobs[key] = {"xyz": table_xyz[key].tolist(), "pixels": 0, "std": [0, 0, 0], "source": "static_layout"}
            reasons.pop(key, None)
    return blobs, table_xyz, reasons


class TPSRNearestPalettePerceptionAdapter(PerceptionAdapter):
    """V1-compatible nearest-palette assignment, applied only inside this adapter; cp_disr...perception._mask is untouched."""

    variant = LOCAL_PERCEPTION_VARIANT
    recorder = None

    def infer(self, observation) -> PerceptionResult:
        if not isinstance(observation, dict) or "rgb" not in observation:
            observation = self.env.public_observation()
        rgb = np.asarray(observation["rgb"])
        depth = _metric_depth(self.env, observation["depth"])
        calib = camera_calibration(self.env)
        self.calibration_hash = sha(calib)
        layout = self.env.public_layout()
        blobs, table_xyz, reasons = blobs_from_frame(rgb, depth, calib, layout, nearest_palette_mask)
        self.env.last_perception = table_xyz
        meas = {
            "blobs": blobs, "unknown_reasons": reasons,
            "eef_pos": np.asarray(observation["eef_pos"], dtype=float).tolist(),
            "gripper_qpos": np.asarray(observation["gripper_qpos"], dtype=float).tolist(),
            "calib": {k: calib[k] for k in ("fovy", "width", "height")},
            "depth_stats": {"min": float(np.nanmin(depth)), "max": float(np.nanmax(depth)), "mean": float(np.nanmean(depth))},
        }
        rec = self.recorder
        if rec is not None:
            rec.perception_frame(rgb, depth, calib, layout, meas, self.variant, self.calibration_hash)
        return PerceptionResult(meas, self.checkpoint_hash, self.calibration_hash, ("agentview_rgb", "agentview_depth", "camera_calibration"), PERCEPTION_VERSION)


# ------------------------------------------------------------------ recorder
def _category(geom_name, body_name):
    s = f"{geom_name or ''}|{body_name or ''}".lower()
    if "finger" in s or "pad" in s and "buffer" not in s:
        return "finger"
    if "hand" in s or "gripper" in s:
        return "palm"
    if "robot" in s or "link" in s:
        return "arm"
    for key in ("target", "interferer", "lid", "container", "buffer"):
        if key in s:
            return key
    if "table" in s or "floor" in s:
        return "table"
    return "other"


class RunRecorder:
    def __init__(self, out_root, branch_id, meta=None):
        self.root = Path(out_root) / "captures" / branch_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.branch_id = branch_id
        self.meta = dict(meta or {})
        self.errors = []
        self.action = -1
        self.adir = None
        self.skill = None
        self.candidate = None
        self.phase = "INIT"
        self._pc = 0
        self._frames = []
        self._frame_no = 0
        self._step_no = 0
        self._phase_open = None
        self.env = None

    # -- paths / io
    def _dir(self):
        d = self.adir or (self.root / "pre_action")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _jl(self, name, rec, d=None):
        with ((d or self._dir()) / name).open("a") as h:
            h.write(json.dumps(jsonable(rec), sort_keys=True) + "\n")

    def _wj(self, name, rec, d=None):
        (d or self._dir()).joinpath(name).write_text(json.dumps(jsonable(rec), indent=1, sort_keys=True))

    @staticmethod
    def _png(path, rgb):
        from PIL import Image
        a = np.asarray(rgb)
        if a.dtype != np.uint8:
            a = (np.clip(a, 0, 1) * 255).astype(np.uint8) if a.max() <= 1.5 else a.astype(np.uint8)
        Image.fromarray(a).save(path)

    def _eef(self, env):
        try:
            return np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id], dtype=float).tolist()
        except Exception:
            return None

    def _grip(self, env):
        try:
            return np.array(env.sim.data.qpos[env.robots[0]._ref_gripper_joint_pos_indexes], dtype=float).tolist()
        except Exception:
            return None

    def _public_xyz(self, env):
        p = getattr(env, "last_perception", None) or {}
        f = lambda k: None if p.get(k) is None else np.asarray(p[k], dtype=float).tolist()
        return f("target"), f("interferer")

    # -- initial capture
    @_safe
    def write_initial(self, env, bundle):
        self.env = env
        d = self.root
        obs = env.public_observation()
        depth = _metric_depth(env, obs["depth"])
        self._png(d / "initial_rgb.png", obs["rgb"])
        np.save(d / "initial_depth.npy", np.asarray(depth, dtype=np.float32))
        snap = bundle.current_snapshot
        facts = {r.fact_id: r for r in snap.facts.records}
        self._wj("initial_public_facts.json", {"facts": {k: {"value": jsonable(v.value), "reason": v.reason} for k, v in facts.items()},
                                               "candidate_ids": list(snap.candidate_ids), "candidate_mask": [bool(x) for x in snap.mask]}, d)
        layout = env.public_layout()
        self._wj("layout.json", {"layout": layout, "second_role": getattr(env, "second_role", None), "case": jsonable(getattr(env, "case", None).__dict__) if getattr(env, "case", None) is not None else None}, d)
        self._wj("initial_object_bindings.json", {"objects": {"target": "target", "interferer": "interferer", "container": "container", "buffer": "buffer"},
                                                 "public_blobs": {k: v for k, v in zip(("target", "interferer"), self._public_xyz(env))}}, d)
        self._wj("camera_config.json", camera_calibration(env), d)
        self._wj("asset_identity.json", {"env_class": type(env).__name__, "case": jsonable(env.case.__dict__), "instrumentation": INSTRUMENTATION_VERSION,
                                         "object_half": jsonable(getattr(env, "second_half", None))}, d)
        self.qa_state(env, "initial", d)

    # -- actions
    @_safe
    def begin_action(self, env, candidate_id):
        self.env = env
        self.action += 1
        self.adir = self.root / f"action_{self.action:02d}"
        self.adir.mkdir(parents=True, exist_ok=True)
        parts = candidate_id.split(":")
        self.candidate, self.skill = candidate_id, (parts[1] if len(parts) > 1 else "")
        self._pc = 0
        self._frames = []
        obs = env.public_observation()
        self._png(self.adir / "before_rgb.png", obs["rgb"])
        np.save(self.adir / "before_depth.npy", np.asarray(_metric_depth(env, obs["depth"]), dtype=np.float32))
        self.qa_state(env, "before")
        self.event("skill_begin", candidate_id=candidate_id)

    @_safe
    def end_action(self, env, result):
        obs = env.public_observation()
        self._png(self.adir / "after_rgb.png", obs["rgb"])
        np.save(self.adir / "after_depth.npy", np.asarray(_metric_depth(env, obs["depth"]), dtype=np.float32))
        self.qa_state(env, "after")
        self.event("skill_end", controller_exit=result.get("controller_exit"), steps=result.get("steps"), states=result.get("states"),
                   sim_duration=result.get("sim_duration"), execution_id=result.get("execution_id"))

    def _sim_time(self, env):
        try:
            return float(env.sim.data.time)
        except Exception:
            return None

    @_safe
    def event(self, name, **fields):
        env = self.env
        t_pub, i_pub = self._public_xyz(env) if env is not None else (None, None)
        self._jl("controller_trace.jsonl", {"kind": "event", "event": name, "branch_id": self.branch_id, "candidate_id": self.candidate,
                                            "skill": self.skill, "phase": self.phase, "sim_time": self._sim_time(env) if env is not None else None,
                                            "eef_pos": self._eef(env) if env is not None else None, "gripper_qpos": self._grip(env) if env is not None else None,
                                            "target_xyz_public": t_pub, "interferer_xyz_public": i_pub, **fields})

    def _phase_name(self, kind):
        names = PICK_PHASES if self.skill == "PICK" else PLACE_PHASES
        i = self._pc
        self._pc += 1
        return names[i] if i < len(names) else f"{kind}_{i}"

    @_safe
    def phase_begin(self, kind, **fields):
        self.phase = self._phase_name(kind)
        self.event("phase_begin", phase_kind=kind, **fields)

    @_safe
    def phase_end(self, **fields):
        self.event("phase_end", **fields)

    @_safe
    def step(self, env, dpos, dgrip, before, after, contacts):
        self._step_no += 1
        rec = {"kind": "step", "step": self._step_no, "active_skill": self.skill, "active_phase": self.phase, "dpos": np.asarray(dpos, dtype=float).tolist(),
               "grip_command": float(dgrip), **before, **after, "contact_pairs": contacts["pairs"], "contact_count": contacts["count"]}
        self._jl("controller_trace.jsonl", rec)
        if contacts["pairs"]:
            self._jl("contacts.jsonl", {"step": self._step_no, "sim_time": after.get("sim_time_after"), "phase": self.phase, "skill": self.skill, "pairs": contacts["pairs"]})

    @_safe
    def qa_state(self, env, tag, d=None):
        h = env.hidden_truth()
        sim = env.sim
        geoms = []
        try:
            import mujoco
            m = sim.model._model
            dd = sim.data._data
            for gid in range(m.ngeom):
                gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)
                bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gid]))
                cat = _category(gname, bname)
                if cat in ("finger", "palm"):
                    geoms.append({"geom": gname, "body": bname, "category": cat, "type": int(m.geom_type[gid]), "size": m.geom_size[gid].tolist(),
                                  "xpos": dd.geom_xpos[gid].tolist(), "xmat": dd.geom_xmat[gid].tolist()})
        except Exception as exc:  # noqa: BLE001
            geoms = [{"error": f"{type(exc).__name__}: {exc}"}]
        qpos, qvel = np.asarray(sim.data.qpos, dtype=float), np.asarray(sim.data.qvel, dtype=float)
        rec = {**QA_FLAGS, "tag": tag, "sim_time": float(sim.data.time), "hidden_truth": h, "gripper_geoms": geoms,
               "qpos_sha256": hsha(qpos), "qvel_sha256": hsha(qvel), "eef_pos": self._eef(env)}
        if tag == "initial":
            rec["qpos"], rec["qvel"] = qpos.tolist(), qvel.tolist()
        self._jl("qa_state.jsonl", rec, d or self._dir())

    # -- perception / facts / evaluator / planner
    @_safe
    def perception_frame(self, rgb, depth, calib, layout, meas, variant, calib_hash):
        d = self._dir()
        self._frame_no += 1
        fid = f"frame_{self._frame_no:03d}"
        self._png(d / f"{fid}_rgb.png", rgb)
        np.save(d / f"{fid}_depth_metric.npy", np.asarray(depth, dtype=np.float32))
        rec = {"frame_id": fid, "sim_time": self._sim_time(self.env) if self.env is not None else None, "camera_identity": {"calibration_sha256": calib_hash, "calib": calib},
               "perception_variant": variant, "blob_ids": sorted(k for k, v in meas["blobs"].items() if v is not None),
               "pixel_counts": {k: (v or {}).get("pixels") for k, v in meas["blobs"].items()}, "blob_xyz": {k: (v or {}).get("xyz") for k, v in meas["blobs"].items()},
               "unknown_reasons": meas["unknown_reasons"], "depth_stats": meas["depth_stats"], "layout": layout, "eef_pos": meas.get("eef_pos"), "gripper_qpos": meas.get("gripper_qpos"), "rgb_sha256": hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()}
        self._frames.append(rec)
        self._wj("perception.json", self._frames, d)

    @_safe
    def facts(self, records):
        recs = [{"fact_id": r.fact_id, "value": jsonable(r.value), "reason": r.reason, "evidence_ids": list(r.evidence_ids), "capture_time": r.capture_time,
                 "available_time": r.available_time, "last_confirmed_value": jsonable(r.last_confirmed_value), "last_confirmed_time": r.last_confirmed_time} for r in records]
        self._wj("facts.json", recs)

    @_safe
    def evaluator(self, value, result):
        self._wj("evaluator.json", {"input": {"elapsed_seconds": value.elapsed_seconds, "interval_start_seconds": value.interval_start_seconds,
                                              "interval_end_seconds": value.interval_end_seconds, "task_id": value.task_id},
                                    "task_success": bool(result.success), "terminated": bool(result.terminated), "truncated": bool(result.truncated), "reason": result.reason})

    @_safe
    def snapshot(self, snap):
        self._wj("snapshot.json", {"candidate_ids": list(snap.candidate_ids), "candidate_mask": [bool(x) for x in snap.mask],
                                   "fact_values": {k: jsonable(v) for k, v in dict(snap.facts.values).items()}})

    @_safe
    def planner(self, facts, template, remaining, plan):
        self._wj("planner.json", {"input_facts": {k: jsonable(v) for k, v in dict(facts.values).items()}, "candidate_ids": [c.id for c in template.contracts],
                                  "goal": [str(g) for g in getattr(template, "goals", ())], "remaining_deadline": float(remaining), "status": plan.status,
                                  "plan": list(plan.plan), "expanded_nodes": plan.expanded_nodes, "cpu_seconds": plan.cpu_seconds})

    def close(self, env_counts):
        try:
            self._wj("branch_recorder_summary.json", {"actions": self.action + 1, "control_steps": self._step_no, "frames": self._frame_no,
                                                       "recorder_errors": len(self.errors), "env_counts": env_counts}, self.root)
        except Exception:
            pass


# ------------------------------------------------------------------ env mixin (task-specific env subclasses only)
class RecordingEnvMixin:
    recorder = None
    reset_calls = 0
    internal_resets = 0

    def reset(self):
        self.reset_calls = int(getattr(self, "reset_calls", 0)) + 1
        return super().reset()

    def _reset_internal(self):
        self.internal_resets = int(getattr(self, "internal_resets", 0)) + 1
        return super()._reset_internal()

    def _contact_snapshot(self):
        import mujoco
        m, d = self.sim.model._model, self.sim.data._data
        pairs, force = [], np.zeros(6)
        for i in range(int(d.ncon)):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            names = []
            for g in (g1, g2):
                gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
                names.append((gn, bn, _category(gn, bn)))
            cats = {names[0][2], names[1][2]}
            if not (cats & {"finger", "palm", "target", "interferer", "lid"}):
                continue
            try:
                mujoco.mj_contactForce(m, d, i, force)
                nf = float(force[0])
            except Exception:
                nf = "NOT_MEASURED"
            pairs.append({"geom1": names[0][0], "geom2": names[1][0], "body1": names[0][1], "body2": names[1][1], "cat1": names[0][2], "cat2": names[1][2],
                          "distance": float(c.dist), "normal_force": nf})
            if len(pairs) >= 80:
                break
        return {"pairs": pairs, "count": int(d.ncon)}

    def step_osc(self, dpos, dgrip, n=1):
        rec = self.recorder
        if rec is None:
            return super().step_osc(dpos, dgrip, n=n)
        try:
            before = {"sim_time_before": float(self.sim.data.time), "eef_pos_before": rec._eef(self)}
        except Exception:
            before = {}
        out = super().step_osc(dpos, dgrip, n=n)
        try:
            after = {"sim_time_after": float(self.sim.data.time), "eef_pos_after": rec._eef(self), "qpos_sha256": hsha(self.sim.data.qpos), "qvel_sha256": hsha(self.sim.data.qvel)}
            contacts = self._contact_snapshot()
        except Exception as exc:  # noqa: BLE001
            after, contacts = {"recorder_error": f"{type(exc).__name__}: {exc}"}, {"pairs": [], "count": -1}
        rec.step(self, dpos, dgrip, before, after, contacts)
        return out


# ------------------------------------------------------------------ passive wrappers
class InstrumentedSkillExecutor(SkillExecutor):
    recorder = None

    def execute(self, candidate_id, timeout_seconds):
        rec = self.recorder
        if rec is not None:
            rec.begin_action(self.env, candidate_id)
        out = super().execute(candidate_id, timeout_seconds)
        if rec is not None:
            rec.end_action(self.env, out)
        return out

    def _move_to(self, trace, target_xyz, grip, timeout, sim_start, tol=0.018, max_steps=220, tag="APPROACH"):
        rec = self.recorder
        if rec is None:
            return super()._move_to(trace, target_xyz, grip, timeout, sim_start, tol=tol, max_steps=max_steps, tag=tag)
        rec.phase_begin("MOVE", tag=tag, target_xyz=np.asarray(target_xyz, dtype=float).tolist(), grip=float(grip), tol=tol, max_steps=max_steps, steps_before=trace.steps)
        ok = super()._move_to(trace, target_xyz, grip, timeout, sim_start, tol=tol, max_steps=max_steps, tag=tag)
        rec.phase_end(ok=bool(ok), steps_after=trace.steps, exit_reason=trace.exit_reason)
        return ok

    def _hold(self, trace, grip, n, timeout, sim_start):
        rec = self.recorder
        if rec is None:
            return super()._hold(trace, grip, n, timeout, sim_start)
        rec.phase_begin("HOLD", grip=float(grip), n=int(n), steps_before=trace.steps)
        ok = super()._hold(trace, grip, n, timeout, sim_start)
        rec.phase_end(ok=bool(ok), steps_after=trace.steps, exit_reason=trace.exit_reason)
        return ok

    def _pick(self, trace, obj, timeout, sim_start):
        if self.recorder is not None:
            self.recorder.event("pick_begin", obj=obj)
        return super()._pick(trace, obj, timeout, sim_start)

    def _place(self, trace, obj, container, timeout, sim_start):
        if self.recorder is not None:
            self.recorder.event("place_begin", obj=obj, container=container)
        return super()._place(trace, obj, container, timeout, sim_start)

    def _place_buffer(self, trace, obj, buffer, timeout, sim_start):
        if self.recorder is not None:
            self.recorder.event("place_buffer_begin", obj=obj, buffer=buffer)
        return super()._place_buffer(trace, obj, buffer, timeout, sim_start)


class RecordingVerifier(FactVerifier):
    recorder = None

    def verify(self, measurement, execution=None):
        recs = super().verify(measurement, execution)
        if self.recorder is not None:
            self.recorder.facts(recs)
        return recs


class RecordingEvaluatorProxy:
    def __init__(self, inner, recorder):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_rec", recorder)

    def __getattr__(self, n):
        return getattr(self._inner, n)

    def __setattr__(self, n, v):
        setattr(self._inner, n, v)

    def evaluate(self, value):
        res = self._inner.evaluate(value)
        self._rec.evaluator(value, res)
        return res


class RecordingSnapshotBuilderProxy:
    def __init__(self, inner, recorder):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_rec", recorder)

    def __getattr__(self, n):
        return getattr(self._inner, n)

    def build(self, *a, **k):
        snap = self._inner.build(*a, **k)
        self._rec.snapshot(snap)
        return snap


class RecordingPlanner:
    def __init__(self, inner, recorder):
        self._inner, self._rec = inner, recorder

    def plan(self, facts, template, remaining):
        res = self._inner.plan(facts, template, remaining)
        self._rec.planner(facts, template, remaining, res)
        return res
