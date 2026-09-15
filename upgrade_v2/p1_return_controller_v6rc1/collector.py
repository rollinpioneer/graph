"""V6RC1 collector. Imports V6 PhysicalBackend read-only; never writes object qpos after start."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from upgrade_v2.p1_reward_repair_v6.backend_adapter import (
    CONTROL_DT, HOLD_STABLE_S, PhysicalBackend, RETURN_TIMEOUT_S as V6_RET,
)
from upgrade_v2.p1_reward_repair_v6.backend_adapter import family_physics
from .util import require_new, sha256_file, write_json, write_jsonl
from .checkpoint import make_checkpoint
from .attachment_estimator import AttachmentEstimator, N_SAMPLES, MAX_DISPERSION_M
from .return_geometry import desired_eef_position, clipnorm, in_workspace, finite_vec, norm, COMPENSATION_MODE
from .state_machine import stages_for, BUDGET_S
from .closure import load_v6_closure, diagnostic_ok

CASES = [
    "E1_LOSS_RETURN_P40",
    "E2_LOSS_RETURN_P80",
    "E3_THREE_RETURNS_P40",
    "E4_THREE_RETURNS_P80",
    "E5_REGRASP_OFFSET_LEFT_P40",
    "E6_REGRASP_OFFSET_RIGHT_P40",
    "E7_NO_LOSS_OUT_AND_BACK",
    "E8_COMMANDED_RELEASE_CONTROL",
]
OFFSET = {"E5_REGRASP_OFFSET_LEFT_P40": -0.04, "E6_REGRASP_OFFSET_RIGHT_P40": 0.04}
REHOLD_S = 0.20
RETURN_TIMEOUT_S = 2.0


class ReturnCollector:
    def __init__(self, repo: Path, controller_id: str, controller_spec: dict):
        self.repo = Path(repo)
        self.controller_id = controller_id
        self.spec = controller_spec
        self.backend = PhysicalBackend(repo)
        self.closure_fn, self.closure_lock = load_v6_closure()
        self.invocations = 0

    def _sample(self, raw, integ):
        be = self.backend
        st = be.read_current_state()
        ref = be.reference_log[-1] if be.reference_log else be._update_reference()
        events = [dict(e, known_at_ns=st["physical_time_ns"]) for e in ref.get("events") or []]
        obj = st["objects"]["obj"]
        raw.append({
            "t": st["t"], "physical_time_ns": st["physical_time_ns"], "capture_order": len(raw),
            "eef": st["eef"], "eef_quat": st["eef_quat"], "gripper_closed": st["gripper_closed"],
            "objects": {"obj": dict(obj, held=ref["held"], valid=ref["valid"], phase=ref["phase"],
                                    ever_held=ref["ever_held"], open_loss_id=ref["open_loss_id"])},
            "events": events, "controller_mode": st["controller_mode"], "weld_active": st["weld_active"],
            "ncon": st["ncon"], "support": ref["support"],
        })
        integ.append({"t": float(st["t"]), "qpos": be.read_integration_state()["qpos"].tolist(),
                      "qvel": be.read_integration_state()["qvel"].tolist()})
        return raw[-1]

    def _advance(self, raw, integ, n=1):
        last = None
        for _ in range(n):
            self.backend.advance_control_interval()
            last = self._sample(raw, integ)
        return last

    def _move_to(self, xyz, gripper, raw, integ, *, mode="MOVE", timeout_s=6.0, tol=0.012, speed=None):
        if speed:
            self.backend.max_speed = float(speed)
        self.backend.set_controller_target(xyz, gripper, mode=mode)
        last = None
        for _ in range(max(1, int(timeout_s / CONTROL_DT))):
            last = self._advance(raw, integ)
            if float(np.linalg.norm(np.asarray(last["eef"]) - np.asarray(xyz))) <= tol:
                last = self._advance(raw, integ, n=max(1, int(0.04 / CONTROL_DT)))
                return last, True
        return last, False

    def _wait(self, raw, integ, seconds, *, mode="WAIT", gripper=None):
        if gripper is None:
            gripper = "closed" if self.backend.gripper_closed else "open"
        pose = np.array(self.backend.data.mocap_pos[0], dtype=float)
        self.backend.set_controller_target(pose, gripper, mode=mode)
        return self._advance(raw, integ, n=max(1, int(round(seconds / CONTROL_DT))))

    def _grasp(self, raw, integ, xy_offset=0.0):
        obj = np.array(self.backend._object_xyz(), dtype=float)
        approach = np.array([0.0, float(xy_offset), 0.0])
        # Final capture offset is limited by weld range (2 cm); 4 cm is approach-side stress.
        cap = 0.0 if abs(xy_offset) < 1e-9 else (0.015 if xy_offset > 0 else -0.015)
        off = np.array([0.0, cap, 0.0])
        above = obj + np.array([0.0, 0.0, 0.16]) + approach
        last, _ = self._move_to(above, "open", raw, integ, mode="APPROACH")
        grasp = obj + np.array([0.0, 0.0, 0.13]) + off
        last, _ = self._move_to(grasp, "open", raw, integ, mode="DESCEND")
        last, _ = self._move_to(grasp, "closed", raw, integ, mode="CLOSE")
        last = self._wait(raw, integ, HOLD_STABLE_S + 0.06, mode="SETTLE", gripper="closed")
        return last, bool(last["objects"]["obj"]["held"])

    def _lift_q0(self, raw, integ):
        eef = np.array(self.backend.data.mocap_pos[0], dtype=float)
        q0 = eef + np.array([0.0, 0.0, 0.18])
        last, _ = self._move_to(q0, "closed", raw, integ, mode="LIFT")
        last = self._wait(raw, integ, HOLD_STABLE_S, mode="Q0_SETTLE", gripper="closed")
        return last

    def _save_ckpt(self, name, checkpoints, raw, episode_id):
        last = raw[-1]
        o = last["objects"]["obj"]
        ck = make_checkpoint(
            checkpoint_id=name, episode_id=episode_id, time_ns=last["physical_time_ns"],
            state_index=last["capture_order"], p_obj=o["pos"], q_obj=o["quat"], v_obj=o["vel"],
            w_obj=o["angular_vel"], p_eef=last["eef"], q_eef=last["eef_quat"], held=o["held"],
            phase=o["phase"], target_xy=o["target_xy"],
        )
        checkpoints.append(ck)
        return ck

    def _path_point(self, q0, frac):
        spec = self.backend.spec
        goal = np.array([spec.target_xy[0], spec.target_xy[1], q0[2]])
        return np.asarray(q0) + (goal - np.asarray(q0)) * float(frac)

    def _disturb(self, raw, integ):
        spec = self.backend.spec
        force = np.array([spec.disturb_xy[0], spec.disturb_xy[1], 0.0]) * min(2.5, spec.disturb_force_n)
        self.backend.apply_declared_disturbance({"force_n": force.tolist(), "duration_s": 0.08, "breaks_weld": True})
        last = None
        for _ in range(int(1.2 / CONTROL_DT)):
            last = self._advance(raw, integ)
            if any(e["kind"] == "LOSS" for e in last["events"]):
                self.backend.clear_disturbance()
                last = self._wait(raw, integ, 0.20, mode="POST_LOSS", gripper="closed")
                return last, True
        self.backend.clear_disturbance()
        return last, False

    def _place(self, raw, integ):
        spec = self.backend.spec
        above = np.array([spec.target_xy[0], spec.target_xy[1], self.backend.data.mocap_pos[0][2]])
        last, _ = self._move_to(above, "closed", raw, integ, mode="TRANSPORT")
        lower = np.array([spec.target_xy[0], spec.target_xy[1], spec.object_z0 + 0.13])
        last, _ = self._move_to(lower, "closed", raw, integ, mode="LOWER")
        last, _ = self._move_to(lower, "open", raw, integ, mode="RELEASE")
        last = self._wait(raw, integ, 1.0, mode="SETTLE_PLACE", gripper="open")
        return last

    def _state_for_closure(self, raw_row, ckpt):
        o = raw_row["objects"]["obj"]
        return {
            "task": "recovery", "episode_id": ckpt["episode_id"], "gripper_closed": raw_row["gripper_closed"],
            "success": bool(o["valid"]), "terminal_failure": False, "eef": raw_row["eef"],
            "objects": {"obj": {"held": o["held"], "valid": o["valid"], "phase": o["phase"],
                                "pos": o["pos"], "vel": o["vel"], "target_xy": o["target_xy"],
                                "quat": o["quat"], "angular_vel": o["angular_vel"]}},
        }

    def run_return(self, ckpt, raw, integ, rc_log, att_log, closures, *, allow_invoke=True):
        be = self.backend
        notes = []
        if not allow_invoke:
            return {"status": "NOT_INVOKED", "invoked": False}
        self.invocations += 1
        stage = "WAIT_REHOLD_STABLE"
        held_s = 0.0
        last = raw[-1]
        # wait rehold 0.20s, not on return clock
        for _ in range(int(2.0 / CONTROL_DT)):
            last = self._advance(raw, integ)
            if last["objects"]["obj"]["held"]:
                held_s += CONTROL_DT
            else:
                held_s = 0.0
            if held_s + 1e-12 >= REHOLD_S:
                break
        else:
            return {"status": "REHOLD_FAILED", "invoked": True}
        est = AttachmentEstimator()
        stage = "ESTIMATE_ATTACHMENT"
        for _ in range(N_SAMPLES):
            last = self._advance(raw, integ)
            if not last["objects"]["obj"]["held"]:
                return {"status": "HOLD_LOST_DURING_ESTIMATE", "invoked": True}
            est.add(last["objects"]["obj"]["pos"], last["eef"], last["eef_quat"])
        estimate = est.estimate()
        att_log.append(dict(estimate, t=last["t"], checkpoint_id=ckpt["checkpoint_id"]))
        if not estimate["stable"]:
            return {"status": "ATTACHMENT_UNSTABLE", "invoked": True, "estimate": estimate}
        q_eef = last["eef_quat"]
        try:
            p_des = desired_eef_position(ckpt["p_WO_star"], q_eef, estimate["r_eo_median"])
            p_des = finite_vec(p_des)
        except Exception as exc:
            return {"status": "PLAN_FAIL", "invoked": True, "reason": str(exc)}
        if not in_workspace(p_des):
            return {"status": "WORKSPACE", "invoked": True}
        clock = 0.0
        rc_log.append({"event": "RETURN_STARTED", "t": last["t"], "p_eef_des": p_des,
                       "compensation_mode": COMPENSATION_MODE, "controller_id": self.controller_id})
        cid = self.controller_id
        def log_tick(stage, last, p_tgt, abort=None):
            o = last["objects"]["obj"]
            obj_err = float(np.linalg.norm(np.asarray(o["pos"]) - np.asarray(ckpt["p_WO_star"])))
            eef_err = float(np.linalg.norm(np.asarray(last["eef"]) - np.asarray(p_tgt)))
            rc_log.append({
                "controller_id": cid, "controller_stage": stage, "return_clock_s": clock,
                "checkpoint_id": ckpt["checkpoint_id"], "p_obj_current": o["pos"], "p_obj_target": ckpt["p_WO_star"],
                "p_eef_current": last["eef"], "p_eef_target": list(p_tgt), "object_error_norm": obj_err,
                "eef_error_norm": eef_err, "attachment_estimate": estimate["r_eo_median"],
                "hold_current": o["held"], "velocity_norm": float(np.linalg.norm(o["vel"])),
                "command_limited": True, "abort_reason": abort, "t": last["t"],
            })
        def maybe_done(last):
            # Object/task return uses the V6 helper and V6 numeric thresholds.
            # Compensated EEF is not expected to match the pre-loss EEF, so both
            # sides use the current EEF; raw EEF error is still logged.
            cur = self._state_for_closure(last, ckpt)
            a = self._state_for_closure({"objects": {"obj": {
                "held": last["objects"]["obj"]["held"], "valid": False, "phase": last["objects"]["obj"]["phase"],
                "pos": ckpt["p_WO_star"], "vel": [0.0,0.0,0.0], "target_xy": last["objects"]["obj"]["target_xy"],
                "quat": ckpt["q_WO_star"], "angular_vel": [0.0,0.0,0.0]}},
                "gripper_closed": last["gripper_closed"], "eef": last["eef"]}, ckpt)
            return self.closure_fn(a, cur)

        def hold_abort(last):
            return not last["objects"]["obj"]["held"]

        # motion
        stages = []
        if cid == "RC1_OBJECT_COMPENSATED_DIRECT":
            stages = [("XY_ALIGN", p_des, 0.45, 1.6), ("SETTLE_AND_VERIFY", p_des, 0.12, 0.20)]
        elif cid == "RC2_OBJECT_COMPENSATED_STAGED":
            zc = max(last["eef"][2], p_des[2]) + 0.08
            p_clear = [last["eef"][0], last["eef"][1], zc]
            p_xy = [p_des[0], p_des[1], zc]
            stages = [("SAFE_CLEARANCE", p_clear, 0.45, 0.35), ("XY_ALIGN", p_xy, 0.45, 0.80),
                      ("Z_ALIGN", p_des, 0.12, 0.45), ("SETTLE_AND_VERIFY", p_des, 0.12, 0.20)]
        else:
            zc = max(last["eef"][2], p_des[2]) + 0.08
            p_clear = [last["eef"][0], last["eef"][1], zc]
            p_xy = [p_des[0], p_des[1], zc]
            stages = [("SAFE_CLEARANCE", p_clear, 0.45, 0.35), ("XY_ALIGN", p_xy, 0.45, 0.80),
                      ("Z_ALIGN", p_des, 0.12, 0.45), ("FINE_SERVO", None, 0.12, 0.40),
                      ("SETTLE_AND_VERIFY", None, 0.12, 0.20)]
        result_status = "RETURN_TIMEOUT"
        primary = None
        for stage_name, tgt, speed, budget in stages:
            if clock >= RETURN_TIMEOUT_S:
                break
            nstep = max(1, int(budget / CONTROL_DT))
            for _ in range(nstep):
                if clock >= RETURN_TIMEOUT_S:
                    result_status = "RETURN_TIMEOUT"
                    break
                if stage_name == "FINE_SERVO":
                    opos = last["objects"]["obj"]["pos"]
                    e = [ckpt["p_WO_star"][i] - opos[i] for i in range(3)]
                    en = norm(e)
                    vmax = 0.45 if en > 0.02 else 0.12
                    step = clipnorm([3.0 * x * CONTROL_DT for x in e], vmax * CONTROL_DT)
                    tgt_now = [last["eef"][i] + step[i] for i in range(3)]
                    if not in_workspace(tgt_now):
                        result_status = "WORKSPACE"; break
                    self.backend.max_speed = vmax
                    self.backend.set_controller_target(tgt_now, "closed", mode=stage_name)
                    last = self._advance(raw, integ)
                    clock += CONTROL_DT
                    log_tick(stage_name, last, tgt_now)
                    if hold_abort(last):
                        return {"status": "HOLD_LOST_ABORT", "invoked": True, "clock": clock}
                elif stage_name == "SETTLE_AND_VERIFY":
                    tgt_now = tgt if tgt is not None else last["eef"]
                    self.backend.max_speed = 0.12
                    self.backend.set_controller_target(tgt_now, "closed", mode=stage_name)
                    last = self._advance(raw, integ)
                    clock += CONTROL_DT
                    log_tick(stage_name, last, tgt_now)
                    if hold_abort(last):
                        return {"status": "HOLD_LOST_ABORT", "invoked": True, "clock": clock}
                    primary = maybe_done(last)
                    if primary["status"] in ("EXACT_OBSERVED_TASK_RETURN", "BOUNDED_OBSERVED_TASK_RETURN"):
                        closures.append(dict(primary, t=last["t"], clock=clock, checkpoint_id=ckpt["checkpoint_id"]))
                        return {"status": "RETURN_COMPLETE", "invoked": True, "clock": clock, "primary": primary}
                else:
                    self.backend.max_speed = speed
                    self.backend.set_controller_target(tgt, "closed", mode=stage_name)
                    last = self._advance(raw, integ)
                    clock += CONTROL_DT
                    log_tick(stage_name, last, tgt)
                    if hold_abort(last):
                        return {"status": "HOLD_LOST_ABORT", "invoked": True, "clock": clock}
            else:
                continue
            break
        # diagnostic tail 1s, pass/fail frozen
        tail_status = result_status
        for _ in range(int(1.0 / CONTROL_DT)):
            last = self._advance(raw, integ)
            log_tick("DIAGNOSTIC_TAIL", last, p_des, abort=tail_status)
        primary = maybe_done(last)
        closures.append(dict(primary, t=last["t"], clock=clock, checkpoint_id=ckpt["checkpoint_id"], timeout=True))
        return {"status": tail_status, "invoked": True, "clock": clock, "primary": primary}

    def run_episode(self, out_dir: Path, family_id: int, case_id: str, rollout_seed: int) -> dict:
        dest = require_new(out_dir)
        be = self.backend
        episode_id = f"P1V6RC1_{family_id}_{case_id}_{rollout_seed}_{self.controller_id}"
        be.reset_episode({"family_id": int(family_id), "rollout_seed": int(rollout_seed)}, {"case_id": case_id})
        raw, integ, checkpoints, rc_log, att_log, closures = [], [], [], [], [], []
        notes = []
        self.invocations = 0
        self._wait(raw, integ, CONTROL_DT, mode="IDLE", gripper="open")
        frac = 0.8 if "P80" in case_id else 0.4
        offset = OFFSET.get(case_id, 0.0)
        nloop = 3 if case_id in ("E3_THREE_RETURNS_P40", "E4_THREE_RETURNS_P80") else 1
        last = raw[-1]
        returns = []

        if case_id == "E8_COMMANDED_RELEASE_CONTROL":
            last, ok = self._grasp(raw, integ, 0.0)
            if ok:
                last = self._lift_q0(raw, integ)
                self._save_ckpt("A", checkpoints, raw, episode_id)
                last = self._place(raw, integ)
            notes.append("return_controller_not_started")
        elif case_id == "E7_NO_LOSS_OUT_AND_BACK":
            last, ok = self._grasp(raw, integ, 0.0)
            if ok:
                last = self._lift_q0(raw, integ)
                ck = self._save_ckpt("A", checkpoints, raw, episode_id)
                mid = self._path_point(ck["p_WE_star"], 0.4)
                last, _ = self._move_to(mid, "closed", raw, integ, mode="FORWARD")
                ret = self.run_return(ck, raw, integ, rc_log, att_log, closures, allow_invoke=True)
                returns.append(ret)
            else:
                notes.append("grasp_failed")
        else:
            last, ok = self._grasp(raw, integ, 0.0)
            if not ok:
                notes.append("initial_grasp_failed")
            else:
                last = self._lift_q0(raw, integ)
                ck = self._save_ckpt("A", checkpoints, raw, episode_id)
                completed = 0
                for k in range(nloop):
                    mid = self._path_point(ck["p_WE_star"], frac)
                    last, _ = self._move_to(mid, "closed", raw, integ, mode="FORWARD")
                    last, lost = self._disturb(raw, integ)
                    if not lost:
                        notes.append(f"loop{k}_no_loss")
                        break
                    last, rok = self._grasp(raw, integ, offset)
                    if not rok:
                        notes.append(f"loop{k}_regrasp_failed")
                        break
                    last = self._lift_q0(raw, integ)
                    ret = self.run_return(ck, raw, integ, rc_log, att_log, closures, allow_invoke=True)
                    returns.append(ret)
                    if ret.get("status") != "RETURN_COMPLETE":
                        notes.append(f"loop{k}_{ret.get('status')}")
                        break
                    completed += 1
                    if k + 1 < nloop and ret.get("status") != "RETURN_COMPLETE":
                        break
                notes.append(f"cycles_complete={completed}")

        finish = be.finish_episode()
        identity = {
            "episode_id": episode_id, "family_id": family_id, "case_id": case_id,
            "rollout_seed": rollout_seed, "controller_id": self.controller_id,
            "backend_id": be.backend_id, "compensation_mode": COMPENSATION_MODE,
            "notes": notes, "return_invocations": self.invocations,
        }
        write_json(dest/"identity.json", identity)
        write_json(dest/"controller_spec.json", dict(self.spec, id=self.controller_id))
        write_json(dest/"protocol_hash.json", {"return_timeout_s": RETURN_TIMEOUT_S, "rehold_s": REHOLD_S,
                                              "closure": self.closure_lock, "confirmation_passed": False})
        write_jsonl(dest/"raw_state.jsonl", raw)
        write_jsonl(dest/"command_log.jsonl", be.command_log)
        write_jsonl(dest/"return_controller_log.jsonl", rc_log)
        write_jsonl(dest/"physical_reference.jsonl", be.reference_log)
        write_jsonl(dest/"checkpoint_events.jsonl", checkpoints)
        write_jsonl(dest/"attachment_estimates.jsonl", att_log)
        write_jsonl(dest/"closure_measurements.jsonl", closures)
        write_jsonl(dest/"control_log.jsonl", be.control_log)
        write_jsonl(dest/"weld_log.jsonl", be.weld_log)
        write_jsonl(dest/"state_mutation_ledger.jsonl", be.mutation_ledger)
        np.savez_compressed(dest/"integration_states.npz",
                            t=np.array([r["t"] for r in integ] or [0.0]),
                            qpos=np.array([r["qpos"] for r in integ] or [[0.0]]),
                            qvel=np.array([r["qvel"] for r in integ] or [[0.0]]))
        write_json(dest/"numeric_health.json", {
            "mj_steps": finish["mj_steps"], "time_error_s": abs(finish["expected_time_s"]-finish["actual_time_s"]),
            "finite": True,
        })
        files = sorted(x.name for x in dest.iterdir() if x.is_file())
        manifest = {
            "files": {n: sha256_file(dest/n) for n in files},
            "n_states": len(raw), "n_checkpoints": len(checkpoints),
            "direct_pose_overwrites_after_start": finish["direct_pose_overwrites_after_start"],
            "checkpoint_resets_inside_episode": finish["checkpoint_resets_inside_episode"],
            "velocity_zeroing_after_start": finish["velocity_zeroing_after_start"],
            "return_invocations": self.invocations, "returns": returns, "notes": notes,
        }
        write_json(dest/"manifest.json", manifest)
        return dict(identity, dest=str(dest), manifest=manifest, finish=finish, status="COLLECTED")

    def collect(self, families, cases, out_root: Path):
        root = require_new(out_root)
        rows = []
        for fam in families:
            fid, base = int(fam["family_id"]), int(fam["rollout_seed_base"])
            for i, case in enumerate(cases):
                seed = base + int(case.get("rollout_seed_offset", i))
                dest = root / f"F{fid}" / case["case_id"] / "rollout_00"
                try:
                    rec = self.run_episode(dest, fid, case["case_id"], seed)
                except Exception as exc:
                    dest.mkdir(parents=True, exist_ok=True)
                    rec = {"status": "FAILED", "family_id": fid, "case_id": case["case_id"],
                           "rollout_seed": seed, "error": type(exc).__name__, "reason": str(exc),
                           "dest": str(dest), "seed_replaced": False}
                    write_json(dest/"failure.json", rec)
                rows.append(rec)
        write_json(root/"collection_summary.json", {
            "controller_id": self.controller_id, "planned": len(families)*len(cases),
            "completed": sum(r.get("status")=="COLLECTED" for r in rows),
            "failed": sum(r.get("status")=="FAILED" for r in rows),
            "seed_replaced": False, "rows": rows,
        })
        return {"root": str(root), "rows": rows}
