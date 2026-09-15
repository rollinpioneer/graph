"""Closed-loop regrasp collector. RC1 return and V6 backend remain frozen."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from upgrade_v2.p1_return_controller_v6rc1.collector import ReturnCollector, CONTROL_DT
from upgrade_v2.p1_return_controller_v6rc1.util import require_new, sha256_file, write_json, write_jsonl
from .tracking_controller import above_target, grasp_target, xy_error, z_error, norm, preclose_ok, next_attempt, in_workspace
from .regrasp_state_machine import (
    SETTLE_V, SETTLE_W, SETTLE_S, SETTLE_WAIT_MAX, MOVE_TRACK_V,
    APPROACH_SPEED, DESCEND_SPEED, RECENTER_SPEED, CLOSE_SPEED,
    XY_TOL, Z_TOL, PRECLOSE_TICKS, CLOSE_TRACK_S, VERIFY_S, ATTEMPT_TIMEOUT,
    RETRY_RETREAT, RETRY_WAIT, APPROACH_H, GRASP_H,
)

CASES = [
    "G1_LOSS_REGRASP_RETURN_P40","G2_LOSS_REGRASP_RETURN_P80",
    "G3_THREE_REGRASP_RETURN_P40","G4_THREE_REGRASP_RETURN_P80",
    "G5_HIGH_RESIDUAL_SPEED_P40","G6_LOW_FRICTION_LONG_SLIDE_P80",
    "G7_NO_LOSS_INITIAL_GRASP_CONTROL","G8_COMMANDED_RELEASE_NO_REGRASP",
]
RC1_SPEC = {"id": "RC1_OBJECT_COMPENSATED_DIRECT", "complexity_rank": 1, "max_speed_m_s": 0.45, "path": "direct"}


class RegraspCollector(ReturnCollector):
    def __init__(self, repo: Path, model_id: str):
        super().__init__(repo, "RC1_OBJECT_COMPENSATED_DIRECT", RC1_SPEC)
        self.model_id = model_id
        self.max_attempts = 2 if "ONE_RETRY" in model_id else 1
        self.regrasp_invocations = 0
        self.regrasp_logs = []
        self.regrasp_attempts = []
        self.case_id_for_log = None  # never passed into tracking decisions

    def _obj(self, last):
        return last["objects"]["obj"]

    def _tick_log(self, stage, last, target, attempt, consec=0, abort=None):
        o = self._obj(last)
        row = {
            "model_id": self.model_id, "attempt": attempt, "stage": stage, "t": last["t"],
            "object_pos": o["pos"], "object_vel": o["vel"], "object_ang": o["angular_vel"],
            "eef": last["eef"], "target": target, "xy_error": xy_error(last["eef"], o["pos"]),
            "z_error": z_error(last["eef"], o["pos"], GRASP_H),
            "rel": float(np.linalg.norm(np.asarray(o["pos"])-np.asarray(last["eef"]))),
            "held": o["held"], "weld": last.get("weld_active"), "gripper_closed": last["gripper_closed"],
            "consecutive_align": consec, "abort_reason": abort,
        }
        self.regrasp_logs.append(row)
        return row

    def _closed_loop_one(self, raw, integ, attempt: int):
        last = raw[-1]
        t0 = last["t"]
        settle_s = 0.0
        moving = False
        stage = "WAIT_OBJECT_SETTLE"
        while last["t"] - t0 < SETTLE_WAIT_MAX:
            o = self._obj(last)
            v = norm(o["vel"]); w = norm(o["angular_vel"])
            self._tick_log(stage, last, last["eef"], attempt)
            if v <= SETTLE_V and w <= SETTLE_W:
                settle_s += CONTROL_DT
            else:
                settle_s = 0.0
            if settle_s + 1e-12 >= SETTLE_S:
                break
            self.backend.set_controller_target(last["eef"], "open", mode=stage)
            last = self._advance(raw, integ)
        else:
            o = self._obj(last)
            if norm(o["vel"]) <= MOVE_TRACK_V and in_workspace(o["pos"]):
                moving = True
            else:
                return last, {"status": "OBJECT_NOT_CAPTURABLE", "attempt": attempt, "moving_track": False}
        # TRACK_ABOVE
        stage = "TRACK_ABOVE"
        while last["t"] - t0 < ATTEMPT_TIMEOUT:
            o = self._obj(last)
            tgt = above_target(o["pos"], APPROACH_H)
            if not in_workspace(tgt):
                return last, {"status": "WORKSPACE", "attempt": attempt}
            self.backend.max_speed = APPROACH_SPEED
            self.backend.set_controller_target(tgt, "open", mode=stage)
            last = self._advance(raw, integ)
            self._tick_log(stage, last, tgt, attempt)
            if xy_error(last["eef"], self._obj(last)["pos"]) <= 0.02 and abs(last["eef"][2]-tgt[2]) <= 0.02:
                break
        else:
            return last, {"status": "APPROACH_TIMEOUT", "attempt": attempt}
        # TRACK_DESCEND
        stage = "TRACK_DESCEND"
        while last["t"] - t0 < ATTEMPT_TIMEOUT:
            o = self._obj(last)
            tgt = grasp_target(o["pos"], GRASP_H)
            self.backend.max_speed = DESCEND_SPEED
            self.backend.set_controller_target(tgt, "open", mode=stage)
            last = self._advance(raw, integ)
            self._tick_log(stage, last, tgt, attempt)
            if z_error(last["eef"], self._obj(last)["pos"], GRASP_H) <= 0.015 and xy_error(last["eef"], self._obj(last)["pos"]) <= 0.02:
                break
        else:
            return last, {"status": "DESCEND_TIMEOUT", "attempt": attempt}
        # FINAL_RECENTER
        stage = "FINAL_RECENTER"
        consec = 0
        while last["t"] - t0 < ATTEMPT_TIMEOUT:
            o = self._obj(last)
            tgt = grasp_target(o["pos"], GRASP_H)
            self.backend.max_speed = RECENTER_SPEED
            self.backend.set_controller_target(tgt, "open", mode=stage)
            last = self._advance(raw, integ)
            o = self._obj(last)
            ok = xy_error(last["eef"], o["pos"]) <= XY_TOL and z_error(last["eef"], o["pos"], GRASP_H) <= Z_TOL and norm(o["vel"]) <= MOVE_TRACK_V
            consec = consec + 1 if ok else 0
            self._tick_log(stage, last, tgt, attempt, consec=consec)
            if consec >= PRECLOSE_TICKS:
                break
        else:
            return last, {"status": "RECENTER_TIMEOUT", "attempt": attempt}
        # CLOSE_TRACK
        stage = "CLOSE_TRACK"
        close_t0 = last["t"]
        while last["t"] - close_t0 < CLOSE_TRACK_S:
            o = self._obj(last)
            tgt = grasp_target(o["pos"], GRASP_H)
            self.backend.max_speed = CLOSE_SPEED
            self.backend.set_controller_target(tgt, "closed", mode="CLOSE")
            last = self._advance(raw, integ)
            self._tick_log(stage, last, tgt, attempt)
        # VERIFY_HOLD
        stage = "VERIFY_HOLD"
        held_s = 0.0
        verify_t0 = last["t"]
        while last["t"] - verify_t0 < max(VERIFY_S, 0.20):
            o = self._obj(last)
            tgt = grasp_target(o["pos"], GRASP_H)
            self.backend.set_controller_target(tgt, "closed", mode="CLOSE")
            last = self._advance(raw, integ)
            self._tick_log(stage, last, tgt, attempt)
            if last["objects"]["obj"]["held"]:
                held_s += CONTROL_DT
            else:
                held_s = 0.0
            if held_s + 1e-12 >= VERIFY_S:
                return last, {"status": "REGRASP_CONFIRMED", "attempt": attempt, "moving_track": moving, "time_s": last["t"]-t0}
        return last, {"status": "REGRASP_ATTEMPT_FAILED", "attempt": attempt, "moving_track": moving}

    def closed_loop_regrasp(self, raw, integ):
        self.regrasp_invocations += 1
        attempt = 1
        last = raw[-1]
        while True:
            last, rec = self._closed_loop_one(raw, integ, attempt)
            self.regrasp_attempts.append(rec)
            if rec["status"] == "REGRASP_CONFIRMED":
                return last, True
            nxt = next_attempt(attempt, self.max_attempts)
            if nxt is None:
                return last, False
            # bounded retry: open, retreat 0.10 m, wait 0.20 s
            eef = np.array(last["eef"], dtype=float)
            up = eef + np.array([0.0, 0.0, RETRY_RETREAT])
            self.backend.max_speed = 0.35
            self.backend.set_controller_target(up, "open", mode="BOUNDED_RETRY")
            for _ in range(int(0.6 / CONTROL_DT)):
                last = self._advance(raw, integ)
                self._tick_log("BOUNDED_RETRY", last, up.tolist(), nxt)
                if abs(last["eef"][2] - up[2]) <= 0.02:
                    break
            last = self._wait(raw, integ, RETRY_WAIT, mode="RETRY_WAIT", gripper="open")
            attempt = nxt

    def run_episode(self, out_dir: Path, family_id: int, case_id: str, rollout_seed: int) -> dict:
        # reuse parent infrastructure but G-cases and closed-loop regrasp after loss
        dest = require_new(out_dir)
        be = self.backend
        self.regrasp_invocations = 0
        self.regrasp_logs = []
        self.regrasp_attempts = []
        episode_id = f"P1V6RG1_{family_id}_{case_id}_{rollout_seed}_{self.model_id}"
        be.reset_episode({"family_id": int(family_id), "rollout_seed": int(rollout_seed)}, {"case_id": case_id})
        # G6 declared friction multiplier: geom friction only, not object qpos
        if case_id == "G6_LOW_FRICTION_LONG_SLIDE_P80":
            gid = int(be.mujoco.mj_name2id(be.model, be.mujoco.mjtObj.mjOBJ_GEOM, "object_geom"))
            be.model.geom_friction[gid] = be.model.geom_friction[gid] * 0.7
            be.mutation_ledger.append({"when": "after_reset_declared_case_param", "kind": "geom_friction_multiplier", "value": 0.7, "not_object_qpos": True})
        raw, integ, checkpoints, rc_log, att_log, closures = [], [], [], [], [], []
        notes = []
        self._wait(raw, integ, CONTROL_DT, mode="IDLE", gripper="open")
        frac = 0.8 if "P80" in case_id else 0.4
        nloop = 3 if case_id in ("G3_THREE_REGRASP_RETURN_P40", "G4_THREE_REGRASP_RETURN_P80") else 1
        last = raw[-1]
        returns = []
        if case_id == "G8_COMMANDED_RELEASE_NO_REGRASP":
            last, ok = super()._grasp(raw, integ, 0.0)
            if ok:
                last = self._lift_q0(raw, integ)
                self._save_ckpt("A", checkpoints, raw, episode_id)
                last = self._place(raw, integ)
            notes.append("closed_loop_regrasp_not_started")
        elif case_id == "G7_NO_LOSS_INITIAL_GRASP_CONTROL":
            last, ok = super()._grasp(raw, integ, 0.0)
            if ok:
                last = self._lift_q0(raw, integ)
                ck = self._save_ckpt("A", checkpoints, raw, episode_id)
                mid = self._path_point(ck["p_WE_star"], 0.4)
                last, _ = self._move_to(mid, "closed", raw, integ, mode="FORWARD")
                ret = self.run_return(ck, raw, integ, rc_log, att_log, closures, allow_invoke=True)
                returns.append(ret)
            else:
                notes.append("initial_grasp_failed")
        else:
            last, ok = super()._grasp(raw, integ, 0.0)
            if not ok:
                notes.append("initial_grasp_failed")
            else:
                last = self._lift_q0(raw, integ)
                ck = self._save_ckpt("A", checkpoints, raw, episode_id)
                completed = 0
                for k in range(nloop):
                    mid = self._path_point(ck["p_WE_star"], frac)
                    last, _ = self._move_to(mid, "closed", raw, integ, mode="FORWARD")
                    # G5 impulse
                    old_force = None
                    if case_id == "G5_HIGH_RESIDUAL_SPEED_P40":
                        spec = self.backend.spec
                        force = np.array([spec.disturb_xy[0], spec.disturb_xy[1], 0.0]) * min(2.5, spec.disturb_force_n) * 1.25
                        self.backend.apply_declared_disturbance({"force_n": force.tolist(), "duration_s": 0.08, "breaks_weld": True})
                        last = None
                        lost = False
                        for _ in range(int(1.2 / CONTROL_DT)):
                            last = self._advance(raw, integ)
                            if any(e["kind"] == "LOSS" for e in last["events"]):
                                self.backend.clear_disturbance(); lost = True
                                last = self._wait(raw, integ, 0.20, mode="POST_LOSS", gripper="closed")
                                break
                        if not lost:
                            self.backend.clear_disturbance()
                    else:
                        last, lost = self._disturb(raw, integ)
                    if not lost:
                        notes.append(f"loop{k}_no_loss"); break
                    last, rok = self.closed_loop_regrasp(raw, integ)
                    if not rok:
                        notes.append(f"loop{k}_regrasp_failed"); break
                    last = self._lift_q0(raw, integ)
                    ret = self.run_return(ck, raw, integ, rc_log, att_log, closures, allow_invoke=True)
                    returns.append(ret)
                    if ret.get("status") != "RETURN_COMPLETE":
                        notes.append(f"loop{k}_{ret.get('status')}"); break
                    completed += 1
                notes.append(f"cycles_complete={completed}")
        finish = be.finish_episode()
        identity = {
            "episode_id": episode_id, "family_id": family_id, "case_id": case_id,
            "rollout_seed": rollout_seed, "model_id": self.model_id,
            "return_controller_id": "RC1_OBJECT_COMPENSATED_DIRECT",
            "regrasp_invocations": self.regrasp_invocations, "notes": notes,
        }
        write_json(dest/"identity.json", identity)
        write_json(dest/"model_spec.json", {"model_id": self.model_id, "max_attempts": self.max_attempts})
        write_json(dest/"protocol_hash.json", {"confirmation_passed": False, "REL_HOLD_M": 0.02, "HOLD_STABLE_S": 0.10})
        write_jsonl(dest/"raw_state.jsonl", raw)
        write_jsonl(dest/"command_log.jsonl", be.command_log)
        write_jsonl(dest/"intervention_log.jsonl", be.intervention_log)
        write_jsonl(dest/"physical_reference.jsonl", be.reference_log)
        write_jsonl(dest/"checkpoint_events.jsonl", checkpoints)
        write_jsonl(dest/"regrasp_controller_log.jsonl", self.regrasp_logs)
        write_jsonl(dest/"regrasp_attempts.jsonl", self.regrasp_attempts)
        write_jsonl(dest/"return_controller_log.jsonl", rc_log)
        write_jsonl(dest/"attachment_estimates.jsonl", att_log)
        write_jsonl(dest/"closure_measurements.jsonl", closures)
        write_jsonl(dest/"state_mutation_ledger.jsonl", be.mutation_ledger)
        np.savez_compressed(dest/"integration_states.npz", t=np.array([r["t"] for r in integ] or [0.0]),
                            qpos=np.array([r["qpos"] for r in integ] or [[0.0]]),
                            qvel=np.array([r["qvel"] for r in integ] or [[0.0]]))
        write_json(dest/"numeric_health.json", {"mj_steps": finish["mj_steps"], "finite": True})
        files = sorted(x.name for x in dest.iterdir() if x.is_file())
        manifest = {
            "files": {n: sha256_file(dest/n) for n in files}, "n_states": len(raw),
            "direct_pose_overwrites_after_start": finish["direct_pose_overwrites_after_start"],
            "checkpoint_resets_inside_episode": finish["checkpoint_resets_inside_episode"],
            "velocity_zeroing_after_start": finish["velocity_zeroing_after_start"],
            "regrasp_invocations": self.regrasp_invocations, "returns": returns, "notes": notes,
            "attempts": self.regrasp_attempts,
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
            "model_id": self.model_id, "planned": len(families)*len(cases),
            "completed": sum(r.get("status")=="COLLECTED" for r in rows),
            "failed": sum(r.get("status")=="FAILED" for r in rows),
            "seed_replaced": False, "rows": rows,
        })
        return {"root": str(root), "rows": rows}
