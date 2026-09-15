"""V6 engineering/confirmation collector. Reward is shadow-only.

Does not import upgrade_v2/p1_physical_mechanism_v5 World/collector.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .backend_adapter import (
    CONTROL_DT, HOLD_STABLE_S, PhysicalBackend, RETURN_TIMEOUT_S, family_physics,
)
from .util import require_new, sha256_file, write_json, write_jsonl


CASES = [
    "C1_NORMAL",
    "C2_LOSS_RETURN_P40",
    "C3_LOSS_RETURN_P80",
    "C4_THREE_RETURNS_P40",
    "C5_RECOVERY_COMMAND_NO_MOTION",
    "C6_PARTIAL_APPROACH_NO_REGRASP",
    "C7_NO_LOSS_OUT_AND_BACK",
    "C8_COMMANDED_RELEASE",
]


class CollectionError(RuntimeError):
    pass


class ScriptedCollector:
    def __init__(self, repo: Path, *, confirmation: bool = False):
        self.repo = Path(repo)
        self.confirmation = bool(confirmation)
        self.backend = PhysicalBackend(repo)

    def _sample(self, dest: Path, raw_states: list, integration: list, checkpoints: list):
        be = self.backend
        st = be.read_current_state()
        ref = be.reference_log[-1] if be.reference_log else be._update_reference()
        events = []
        for e in ref.get("events", []):
            events.append(dict(e, known_at_ns=st["physical_time_ns"]))
        obj = st["objects"]["obj"]
        raw_states.append({
            "t": st["t"],
            "physical_time_ns": st["physical_time_ns"],
            "capture_order": len(raw_states),
            "eef": st["eef"],
            "eef_quat": st["eef_quat"],
            "gripper_closed": st["gripper_closed"],
            "objects": {
                "obj": dict(obj, held=ref["held"], valid=ref["valid"], phase=ref["phase"],
                            ever_held=ref["ever_held"], open_loss_id=ref["open_loss_id"])
            },
            "events": events,
            "controller_mode": st["controller_mode"],
            "weld_active": st["weld_active"],
            "ncon": st["ncon"],
            "support": ref["support"],
            "recovery_command_issued": ref["recovery_command_issued"],
            "recovery_executing": ref["recovery_executing"],
            "release_commanded": ref["release_commanded"],
            "goal_xy_error": ref["goal_xy_error"],
        })
        integ = be.read_integration_state()
        integration.append({
            "t": float(integ["t"]),
            "qpos": integ["qpos"].tolist(),
            "qvel": integ["qvel"].tolist(),
            "mocap_pos": integ["mocap_pos"].tolist(),
            "eq_active": [bool(x) for x in integ["eq_active"].tolist()],
        })
        return raw_states[-1]

    def _advance(self, dest, raw_states, integration, checkpoints, n=1):
        last = None
        for _ in range(n):
            self.backend.advance_control_interval()
            last = self._sample(dest, raw_states, integration, checkpoints)
        return last

    def _move_to(self, xyz, gripper, dest, raw_states, integration, checkpoints, *, mode="MOVE", timeout_s=6.0, tol=0.012):
        self.backend.set_controller_target(xyz, gripper, mode=mode)
        steps = max(1, int(timeout_s / CONTROL_DT))
        last = None
        for _ in range(steps):
            last = self._advance(dest, raw_states, integration, checkpoints)
            eef = np.asarray(last["eef"], dtype=float)
            if float(np.linalg.norm(eef - np.asarray(xyz, dtype=float))) <= tol:
                # extra settle
                last = self._advance(dest, raw_states, integration, checkpoints, n=max(1, int(HOLD_STABLE_S / CONTROL_DT)))
                return last, True
        return last, False

    def _wait(self, dest, raw_states, integration, checkpoints, seconds, *, mode="WAIT", gripper=None):
        if gripper is None:
            gripper = "closed" if self.backend.gripper_closed else "open"
        pose = np.array(self.backend.data.mocap_pos[0], dtype=float)
        self.backend.set_controller_target(pose, gripper, mode=mode)
        n = max(1, int(round(seconds / CONTROL_DT)))
        return self._advance(dest, raw_states, integration, checkpoints, n=n)

    def _grasp(self, dest, raw_states, integration, checkpoints):
        spec = self.backend.spec
        obj = np.array(self.backend._object_xyz(), dtype=float)
        above = obj + np.array([0.0, 0.0, 0.16])
        last, ok = self._move_to(above, "open", dest, raw_states, integration, checkpoints, mode="APPROACH")
        if not ok:
            return last, False
        grasp = obj + np.array([0.0, 0.0, 0.13])
        last, ok = self._move_to(grasp, "open", dest, raw_states, integration, checkpoints, mode="DESCEND")
        last, ok2 = self._move_to(grasp, "closed", dest, raw_states, integration, checkpoints, mode="CLOSE")
        last = self._wait(dest, raw_states, integration, checkpoints, HOLD_STABLE_S + 0.04, mode="SETTLE", gripper="closed")
        return last, bool(last["objects"]["obj"]["held"])

    def _lift_q0(self, dest, raw_states, integration, checkpoints):
        eef = np.array(self.backend.data.mocap_pos[0], dtype=float)
        q0 = eef + np.array([0.0, 0.0, 0.18])
        last, ok = self._move_to(q0, "closed", dest, raw_states, integration, checkpoints, mode="LIFT")
        last = self._wait(dest, raw_states, integration, checkpoints, HOLD_STABLE_S, mode="Q0_SETTLE", gripper="closed")
        return last, np.array(last["eef"], dtype=float)

    def _save_ckpt(self, name, checkpoints, raw_states):
        integ = self.backend.read_integration_state()
        st = raw_states[-1]
        ck = {
            "name": name,
            "state_index": st["capture_order"],
            "t": st["t"],
            "eef": st["eef"],
            "object_pos": st["objects"]["obj"]["pos"],
            "object_quat": st["objects"]["obj"]["quat"],
            "object_vel": st["objects"]["obj"]["vel"],
            "held": st["objects"]["obj"]["held"],
            "phase": st["objects"]["obj"]["phase"],
            "target_xy": st["objects"]["obj"]["target_xy"],
            "gripper_closed": st["gripper_closed"],
            "controller_mode": st["controller_mode"],
            "qpos_saved_for_audit_only": True,
            "qpos_not_restored": True,
            "qpos": integ["qpos"].tolist(),
        }
        checkpoints.append(ck)
        return ck

    def _path_point(self, q0, frac):
        spec = self.backend.spec
        goal_eef = np.array([spec.target_xy[0], spec.target_xy[1], q0[2]], dtype=float)
        return q0 + (goal_eef - q0) * float(frac)

    def _disturb_until_loss(self, dest, raw_states, integration, checkpoints):
        spec = self.backend.spec
        # Constraint-break plus a modest lateral pulse; a 10N+ blast throws the object out of regrasp range.
        force = np.array([spec.disturb_xy[0], spec.disturb_xy[1], 0.0], dtype=float) * min(2.5, spec.disturb_force_n)
        self.backend.apply_declared_disturbance({"force_n": force.tolist(), "duration_s": 0.08, "breaks_weld": True})
        last = None
        for _ in range(int(1.2 / CONTROL_DT)):
            last = self._advance(dest, raw_states, integration, checkpoints)
            if any(e["kind"] == "LOSS" for e in last["events"]):
                self.backend.clear_disturbance()
                last = self._wait(dest, raw_states, integration, checkpoints, 0.12, mode="POST_LOSS", gripper="closed")
                return last, True
        self.backend.clear_disturbance()
        return last, False

    def _regrasp(self, dest, raw_states, integration, checkpoints):
        last, ok = self._grasp(dest, raw_states, integration, checkpoints)
        return last, ok

    def _return_to_ckpt(self, ckpt, dest, raw_states, integration, checkpoints):
        target = np.asarray(ckpt["eef"], dtype=float)
        last, ok = self._move_to(target, "closed", dest, raw_states, integration, checkpoints, mode="RETURN", timeout_s=RETURN_TIMEOUT_S)
        # 2s timeout is the contract; record whether bounded return is met
        last = self._wait(dest, raw_states, integration, checkpoints, HOLD_STABLE_S, mode="RETURN_SETTLE", gripper="closed")
        return last, ok

    def _place(self, dest, raw_states, integration, checkpoints):
        spec = self.backend.spec
        above = np.array([spec.target_xy[0], spec.target_xy[1], self.backend.data.mocap_pos[0][2]], dtype=float)
        last, _ = self._move_to(above, "closed", dest, raw_states, integration, checkpoints, mode="TRANSPORT")
        lower = np.array([spec.target_xy[0], spec.target_xy[1], spec.object_z0 + 0.13], dtype=float)
        last, _ = self._move_to(lower, "closed", dest, raw_states, integration, checkpoints, mode="LOWER")
        last, _ = self._move_to(lower, "open", dest, raw_states, integration, checkpoints, mode="RELEASE")
        last = self._wait(dest, raw_states, integration, checkpoints, 1.2, mode="SETTLE_PLACE", gripper="open")
        retreat = lower + np.array([0.0, 0.0, 0.18])
        last, _ = self._move_to(retreat, "open", dest, raw_states, integration, checkpoints, mode="RETREAT")
        last = self._wait(dest, raw_states, integration, checkpoints, HOLD_STABLE_S, mode="GOAL_WAIT", gripper="open")
        return last

    def run_episode(self, out_dir: Path, family_id: int, case_id: str, rollout_seed: int) -> dict:
        dest = require_new(out_dir)
        be = self.backend
        family_spec = {"family_id": int(family_id), "rollout_seed": int(rollout_seed)}
        case_spec = {"case_id": case_id}
        reset_info = be.reset_episode(family_spec, case_spec)
        raw_states: list[dict] = []
        integration: list[dict] = []
        checkpoints: list[dict] = []
        notes: list[str] = []
        # initial sample
        self._wait(dest, raw_states, integration, checkpoints, CONTROL_DT, mode="IDLE", gripper="open")
        loops_done = 0
        loss_happened = False
        returned = []

        def one_loss_return(frac):
            nonlocal loss_happened
            last, q0 = self._lift_q0(dest, raw_states, integration, checkpoints) if raw_states[-1]["objects"]["obj"]["held"] else (raw_states[-1], np.array(raw_states[-1]["eef"]))
            ck = self._save_ckpt(f"A_loop{loops_done}", checkpoints, raw_states)
            mid = self._path_point(np.asarray(ck["eef"]), frac)
            last, _ = self._move_to(mid, "closed", dest, raw_states, integration, checkpoints, mode="FORWARD")
            last, lost = self._disturb_until_loss(dest, raw_states, integration, checkpoints)
            loss_happened = loss_happened or lost
            if not lost:
                notes.append("expected_loss_not_observed")
                return last, ck, False, False
            last, rec_ok = self._regrasp(dest, raw_states, integration, checkpoints)
            if not rec_ok:
                notes.append("regrasp_failed")
            last, ret_ok = self._return_to_ckpt(ck, dest, raw_states, integration, checkpoints)
            returned.append({"ckpt": ck["name"], "returned_move_ok": ret_ok, "end_index": last["capture_order"]})
            return last, ck, lost, ret_ok

        last = raw_states[-1]
        if case_id != "C7_NO_LOSS_OUT_AND_BACK":
            last, grasped = self._grasp(dest, raw_states, integration, checkpoints)
            if not grasped:
                notes.append("initial_grasp_failed")

        if case_id == "C1_NORMAL":
            if last["objects"]["obj"]["held"]:
                last, q0 = self._lift_q0(dest, raw_states, integration, checkpoints)
                self._save_ckpt("A", checkpoints, raw_states)
            last = self._place(dest, raw_states, integration, checkpoints)
        elif case_id in ("C2_LOSS_RETURN_P40", "C3_LOSS_RETURN_P80"):
            frac = 0.4 if "P40" in case_id else 0.8
            if last["objects"]["obj"]["held"]:
                one_loss_return(frac)
            else:
                notes.append("skip_loop_without_hold")
        elif case_id == "C4_THREE_RETURNS_P40":
            if last["objects"]["obj"]["held"]:
                for k in range(3):
                    last, ck, lost, ret_ok = one_loss_return(0.4)
                    loops_done += 1
                    if not last["objects"]["obj"]["held"]:
                        notes.append(f"loop_{k}_not_held_after")
                        break
            else:
                notes.append("skip_loops_without_hold")
        elif case_id == "C5_RECOVERY_COMMAND_NO_MOTION":
            if last["objects"]["obj"]["held"]:
                last, q0 = self._lift_q0(dest, raw_states, integration, checkpoints)
                self._save_ckpt("A", checkpoints, raw_states)
                mid = self._path_point(q0, 0.4)
                last, _ = self._move_to(mid, "closed", dest, raw_states, integration, checkpoints, mode="FORWARD")
                last, lost = self._disturb_until_loss(dest, raw_states, integration, checkpoints)
                loss_happened = lost
            be.issue_recovery_command(execute=False)
            last = self._wait(dest, raw_states, integration, checkpoints, 0.6, mode="WAIT", gripper="closed")
        elif case_id == "C6_PARTIAL_APPROACH_NO_REGRASP":
            if last["objects"]["obj"]["held"]:
                last, q0 = self._lift_q0(dest, raw_states, integration, checkpoints)
                self._save_ckpt("A", checkpoints, raw_states)
                mid = self._path_point(q0, 0.4)
                last, _ = self._move_to(mid, "closed", dest, raw_states, integration, checkpoints, mode="FORWARD")
                last, lost = self._disturb_until_loss(dest, raw_states, integration, checkpoints)
                loss_happened = lost
            be.issue_recovery_command(execute=True)
            obj = np.array(be._object_xyz(), dtype=float)
            near = obj + np.array([0.0, 0.0, 0.18])
            last, _ = self._move_to(near, "open", dest, raw_states, integration, checkpoints, mode="PARTIAL_APPROACH")
            last = self._wait(dest, raw_states, integration, checkpoints, 0.3, mode="NO_CLOSE", gripper="open")
        elif case_id == "C7_NO_LOSS_OUT_AND_BACK":
            last, grasped = self._grasp(dest, raw_states, integration, checkpoints)
            if grasped:
                last, q0 = self._lift_q0(dest, raw_states, integration, checkpoints)
                ck = self._save_ckpt("A", checkpoints, raw_states)
                mid = self._path_point(q0, 0.4)
                last, _ = self._move_to(mid, "closed", dest, raw_states, integration, checkpoints, mode="FORWARD")
                last, ret_ok = self._return_to_ckpt(ck, dest, raw_states, integration, checkpoints)
                returned.append({"ckpt": "A", "returned_move_ok": ret_ok, "end_index": last["capture_order"]})
            else:
                notes.append("c7_grasp_failed")
        elif case_id == "C8_COMMANDED_RELEASE":
            if last["objects"]["obj"]["held"]:
                last, q0 = self._lift_q0(dest, raw_states, integration, checkpoints)
                self._save_ckpt("A", checkpoints, raw_states)
                last = self._place(dest, raw_states, integration, checkpoints)
            else:
                notes.append("c8_no_hold")
        else:
            raise CollectionError(f"unknown case {case_id}")

        finish = be.finish_episode()
        qpos = np.asarray(integration[-1]["qpos"], dtype=float) if integration else np.zeros(1)
        qvel = np.asarray(integration[-1]["qvel"], dtype=float) if integration else np.zeros(1)
        numeric = {
            "finite_qpos": bool(np.isfinite(qpos).all()),
            "finite_qvel": bool(np.isfinite(qvel).all()),
            "mj_steps": finish["mj_steps"],
            "expected_time_s": finish["expected_time_s"],
            "actual_time_s": finish["actual_time_s"],
            "time_error_s": abs(finish["expected_time_s"] - finish["actual_time_s"]),
        }
        identity = {
            "episode_id": f"P1V6_{family_id}_{case_id}_{rollout_seed}",
            "family_id": family_id,
            "case_id": case_id,
            "rollout_seed": rollout_seed,
            "confirmation": self.confirmation,
            "backend_id": be.backend_id,
            "evidence_tier": be.evidence_tier,
            "xml_sha256": be.xml_sha256,
            "mujoco_version": finish["mujoco_version"],
            "physics": reset_info["physics"],
            "notes": notes,
        }
        write_json(dest/"identity.json", identity)
        write_jsonl(dest/"raw_state.jsonl", raw_states)
        write_jsonl(dest/"command_log.jsonl", be.command_log)
        write_jsonl(dest/"intervention_log.jsonl", be.intervention_log)
        write_jsonl(dest/"state_mutation_ledger.jsonl", be.mutation_ledger)
        write_jsonl(dest/"physical_reference.jsonl", be.reference_log)
        write_jsonl(dest/"checkpoint_events.jsonl", checkpoints)
        write_jsonl(dest/"control_log.jsonl", be.control_log)
        write_jsonl(dest/"weld_log.jsonl", be.weld_log)
        # compact integration dump
        np.savez_compressed(dest/"integration_states.npz",
                            t=np.array([r["t"] for r in integration]),
                            qpos=np.array([r["qpos"] for r in integration]),
                            qvel=np.array([r["qvel"] for r in integration]))
        write_json(dest/"numeric_health.json", numeric)
        files = sorted(x.name for x in dest.iterdir() if x.is_file())
        manifest = {
            "files": {name: sha256_file(dest/name) for name in files},
            "n_states": len(raw_states),
            "n_checkpoints": len(checkpoints),
            "loss_happened": loss_happened,
            "returns": returned,
            "direct_pose_overwrites_after_start": finish["direct_pose_overwrites_after_start"],
            "checkpoint_resets_inside_episode": finish["checkpoint_resets_inside_episode"],
            "velocity_zeroing_after_start": finish["velocity_zeroing_after_start"],
            "notes": notes,
        }
        write_json(dest/"manifest.json", manifest)
        return dict(identity, dest=str(dest), manifest=manifest, numeric=numeric, finish=finish)

    def collect_plan(self, plan: dict, out_root: Path, *, confirmation: bool) -> dict:
        families = plan["confirmation_families"] if confirmation else plan["engineering_families"]
        seed_bases = plan["confirmation_rollout_seed_bases"] if confirmation else plan["engineering_rollout_seed_bases"]
        root = require_new(out_root)
        if confirmation:
            lock = root/"NO_RETRY_LOCK.json"
            write_json(lock, {"confirmation_retries": False, "started": True})
        rows = []
        for fam, base in zip(families, seed_bases):
            for i, case in enumerate(CASES):
                seed = int(base) + i
                dest = root / f"F{fam}" / case / "rollout_00"
                try:
                    rec = self.run_episode(dest, int(fam), case, seed)
                    rec["status"] = "COLLECTED"
                except Exception as exc:
                    dest.mkdir(parents=True, exist_ok=True)
                    rec = {"status": "FAILED", "family_id": fam, "case_id": case, "rollout_seed": seed,
                           "error": type(exc).__name__, "reason": str(exc), "dest": str(dest)}
                    write_json(dest/"failure.json", rec)
                    if confirmation:
                        # keep failure, do not replace seed
                        rec["seed_replaced"] = False
                rows.append(rec)
        write_json(root/"collection_summary.json", {
            "confirmation": confirmation,
            "planned": len(families) * len(CASES),
            "completed": sum(r.get("status") == "COLLECTED" for r in rows),
            "failed": sum(r.get("status") == "FAILED" for r in rows),
            "seed_replaced": False,
            "rows": rows,
        })
        return {"root": str(root), "rows": rows}
