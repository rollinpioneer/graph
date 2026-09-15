from __future__ import annotations
import math
from .return_geometry import desired_eef_position, clipnorm, finite_vec, in_workspace, COMPENSATION_MODE, norm
from .attachment_estimator import AttachmentEstimator, N_SAMPLES
from .state_machine import stages_for, BUDGET_S

REHOLD_S = 0.20
RETURN_TIMEOUT_S = 2.0
SETTLE_S = 0.20

class ReturnController:
    def __init__(self, controller_id: str, spec: dict):
        self.controller_id = controller_id
        self.spec = spec
        self.stage = "NOT_STARTED"
        self.return_clock_s = None
        self.invoked = False
        self.abort_reason = None
        self.estimate = None
        self.p_eef_des = None
        self.logs = []
        self.events = []
        self.stage_t = 0.0

    def _event(self, kind, **kw):
        self.events.append({"kind": kind, **kw})

    def plan_target(self, ckpt, q_eef, r_eo):
        p = desired_eef_position(ckpt["p_WO_star"], q_eef, r_eo)
        if not in_workspace(p):
            raise ValueError("workspace")
        self.p_eef_des = finite_vec(p)
        return self.p_eef_des

    def next_direct(self, p_eef, speed, dt):
        delta = [a-b for a,b in zip(self.p_eef_des, p_eef)]
        step = clipnorm(delta, speed * dt)
        return [p_eef[i]+step[i] for i in range(3)]

    def next_servo(self, p_obj, p_eef, dt):
        e = [a-b for a,b in zip(self.estimate and None, None)]
        e = [float(a)-float(b) for a,b in zip(self.ckpt_obj, p_obj)]
        en = norm(e)
        vmax = 0.45 if en > 0.02 else 0.12
        kp = float(self.spec.get("kp_per_s", 3.0))
        raw = [kp * x * dt for x in e]
        step = clipnorm(raw, vmax * dt)
        return [p_eef[i]+step[i] for i in range(3)], en

    def set_ckpt(self, ckpt):
        self.ckpt_obj = ckpt["p_WO_star"]
        self.ckpt = ckpt
