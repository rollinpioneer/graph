"""Deterministic rigid-body-like tabletop physics. Camera unused in state contract."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

DT = 0.05
HOLD_TICKS = 5
PLACE_TICKS = 5
GRASP_R = 0.04
VEL_EPS = 0.05
ANG_EPS = 0.2
PLACE_XY = 0.04
PLACE_Z = 0.05
LOSS_HOLD_TICKS = 5


def family_params(family_seed: int) -> dict:
    rng = np.random.default_rng(family_seed)
    return dict(
        friction=float(0.35 + 0.25*rng.random()),
        mass=float(0.06 + 0.05*rng.random()),
        size=float(0.028 + 0.01*rng.random()),
        target_A=np.array([0.55 + 0.03*rng.normal(), 0.12 + 0.02*rng.normal(), 0.03], float),
        target_B=np.array([0.55 + 0.03*rng.normal(), -0.12 + 0.02*rng.normal(), 0.03], float),
        start_A=np.array([0.10 + 0.02*rng.normal(), 0.12, 0.03], float),
        start_B=np.array([0.10 + 0.02*rng.normal(), -0.12, 0.03], float),
        start_obj=np.array([0.12 + 0.02*rng.normal(), 0.0, 0.03], float),
        target=np.array([0.62 + 0.03*rng.normal(), 0.0, 0.03], float),
        loss_dir=np.array([float(rng.normal()), float(rng.normal()), 0.15]),
    )


@dataclass
class Body:
    pos: np.ndarray
    vel: np.ndarray
    held: bool = False


class World:
    def __init__(self, params: dict, seed: int):
        self.p = params
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.tick = 0
        self.eef = np.array([0.0, 0.0, 0.18], float)
        self.gripper_closed = False
        self.bodies = {
            "A": Body(params["start_A"].copy(), np.zeros(3)),
            "B": Body(params["start_B"].copy(), np.zeros(3)),
            "obj": Body(params["start_obj"].copy(), np.zeros(3)),
        }
        self.place_streak = {"A": 0, "B": 0, "obj": 0}
        self.hold_streak = {"A": 0, "B": 0, "obj": 0}
        self.loss_streak = {"A": 0, "B": 0, "obj": 0}
        self.valid = {"A": False, "B": False}
        self.established = {"A": False, "B": False, "obj": False}

    def _near(self, a, b, r):
        return float(np.linalg.norm(a-b)) <= r

    def _place_ok(self, name: str, target: np.ndarray) -> bool:
        b = self.bodies[name]
        xy = float(np.linalg.norm(b.pos[:2]-target[:2]))
        zok = abs(float(b.pos[2]-target[2])) <= PLACE_Z
        v = float(np.linalg.norm(b.vel))
        return (xy <= PLACE_XY) and zok and v <= VEL_EPS and not b.held

    def step(self, eef_cmd: np.ndarray, close: bool, hold_name: str | None):
        self.eef = 0.7*self.eef + 0.3*np.asarray(eef_cmd, float)
        self.gripper_closed = bool(close)
        for name, b in self.bodies.items():
            want = hold_name == name and close and self._near(self.eef, b.pos, GRASP_R)
            if want:
                self.hold_streak[name] += 1
            else:
                self.hold_streak[name] = 0
            if self.hold_streak[name] >= HOLD_TICKS:
                b.held = True
                self.established[name] = True
            if b.held and (not close or not self._near(self.eef, b.pos, GRASP_R*1.5)):
                b.held = False
            if b.held:
                b.pos = self.eef + np.array([0.0, 0.0, -0.05])
                b.vel = np.zeros(3)
            else:
                b.vel[2] -= 1.6*DT
                b.vel *= (1.0 - 0.15*self.p["friction"])
                b.pos = b.pos + b.vel*DT
                if b.pos[2] < 0.03:
                    b.pos[2] = 0.03
                    b.vel[2] = 0.0
                    b.vel[:2] *= 0.4
        for name, tgt in (("A", self.p["target_A"]), ("B", self.p["target_B"]), ("obj", self.p["target"])):
            if self._place_ok(name, tgt):
                self.place_streak[name] += 1
            else:
                self.place_streak[name] = 0
        for name in ("A", "B"):
            self.valid[name] = self.place_streak[name] >= PLACE_TICKS
        self.t += DT
        self.tick += 1

    def impulse_loss(self, name: str):
        b = self.bodies[name]
        d = self.p["loss_dir"]
        n = d / (np.linalg.norm(d)+1e-9)
        b.held = False
        self.hold_streak[name] = 0
        b.vel = 0.55*n
        b.pos = b.pos + 0.08*n

    def snapshot(self, names=("A","B","obj")) -> dict:
        out = dict(t=self.t, tick=self.tick, eef=self.eef.copy(), gripper_closed=self.gripper_closed)
        for n in names:
            b=self.bodies[n]
            out[n]=dict(pos=b.pos.copy(), vel=b.vel.copy(), held=b.held,
                        established=self.established[n],
                        dist_eef=float(np.linalg.norm(self.eef-b.pos)),
                        valid=self.valid.get(n, False),
                        place_streak=self.place_streak[n])
        return out