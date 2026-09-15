from __future__ import annotations
import math
from .return_geometry import relative_position_in_eef, finite_vec, quat_normalize

N_SAMPLES = 10
MAX_DISPERSION_M = 0.005

def _median(xs):
    ys = sorted(xs)
    n = len(ys)
    if n % 2:
        return ys[n//2]
    return 0.5*(ys[n//2-1]+ys[n//2])

def _mad(xs, med):
    dev = sorted(abs(x-med) for x in xs)
    n = len(dev)
    if n % 2:
        return dev[n//2]
    return 0.5*(dev[n//2-1]+dev[n//2])

class AttachmentEstimator:
    def __init__(self):
        self.samples = []

    def add(self, p_obj, p_eef, q_eef):
        p_obj, p_eef = finite_vec(p_obj), finite_vec(p_eef)
        q_eef = quat_normalize(q_eef)
        rel = relative_position_in_eef(p_obj, p_eef, q_eef)
        self.samples.append({"p_obj": p_obj, "p_eef": p_eef, "q_eef": q_eef, "r_eo": rel})
        return rel

    def estimate(self):
        if len(self.samples) != N_SAMPLES:
            raise ValueError("need %s samples, got %s" % (N_SAMPLES, len(self.samples)))
        xs = [s["r_eo"][0] for s in self.samples]
        ys = [s["r_eo"][1] for s in self.samples]
        zs = [s["r_eo"][2] for s in self.samples]
        med = [_median(xs), _median(ys), _median(zs)]
        disp = math.sqrt(_mad(xs, med[0])**2 + _mad(ys, med[1])**2 + _mad(zs, med[2])**2)
        stable = disp <= MAX_DISPERSION_M
        return {
            "r_eo_median": med,
            "dispersion_m": disp,
            "n": len(self.samples),
            "stable": stable,
            "status": "OK" if stable else "ATTACHMENT_UNSTABLE",
            "samples": self.samples,
            "compensation_mode": "TRANSLATION_COMPENSATED_FIXED_ORIENTATION",
        }
