"""Single un-clipped U1 potential and its signed transition difference."""
from __future__ import annotations


def phi(q: float, d_over_h: float, alpha: float = 0.5) -> float:
    return float(alpha) * float(q) - (1.0 - float(alpha)) * float(d_over_h)


def signed_reward(q_before: float, d_before: float, q_after: float, d_after: float, alpha: float = 0.5) -> dict[str, float]:
    before, after = phi(q_before, d_before, alpha), phi(q_after, d_after, alpha)
    return {"phi_before": before, "phi_after": after, "r_signed": after - before,
            "r_completion": float(q_after) - float(q_before), "r_efficiency": -(float(d_after) - float(d_before))}
