from __future__ import annotations
from .return_geometry import relative_position_in_eef, quat_normalize, finite_vec

def make_checkpoint(*, checkpoint_id, episode_id, time_ns, state_index, p_obj, q_obj, v_obj, w_obj,
                    p_eef, q_eef, held, phase, target_xy, extra=None):
    p_obj, p_eef = finite_vec(p_obj), finite_vec(p_eef)
    q_eef = quat_normalize(q_eef)
    q_obj = quat_normalize(q_obj)
    r_eo = relative_position_in_eef(p_obj, p_eef, q_eef)
    extra = extra or {}
    return {
        "checkpoint_id": checkpoint_id,
        "episode_id": episode_id,
        "object_id": "obj",
        "time_ns": int(time_ns),
        "state_index": int(state_index),
        "p_WO_star": p_obj,
        "q_WO_star": q_obj,
        "v_WO_star": list(v_obj),
        "omega_WO_star": list(w_obj),
        "p_WE_star": p_eef,
        "q_WE_star": q_eef,
        "r_EO_star": r_eo,
        "q_EO_star": [1.0, 0.0, 0.0, 0.0],
        "hold_current": bool(held),
        "phase": phase,
        "graph_state": extra.get("graph_state"),
        "current_valid_subgoals": extra.get("current_valid_subgoals"),
        "goal_position": list(target_xy) + [None],
        "phi_abs": extra.get("phi_abs"),
        "capability_cost": extra.get("capability_cost"),
        "qpos_not_restored": True,
    }
