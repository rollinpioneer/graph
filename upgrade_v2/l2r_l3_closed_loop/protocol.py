from __future__ import annotations

from . import BASE_COMMIT, CLP3_ID, METHODS, PROTOCOL_ID

FROZEN_BLOBS = {
    "clp3_guard": "03b428e2f0fdd215ce31c4149674781bf10129a2",
    "o_c3_evaluation": "f26153784e0375b4ca455b15325930e103d9c455",
    "online_interface": "0d6b2ff93892b18bf358517bbcf473c9458f5d4c",
    "hold_features": "c5ebe94e7999641dfe734ce185f5ae6692bb0e56",
    "hold_predicates": "949b96eadb7f83fd80b383b5e17d79fb10d44dff",
    "physical_reference": "a0e01362537bad716ebfffd88609f0ff4f59e982",
}

ENVIRONMENT = {
    "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1", "MUJOCO_GL": "egl",
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}


def protocol_lock() -> dict:
    return {
        "schema": "l2rar2_r23_l3_closed_loop_protocol_v1",
        "protocol_id": PROTOCOL_ID,
        "base_commit": BASE_COMMIT,
        "frozen_candidate": CLP3_ID,
        "frozen_blobs": FROZEN_BLOBS,
        "methods": list(METHODS),
        "paired_design": {"families": 4, "cases": 5, "methods": 3, "physical_rollouts": 60},
        "primary_metrics": [
            "final_task_success", "retry_recover_execution_success", "recovery_duration_s",
            "recovery_loops", "false_executed_actions",
        ],
        "failure_stages": [
            "TRIGGERING_ERROR", "RELOCATION_ERROR", "REGRASP_ERROR", "TASK_RECOVERY_ERROR",
        ],
        "controller_contract": {
            "recovery_loop_limit": 2, "hold_verification_duration_s": 0.50,
            "hold_verification_physics_hz": 100, "parameter_search": False,
        },
        "candidate_modification_allowed": False,
        "r22_read_only": True,
    }

