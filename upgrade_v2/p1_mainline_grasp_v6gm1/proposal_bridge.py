"""No execution adapter: records incompatibility without substituting GT poses."""
from __future__ import annotations

def compatibility_report(provider: dict) -> dict:
    return {
        "status": "PROPOSAL_ONLY_BACKEND_INCOMPATIBLE" if provider.get("provider_status") not in
                  ("OFFICIAL_SAMPLE_INFERENCE_OK",) else "NEEDS_EXECUTION_CHECK",
        "reason": "Current V6 backend is mocap+weld, fixed orientation, 20mm centroid capture; 6-DoF network grasps are not executable without a new gripper. No GT pose substitution.",
        "evidence_tier": "DEPTH_CONSTRAINT_ASSISTED_EXECUTION_NOT_ATTEMPTED",
        "provider": provider.get("provider_status"),
    }
