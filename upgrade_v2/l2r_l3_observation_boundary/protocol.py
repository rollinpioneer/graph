from . import RUNNER_PARENT

FROZEN_BLOBS={
 "clp3_guard":"03b428e2f0fdd215ce31c4149674781bf10129a2",
 "canonical_capture":"42f76ff62e2ccdf65984ed087ed5ba68b66ca9c1",
 "detector":"efe789eb3fe7f3b5d851ccd317aa40ca20bdead5",
 "contact_proxy":"93ec0eac24b3851bd6abfd189319757c85998cb0",
 "r23_controller":"0f9531bb32361df94a0a8ec6f8cd6ac134363ca4"
}

def protocol_lock():
 return {"schema":"l2rar2_r24_protocol_v1","base_commit":RUNNER_PARENT,"frozen_blobs":FROZEN_BLOBS,"candidate_modified":False,"l2r_parameters_modified":False,"r22_r23_read_only":True,"main_modified":False,"physical_rollouts":96,"special_outcome":"SIGNAL_NOT_OBSERVED","failure_stages":["SIGNAL_NOT_OBSERVED","TRIGGERING_ERROR","RELOCATION_ERROR","REGRASP_ERROR","TASK_RECOVERY_ERROR"]}
