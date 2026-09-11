"""R16 controlled forced-drop experiment components.

The package keeps intervention commands, physical reference labels, and the
online observation contract separate.  MuJoCo is imported only by the runner
after authorization and budget checks have passed.
"""

PROTOCOL_ID = "L2RAR2_R16_CONTROLLED_FORCED_DROP_V1"
SIM_VERSION = "l2rar2_controlled_forced_drop_sim_v1"
REFERENCE_VERSION = "l2rar2_physical_loss_reference_v1"

__all__ = ["PROTOCOL_ID", "SIM_VERSION", "REFERENCE_VERSION"]
