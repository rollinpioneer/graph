"""R14-B reproducible baseline execution chain.

The package deliberately keeps physical execution behind an externally signed,
single-use authorization.  Static protocol and comparison helpers are safe to
import without importing MuJoCo.
"""

PROTOCOL_ID = "L2RAR2_R14B_NEW_REPRODUCIBLE_BASELINE_V1"
SIM_VERSION = "l2rar2_reproducible_baseline_sim_v1"
CAPTURE_VERSION = "l2rar2_reproducible_baseline_capture_v1"
COMPARISON_VERSION = "l2rar2_reproducible_baseline_compare_v1"

__all__ = ["PROTOCOL_ID", "SIM_VERSION", "CAPTURE_VERSION", "COMPARISON_VERSION"]
