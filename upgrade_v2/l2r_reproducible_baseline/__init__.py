"""R14-B reproducible baseline execution chain.

The package deliberately keeps physical execution behind an externally signed,
single-use authorization.  Static protocol and comparison helpers are safe to
import without importing MuJoCo.
"""

PROTOCOL_ID = "L2RAR2_R14B_NEW_REPRODUCIBLE_BASELINE_V2"
SIM_VERSION = "l2rar2_reproducible_baseline_sim_v2"
CAPTURE_VERSION = "l2rar2_reproducible_baseline_capture_v2"
COMPARISON_VERSION = "l2rar2_reproducible_baseline_compare_v2"

__all__ = ["PROTOCOL_ID", "SIM_VERSION", "CAPTURE_VERSION", "COMPARISON_VERSION"]
