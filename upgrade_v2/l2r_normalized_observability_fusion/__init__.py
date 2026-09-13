"""R26 normalized online observability fusion.

The package deliberately keeps the frozen CLP3 executor outside this module.
It can only emit a proposal after a historical hold and never reads reference
data while processing an online stream.
"""

from .fusion import FusionProposal, LossObservabilityFusionV2_NormalizedEvidence

__all__ = ["FusionProposal", "LossObservabilityFusionV2_NormalizedEvidence"]
