"""L2RA-R2 geometry / height / history event diagnostic (R15-A).

Implements the P1 cache geometry baseline of MANUAL.md: a fixed-parameter
comparison of height, relative motion and history on the existing 32 repair
cache rollouts.  No physics, no simulator import, no model API.
"""

__all__ = ["adapters", "features", "state_machine", "evaluate"]
