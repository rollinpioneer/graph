"""Campaign runner errors. No training imports."""
from __future__ import annotations

class GateError(RuntimeError):
    pass

class IncompletePanel(RuntimeError):
    pass

class ResourceLimit(RuntimeError):
    pass

class ProtocolViolation(RuntimeError):
    pass
