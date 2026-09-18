"""Actual optimizer.step() accounting via PyTorch post-step hooks."""
from __future__ import annotations
from dataclasses import dataclass
from .errors import ProtocolViolation


class OptimizerStepCounter:
    def __init__(self):
        self.count = 0
        self._handle = None
        self._optimizer = None

    def hook(self, *args, **kwargs):
        self.count += 1

    def attach(self, optimizer):
        if optimizer is None:
            raise ProtocolViolation("OPTIMIZER_HOOK_UNSUPPORTED")
        if not hasattr(optimizer, "register_step_post_hook"):
            raise ProtocolViolation("OPTIMIZER_HOOK_UNSUPPORTED")
        self._optimizer = optimizer
        self._handle = optimizer.register_step_post_hook(self.hook)
        return self._handle

    def detach(self):
        handle = self._handle
        self._handle = None
        self._optimizer = None
        if handle is None:
            return
        remove = getattr(handle, "remove", None)
        if callable(remove):
            remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.detach()
        return False


@dataclass
class SegmentReceipt:
    requested_delta: int
    before: int
    after: int
    rollout_iterations: int
    ppo_epoch_count_delta: int
    optimizer_step_count_delta: int