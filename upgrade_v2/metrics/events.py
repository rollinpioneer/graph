"""Signed-reward event and cycle calculations."""
from __future__ import annotations

from typing import Iterable


def cycle_return(rewards: Iterable[float]) -> float:
    return float(sum(float(value) for value in rewards))


def potential_residual(rewards: Iterable[float], phi_start: float, phi_end: float) -> float:
    return cycle_return(rewards) - (float(phi_end) - float(phi_start))


def closure_kind(physical_closed: bool, semantic_closed: bool, full_input_closed: bool) -> str:
    if full_input_closed:
        return "full_input_closed"
    if semantic_closed:
        return "semantic_closed_only"
    if physical_closed:
        return "physical_closed_only"
    return "open"
