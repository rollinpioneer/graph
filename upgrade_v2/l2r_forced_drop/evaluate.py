from __future__ import annotations
from .physical_reference import evaluate_loss_trace
def evaluate_trace(rows: list[dict]) -> dict:
    return evaluate_loss_trace(rows)
