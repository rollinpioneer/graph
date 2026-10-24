"""Diagnostic-only straight-through decode; no model parameters or updates.

Forward is EG's exact hard choice. Backward is explicitly a biased surrogate,
not a derivative of argmax. Angle offsets respect parallel-jaw pi symmetry.
"""
import torch
import numpy as np

def angle_st(logits, hard, temperature=1.0):
    bins = torch.arange(12, device=logits.device, dtype=logits.dtype) * np.pi / 12
    offset = torch.remainder(bins[:, None] - hard.detach()[None] + np.pi/2, np.pi) - np.pi/2
    # Exactly perpendicular bin has no preferred direction under pi symmetry.
    offset = torch.where(offset.abs() > np.pi/2 - 1e-6, torch.zeros_like(offset), offset)
    soft = (torch.softmax(logits[:12] / temperature, dim=0) * offset).sum(0)
    return hard + (soft - soft.detach())

def depth_st(logits, hard, temperature=1.0):
    bins = (torch.arange(4, device=logits.device, dtype=logits.dtype) + 1) * .01
    soft = (torch.softmax(logits[:4] / temperature, dim=0) * bins[:, None]).sum(0)
    return hard + (soft - soft.detach())
