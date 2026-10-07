"""Audited v3 reference algebra for next-event/action heads.

This is NOT integrated repository code. It accepts features from M1-GOAL,
keeps all goals in the shared context, and marginalizes over unsatisfied goals.
No simulator, planner, dataset loader, checkpoint, or optimizer is included.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class JointOutput:
    log_action: Tensor       # [B,K]
    log_goal: Tensor         # [B,G]
    log_joint: Tensor        # [B,G,K]
    residual: Tensor        # [B,G,K]


class NextGoalActionHead(nn.Module):
    def __init__(self, width: int = 128, seed: int = 20261007) -> None:
        super().__init__()
        if width < 1:
            raise ValueError('width must be positive')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.gate = nn.Sequential(nn.Linear(2*width, width), nn.ReLU(), nn.Linear(width, 1))
            self.residual = nn.Sequential(nn.Linear(3*width+2, width), nn.ReLU(), nn.Linear(width, 1))
            for final in (self.gate[-1], self.residual[-1]):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    def forward(self, z: Tensor, z_next: Tensor, truth: Tensor, truth_next: Tensor,
                base_logits: Tensor, goal_valid: Tensor, pending: Tensor,
                legal: Tensor) -> JointOutput:
        # z [B,G,D]; z_next [B,K,G,D]. Completed goals remain in z/context.
        if z.ndim != 3 or z_next.ndim != 4:
            raise ValueError('expected z[B,G,D] and z_next[B,K,G,D]')
        B, G, D = z.shape
        K = z_next.shape[1]
        if z_next.shape != (B,K,G,D) or base_logits.shape != (B,K):
            raise ValueError('incompatible feature/logit dimensions')
        expected = ((truth,(B,G)),(truth_next,(B,K,G)),(goal_valid,(B,G)),
                    (pending,(B,G)),(legal,(B,K)))
        if any(tuple(t.shape) != shape for t,shape in expected):
            raise ValueError('incompatible truth/mask dimensions')
        if any(t.dtype != torch.bool for t in (goal_valid,pending,legal)):
            raise TypeError('masks must be bool')
        if bool((pending & ~goal_valid).any()):
            raise ValueError('pending goal must be valid')
        if not bool(pending.any(-1).all() and legal.any(-1).all()):
            raise ValueError('terminal or no-legal-action rows must be handled by runner')
        if not bool(torch.isfinite(base_logits[legal]).all()):
            raise ValueError('legal base logits must be finite')
        mask = goal_valid.unsqueeze(-1).to(z.dtype)
        context = (z*mask).sum(1) / mask.sum(1).clamp_min(1)
        gate_in = torch.cat((z,context[:,None,:].expand(B,G,D)),dim=-1)
        goal_logits = self.gate(gate_in).squeeze(-1).masked_fill(~pending,-torch.inf)
        log_q = F.log_softmax(goal_logits,dim=-1)
        current = z[:,None,:,:].expand(B,K,G,D)
        r_in = torch.cat((current,z_next,z_next-current,
                          truth[:,None,:,None].expand(B,K,G,1),truth_next[...,None]),dim=-1)
        delta = self.residual(r_in).squeeze(-1).transpose(1,2)   # [B,G,K]
        logits = (base_logits[:,None,:] + delta).masked_fill(~legal[:,None,:],-torch.inf)
        log_cond = F.log_softmax(logits,dim=-1)
        log_joint = log_q[:,:,None] + log_cond
        # Invalid action columns would otherwise reduce all -inf across goals,
        # producing undefined backward even if subsequently masked in the loss.
        safe_joint = log_joint.masked_fill(~legal[:,None,:],0.0)
        log_action = torch.logsumexp(safe_joint,dim=1).masked_fill(~legal,-torch.inf)
        return JointOutput(log_action,log_q,log_joint,delta)



def _validate_targets(out: JointOutput, optimal: Tensor, legal: Tensor,
                      pending: Tensor, labels: Tensor, complete: Tensor) -> tuple[Tensor, Tensor]:
    """Return safe full pair labels and their Cartesian projection target."""
    B, K = out.log_action.shape
    G = out.log_goal.shape[1]
    if any(t.dtype != torch.bool for t in (optimal, legal, pending, labels, complete)):
        raise TypeError("targets and masks must be boolean")
    if (optimal.shape != (B,K) or legal.shape != (B,K) or
        pending.shape != (B,G) or labels.shape != (B,G,K) or complete.shape != (B,)):
        raise ValueError("target shape mismatch")
    if not bool(optimal.any(-1).all()):
        raise ValueError("terminal/unknown-optimal states require caller handling")
    if bool((optimal & ~legal).any()):
        raise ValueError("an optimal action cannot be illegal")
    allowed = pending[:,:,None] & optimal[:,None,:]
    if bool((labels & ~allowed & complete[:,None,None]).any()):
        raise ValueError("complete pair labels contain invalid goals/actions")
    if bool(((labels.any(1) != optimal) & complete[:,None]).any()):
        raise ValueError("complete pair labels must project to all optimal actions")
    safe_y = torch.where(complete[:,None,None], labels, allowed)
    qset = safe_y.any(-1)
    safe_fact = qset[:,:,None] & optimal[:,None,:]
    return safe_y, safe_fact


def label_information(labels: Tensor, optimal: Tensor, pending: Tensor,
                      complete: Tensor) -> dict[str, Tensor]:
    """Per-root information flags; unknown labels never count as informative.

    "Pair information" means Y is not Q x A, not merely Y != U x A.
    """
    B,G,K = labels.shape
    if optimal.shape != (B,K) or pending.shape != (B,G) or complete.shape != (B,):
        raise ValueError("shape mismatch")
    if any(t.dtype != torch.bool for t in (labels,optimal,pending,complete)):
        raise TypeError("boolean masks required")
    Q = labels.any(-1)
    cart = Q[:,:,None] & optimal[:,None,:]
    pair = (labels != cart).flatten(1).any(-1) & complete
    event = (Q != pending).any(-1) & complete
    singleton = (Q.sum(-1) == 1) & ~pair & complete
    n_pairs = labels.flatten(1).sum(-1)
    denom = (Q.sum(-1) * optimal.sum(-1)).clamp_min(1)
    density = n_pairs.to(torch.float64) / denom
    return {
        "event_information": event,
        "pair_information": pair,
        "singleton_event_rectangle": singleton,
        "pair_density": torch.where(complete, density, torch.nan),
        "event_count": Q.sum(-1),
        "optimal_count": optimal.sum(-1),
    }


def loss_terms_v3(out: JointOutput, optimal: Tensor, distances: Tensor,
                  legal: Tensor, pending: Tensor, labels: Tensor,
                  complete: Tensor, mode: str = "JOINT",
                  rank_weight: float = 1.0) -> dict[str, Tensor]:
    """Per-decision BASE/FACT/JOINT losses; caller applies trajectory weights.

    FACT uses only Q=projection_goal(Y) and A=optimal actions.
    It has the same model and set-likelihood form as JOINT.
    Unknown labels have no auxiliary supervision.
    """
    if mode not in {"BASE","FACT","JOINT"}:
        raise ValueError("mode must be BASE, FACT or JOINT")
    if distances.shape != out.log_action.shape:
        raise ValueError("distance shape mismatch")
    safe_y,safe_fact = _validate_targets(out,optimal,legal,pending,labels,complete)
    log_pa = torch.logsumexp(out.log_action.masked_fill(~optimal,-torch.inf),-1)
    log_pf = torch.logsumexp(out.log_joint.masked_fill(~safe_fact,-torch.inf).flatten(1),-1)
    log_py = torch.logsumexp(out.log_joint.masked_fill(~safe_y,-torch.inf).flatten(1),-1)
    score = out.log_action.masked_fill(~legal,0.)
    diff = score[:,:,None] - score[:,None,:]
    pair_mask = (distances[:,:,None] < distances[:,None,:]) & legal[:,:,None] & legal[:,None,:]
    count = pair_mask.sum((1,2)).clamp_min(1)
    rank = (F.softplus(-diff) * pair_mask).sum((1,2)) / count
    target = {"BASE":log_pa,"FACT":log_pf,"JOINT":log_py}[mode]
    return {
        "loss": -target + rank_weight*rank,
        "action_nll": -log_pa,
        "rank": rank,
        "fact_organisation": log_pa-log_pf,
        "joint_organisation": log_pa-log_py,
        "pair_specific": log_pf-log_py,
        "log_pa": log_pa,
        "log_pf": log_pf,
        "log_py": log_py,
    }


def calibrated_logits(v: Tensor, v_next: Tensor, legal: Tensor,
                       a0: float, c0: float, log_beta: Tensor) -> Tensor:
    """Common positive-temperature affine reparameterization for ABS and REL."""
    if a0 <= 0 or not torch.isfinite(torch.as_tensor(a0)):
        raise ValueError("a0 must be positive and finite")
    if v.ndim != 1 or v_next.shape[0] != v.shape[0] or legal.shape != v_next.shape:
        raise ValueError("expected v[B], v_next[B,K], legal[B,K]")
    if legal.dtype != torch.bool:
        raise TypeError("legal must be boolean")
    current = a0*v+c0
    nxt = a0*v_next+c0
    return (log_beta.exp()*(current[:,None]-nxt)).masked_fill(~legal,-torch.inf)


def calibration_objectives(w: Tensor, d_star: Tensor, pairs: Tensor,
                            scale: float = 1.) -> dict[str, Tensor]:
    """ABS uses absolute targets; REL uses only pair differences.

    Both use the same endpoint pool. This is an objective comparison,
    not a claim that relative labels contain information absent from d_star.
    """
    if w.ndim != 1 or d_star.shape != w.shape:
        raise ValueError("w and d_star must be vectors")
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype != torch.long:
        raise ValueError("pairs must be LongTensor[M,2]")
    if scale <= 0:
        raise ValueError("scale must be positive")
    absolute = F.smooth_l1_loss((w-d_star)/scale,torch.zeros_like(w),reduction="mean")
    if len(pairs):
        i,j = pairs[:,0],pairs[:,1]
        residual = ((w[i]-w[j])-(d_star[i]-d_star[j]))/scale
        relative = F.smooth_l1_loss(residual,torch.zeros_like(residual),reduction="mean")
    else:
        relative = w.sum()*0
    return {"absolute":absolute,"relative":relative}
