from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class RewardState:
    task_id: str
    episode_id: str
    step: int = 0
    edge_history: List[int] = field(default_factory=list)
    failure_debt: List[float] = field(default_factory=list)

@dataclass
class RewardResult:
    reward_mu: float
    reward_std: float
    reward_lcb: float
    weight_positive: float
    cost_delta_mu: float
    phi_delta_mu: float
    loop_penalty: float
    loop_count: int
    failure_debt_before: float
    failure_debt_after: float
    recovery_cap_applied: bool
    node_id_prev: int
    node_id_next: int
    edge_type_pred: int
    edge_id_pred: int
    node_confidence: float
    edge_confidence: float
    per_seed_reward: List[float] = field(default_factory=list)
    uncertainty_penalty: float = 0.0
    loop_count_skipped_low_confidence: bool = False
