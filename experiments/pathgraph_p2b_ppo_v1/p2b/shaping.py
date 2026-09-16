"""Discount-correct PBRS arithmetic, separate from historical undiscounted V6."""
from __future__ import annotations
from dataclasses import dataclass
import math
from .io_utils import finite

def shaped_transition(task_reward,phi_before,phi_after,*,gamma=.99,beta=1.,terminated=False,truncated=False):
    if not (0<gamma<=1) or beta<0:raise ValueError('invalid gamma/beta')
    if type(terminated) is not bool or type(truncated) is not bool:raise TypeError('boolean boundary required')
    if terminated and truncated:raise ValueError('ambiguous termination and truncation')
    r,p,q=map(finite,(task_reward,phi_before,phi_after))
    effective=0. if terminated else q
    bonus=beta*(gamma*effective-p)
    return {'task_reward':r,'phi_before':p,'phi_after_raw':q,'phi_after_effective':effective,
            'shaping_reward':bonus,'training_reward':r+bonus,
            'undiscounted_delta_raw':q-p,
            'discount_correction_nonterminal':beta*(gamma-1)*q,
            'terminated':terminated,'truncated':truncated}

@dataclass
class DiscountAudit:
    gamma:float=.99
    beta:float=1.
    n:int=0
    initial:float|None=None
    last:float|None=None
    discounted_shaping:float=0.
    stopped:bool=False
    max_step_error:float=0.
    def add(self,row):
        if self.stopped:raise RuntimeError('reset audit after boundary')
        p,q=finite(row['phi_before']),finite(row['phi_after_effective'])
        if self.initial is None:self.initial=p
        elif not math.isclose(p,self.last,rel_tol=1e-10,abs_tol=1e-12):
            raise ValueError('potential discontinuity')
        exp=self.beta*(self.gamma*q-p)
        self.max_step_error=max(self.max_step_error,abs(row['shaping_reward']-exp))
        self.discounted_shaping+=self.gamma**self.n*row['shaping_reward']
        self.n+=1;self.last=q
        self.stopped=bool(row['terminated'] or row['truncated'])
    def result(self):
        if self.n==0:raise ValueError('empty audit')
        target=self.beta*(-self.initial+self.gamma**self.n*self.last)
        residual=self.discounted_shaping-target
        return {'n_steps':self.n,'discounted_shaping':self.discounted_shaping,
                'endpoint_prediction':target,'identity_error':residual,
                'max_step_error':self.max_step_error,
                'passed':math.isclose(self.discounted_shaping,target,rel_tol=1e-9,abs_tol=1e-10) and self.max_step_error<=1e-10}
