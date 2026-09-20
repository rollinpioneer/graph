"""Pure data/time layer. Differentiable PPO and recurrence live in torch_rl."""
from dataclasses import dataclass
import math
from .common import DataIntegrityError,digest
from .facts import FactStore

def gamma(duration):
    if not math.isfinite(duration) or duration<=0:raise DataIntegrityError('Duration must be positive finite seconds')
    return 0.99**duration

def interval_reward(duration,events):
    gamma(duration); result=0.0
    for offset,reward in events:
        if not math.isfinite(offset) or not 0<=offset<=duration or reward not in (0,1):raise DataIntegrityError('Invalid independently confirmed reward event')
        result+=0.99**offset*reward
    if sum(v for _,v in events)>1:raise DataIntegrityError('Only first task success may reward')
    return result

def cumulative_weights(durations):
    w=1.;out=[]
    for d in durations:out.append(w);w*=gamma(d)
    return tuple(out)

@dataclass(frozen=True)
class Snapshot:
    env_id:str
    episode_id:str
    decision_id:int
    template:object
    facts:FactStore
    candidate_ids:tuple[str,...]
    mask:tuple[bool,...]
    prior_edges:tuple
    prior_hash:str
    base_input:tuple[float,...]
    candidate_features:tuple[tuple[float,...],...]
    observation_ref:str
    clock_seconds:float
    execution_summary:tuple=()
    synthetic_unit_fixture:bool=False
    def __post_init__(self):
        for k in ('candidate_ids','mask','prior_edges','base_input','candidate_features','execution_summary'):object.__setattr__(self,k,tuple(tuple(x) if isinstance(x,list) else x for x in getattr(self,k)))
        if len(set(self.candidate_ids))!=len(self.candidate_ids) or len(self.mask)!=len(self.candidate_ids) or len(self.candidate_features)!=len(self.candidate_ids):raise DataIntegrityError('Candidate identity/shape mismatch')
        if not set(self.candidate_ids)<={c.id for c in self.template.contracts}:raise DataIntegrityError('Unregistered candidate')
        if digest(self.prior_edges)!=self.prior_hash:raise DataIntegrityError('Stored prior hash mismatch')
    @property
    def mask_hash(self):return digest(sorted(zip(self.candidate_ids,self.mask)))

@dataclass(frozen=True)
class Transition:
    snapshot:Snapshot
    next_snapshot:Snapshot
    selected_candidate_id:str
    old_logp:float
    old_v:float
    old_v_next:float
    reward:float
    duration:float
    weight:float
    terminated:bool
    truncated:bool
    reason:str
    prefix:tuple[Snapshot,...]
    def __post_init__(self):
        gamma(self.duration)
        if self.selected_candidate_id not in self.snapshot.candidate_ids:raise DataIntegrityError('Selected ID not in snapshot')
        if not self.snapshot.mask[self.snapshot.candidate_ids.index(self.selected_candidate_id)]:raise DataIntegrityError('Selected masked action')
        if any(not math.isfinite(x) for x in [self.old_logp,self.old_v,self.old_v_next,self.reward,self.weight]):raise DataIntegrityError('Nonfinite transition')
        if self.weight<=0 or self.reward<0 or self.reward>1:raise DataIntegrityError('Invalid reward/weight')
        if self.next_snapshot.env_id!=self.snapshot.env_id or self.next_snapshot.episode_id!=self.snapshot.episode_id:raise DataIntegrityError('Bootstrap from reset/other episode prohibited')
        if self.next_snapshot.prior_hash!=self.snapshot.prior_hash:raise DataIntegrityError('Prior changed within episode')
        if self.next_snapshot.decision_id!=self.snapshot.decision_id+1:raise DataIntegrityError('Next snapshot not continuous')
        object.__setattr__(self,'prefix',tuple(self.prefix))
        if tuple(s.decision_id for s in self.prefix)!=tuple(range(self.snapshot.decision_id)):raise DataIntegrityError('Full recurrent prefix required')
        if any(s.env_id!=self.snapshot.env_id or s.episode_id!=self.snapshot.episode_id for s in self.prefix):raise DataIntegrityError('Prefix environment/episode mismatch')
    @property
    def Gamma(self):return gamma(self.duration)

def scalar_targets(transitions,lam=.95):
    """GAE never crosses a reset, truncation, absent successor, or env boundary."""
    adv=[0.]*len(transitions);vq=[0.]*len(transitions);qq=[0.]*len(transitions)
    for i in range(len(transitions)-1,-1,-1):
        t=transitions[i]; qq[i]=t.reward+t.Gamma*(not t.terminated)*t.old_v_next
        continuous=(not t.terminated and not t.truncated and i+1<len(transitions) and transitions[i+1].snapshot.env_id==t.snapshot.env_id and transitions[i+1].snapshot.episode_id==t.snapshot.episode_id and transitions[i+1].snapshot.decision_id==t.next_snapshot.decision_id)
        adv[i]=qq[i]-t.old_v+(t.Gamma*lam*adv[i+1] if continuous else 0.)
        vq[i]=t.old_v+adv[i]
    return tuple(adv),tuple(vq),tuple(qq)

def remap_stored(snapshot,candidate_ids,mask):
    if len(set(candidate_ids))!=len(candidate_ids) or set(candidate_ids)!=set(snapshot.candidate_ids):raise DataIntegrityError('Changed candidate set')
    if digest(sorted(zip(candidate_ids,mask)))!=snapshot.mask_hash:raise DataIntegrityError('Changed rollout mask')
    return tuple(snapshot.candidate_ids.index(k) for k in candidate_ids)

class Rollout:
    def __init__(self):self.transitions=[]
    def append(self,transition):
        if not isinstance(transition,Transition):raise DataIntegrityError('Real Transition required')
        self.transitions.append(transition)
    def clear(self):self.transitions.clear()
