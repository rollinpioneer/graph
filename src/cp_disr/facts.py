"""Immutable measured facts and distinct nominal overlays; no environment effects."""
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from collections.abc import Mapping
from .common import ContractError

class Truth(str,Enum):
    TRUE='TRUE'; FALSE='FALSE'; UNKNOWN='UNKNOWN'

def negate(value):
    return {Truth.TRUE:Truth.FALSE,Truth.FALSE:Truth.TRUE,Truth.UNKNOWN:Truth.UNKNOWN}[Truth(value)]

def conjunction(values):
    values=tuple(Truth(v) for v in values)
    return Truth.FALSE if Truth.FALSE in values else Truth.UNKNOWN if Truth.UNKNOWN in values else Truth.TRUE

@dataclass(frozen=True)
class FactRecord:
    fact_id:str
    value:Truth
    capture_time:float
    available_time:float
    evidence_ids:tuple[str,...]=()
    last_confirmed_value:Truth=Truth.UNKNOWN
    last_confirmed_time:float|None=None
    reason:str=''
    quality:float|None=None
    hold_epoch:int=0
    def __post_init__(self):
        object.__setattr__(self,'value',Truth(self.value))
        object.__setattr__(self,'last_confirmed_value',Truth(self.last_confirmed_value))
        object.__setattr__(self,'evidence_ids',tuple(self.evidence_ids))
        if self.available_time<self.capture_time: raise ContractError('Evidence available before capture')

@dataclass(frozen=True)
class FactStore:
    records:tuple[FactRecord,...]
    def __post_init__(self):
        object.__setattr__(self,'records',tuple(sorted(self.records,key=lambda r:r.fact_id)))
        if len({r.fact_id for r in self.records})!=len(self.records): raise ContractError('Duplicate fact ID')
    @property
    def values(self): return MappingProxyType({r.fact_id:r.value for r in self.records})
    def replace_measured(self,records): return FactStore(tuple(records))

@dataclass(frozen=True)
class ReadOnlyOverlay(Mapping):
    base:Mapping
    patch:Mapping
    def __post_init__(self):
        object.__setattr__(self,'base',MappingProxyType(dict(self.base)))
        object.__setattr__(self,'patch',MappingProxyType({k:Truth(v) for k,v in self.patch.items()}))
        if not set(self.patch)<=set(self.base): raise ContractError('Patch contains unregistered facts')
    def __getitem__(self,key): return self.patch.get(key,self.base[key])
    def __iter__(self): return iter(self.base)
    def __len__(self): return len(self.base)

@dataclass(frozen=True)
class DerivedRule:
    target:str
    positive:tuple[str,...]
    negative:tuple[str,...]=()

def derive(values,rules):
    """Registered acyclic definitions only. Reject cycles/missing dependencies."""
    result=dict(values); pending=list(rules); targets={r.target for r in pending}; done=set(result)-targets
    if len(targets)!=len(pending): raise ContractError('Duplicate derived definition')
    while pending:
        ready=[r for r in pending if set(r.positive+r.negative)<=done]
        if not ready: raise ContractError('Cyclic or unregistered derived dependency')
        for rule in ready:
            if rule.target not in result: raise ContractError('Unregistered derived target')
            result[rule.target]=conjunction([result[p] for p in rule.positive]+[negate(result[p]) for p in rule.negative])
            done.add(rule.target); pending.remove(rule)
    return result
