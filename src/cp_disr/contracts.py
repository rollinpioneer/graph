"""Typed, immutable contract registry with simultaneous conditional effects."""
from dataclasses import dataclass,replace
from itertools import product
from .common import ContractError,unresolved
from .facts import Truth,negate,conjunction,ReadOnlyOverlay,derive

@dataclass(frozen=True,order=True)
class Atom:
    predicate:str
    arguments:tuple[str,...]=()
    def __post_init__(self): object.__setattr__(self,'arguments',tuple(self.arguments))
    @property
    def id(self): return 'p:'+self.predicate+(':'+':'.join(self.arguments) if self.arguments else '')
    def ground(self,binding): return Atom(self.predicate,tuple(binding.get(v,v) for v in self.arguments))

@dataclass(frozen=True)
class TypedArgument:
    name:str
    type:str

@dataclass(frozen=True)
class Effects:
    add:tuple[Atom,...]=()
    delete:tuple[Atom,...]=()
    unknown:tuple[Atom,...]=()
    def __post_init__(self):
        for name in ('add','delete','unknown'): object.__setattr__(self,name,tuple(getattr(self,name)))
        sets=[set(self.add),set(self.delete),set(self.unknown)]
        if any(sets[i]&sets[j] for i in range(3) for j in range(i)): raise ContractError('Conflicting ADD/DEL/UNKNOWN assignments')
    def assignments(self): return {a.id:t for group,t in [(self.add,Truth.TRUE),(self.delete,Truth.FALSE),(self.unknown,Truth.UNKNOWN)] for a in group}

@dataclass(frozen=True)
class ConditionalEffect:
    positive:tuple[Atom,...]
    negative:tuple[Atom,...]
    effects:Effects

@dataclass(frozen=True)
class SkillContract:
    name:str
    arguments:tuple[TypedArgument,...]
    pre_pos:tuple[Atom,...]
    pre_neg:tuple[Atom,...]
    effects:Effects
    version:str
    provenance:str
    controller_ref:str='MUST_BIND'
    verifier_ref:str='MUST_BIND'
    timeout_seconds:float|str='MUST_BIND'
    verification:tuple[str,...]=()
    failures:tuple[str,...]=('EXECUTION_FAILED','VERIFICATION_FAILED')
    interrupts:tuple[str,...]=('SAFETY_STOP','CRITICAL_FACT_LOST')
    conditional:tuple[ConditionalEffect,...]=()
    bound_arguments:tuple[str,...]=()
    parameter_version:str='v1'
    def __post_init__(self):
        for k in ('arguments','pre_pos','pre_neg','verification','failures','interrupts','conditional','bound_arguments'): object.__setattr__(self,k,tuple(getattr(self,k)))
        if not self.version or not self.provenance: raise ContractError('Contract version/provenance required')
        if set(self.pre_pos)&set(self.pre_neg): raise ContractError('Contradictory preconditions')
        if len({a.name for a in self.arguments})!=len(self.arguments): raise ContractError('Duplicate argument')
        if isinstance(self.timeout_seconds,(float,int)) and self.timeout_seconds<=0: raise ContractError('Nonpositive timeout')
        assigned=self.effects.assignments()
        for c in self.conditional:
            for key,value in c.effects.assignments().items():
                if key in assigned and assigned[key]!=value: raise ContractError('Potential conditional effect conflict: '+key)
                assigned[key]=value
    @property
    def id(self): return 'a:'+self.name+(':'+':'.join(self.bound_arguments) if self.bound_arguments else '')+':'+self.parameter_version
    def atoms(self):
        atoms=set(self.pre_pos+self.pre_neg+self.effects.add+self.effects.delete+self.effects.unknown)
        for c in self.conditional: atoms.update(c.positive+c.negative+c.effects.add+c.effects.delete+c.effects.unknown)
        return atoms
    def ground(self,binding):
        if set(binding)!={a.name for a in self.arguments}: raise ContractError('Incomplete grounding')
        def aa(seq):return tuple(a.ground(binding) for a in seq)
        def ee(e):return Effects(aa(e.add),aa(e.delete),aa(e.unknown))
        return replace(self,pre_pos=aa(self.pre_pos),pre_neg=aa(self.pre_neg),effects=ee(self.effects),conditional=tuple(ConditionalEffect(aa(c.positive),aa(c.negative),ee(c.effects)) for c in self.conditional),bound_arguments=tuple(binding[a.name] for a in self.arguments))

class Registry:
    def __init__(self,predicate_types):
        self.predicate_types={k:tuple(v) for k,v in predicate_types.items()}; self._skills={}
    def register(self,contract):
        if contract.name in self._skills: raise ContractError('Duplicate skill schema')
        args={a.name:a.type for a in contract.arguments}
        for atom in contract.atoms():
            expected=self.predicate_types.get(atom.predicate)
            if expected is None or len(expected)!=len(atom.arguments): raise ContractError('Unknown predicate/arity: '+atom.id)
            if any(v not in args or args[v]!=t for v,t in zip(atom.arguments,expected)): raise ContractError('Untyped or mismatched argument: '+atom.id)
        self._skills[contract.name]=contract
    def ground(self,objects):
        result=[]
        for c in self._skills.values():
            pools=[sorted(k for k,t in objects.items() if t==a.type) for a in c.arguments]
            for values in product(*pools): result.append(c.ground(dict(zip((a.name for a in c.arguments),values))))
        return tuple(sorted(result,key=lambda c:c.id))

def precondition_value(contract,facts):
    return conjunction([facts.get(a.id,Truth.UNKNOWN) for a in contract.pre_pos]+[negate(facts.get(a.id,Truth.UNKNOWN)) for a in contract.pre_neg])

def nominal_overlay(contract,facts,derived=(),exclusive_groups=()):
    if precondition_value(contract,facts)!=Truth.TRUE: raise ContractError('required_precondition_not_confirmed: '+contract.id)
    patch=contract.effects.assignments()
    for c in contract.conditional:
        guard=conjunction([facts.get(a.id,Truth.UNKNOWN) for a in c.positive]+[negate(facts.get(a.id,Truth.UNKNOWN)) for a in c.negative])
        for key,value in c.effects.assignments().items():
            if guard==Truth.TRUE: patch[key]=value
            elif guard==Truth.UNKNOWN:
                before=patch.get(key,facts.get(key,Truth.UNKNOWN))
                patch[key]=value if before==value else Truth.UNKNOWN
    view=ReadOnlyOverlay(facts,patch)
    final=derive(view,derived)
    for group in exclusive_groups:
        if sum(final.get(k)==Truth.TRUE for k in group)>1: raise ContractError('Mutually exclusive facts set TRUE')
    return ReadOnlyOverlay(facts,{k:v for k,v in final.items() if v!=facts[k]})

def from_dict(d):
    def atom(a): return Atom(a['predicate'],tuple(a.get('arguments',())))
    def effects(e):return Effects(*(tuple(atom(a) for a in e[k]) for k in ['add','delete','unknown']))
    return SkillContract(name=d['name'],arguments=tuple(TypedArgument(**a) for a in d['arguments']),
        pre_pos=tuple(atom(a) for a in d['initiation']['pre_pos']),pre_neg=tuple(atom(a) for a in d['initiation']['pre_neg']),
        effects=Effects(*(tuple(atom(a) for a in d['nominal'][k]) for k in ['add','delete','unknown'])),
        version=d['version'],provenance=d['provenance'],controller_ref=d['controller_ref'],verifier_ref=d['verifier_ref'],
        timeout_seconds=d['timeout_seconds'],verification=tuple(d['verification']),failures=tuple(d['failures']),interrupts=tuple(d['interrupts']),
        conditional=tuple(ConditionalEffect(tuple(atom(a) for a in c['pre_pos']),tuple(atom(a) for a in c['pre_neg']),effects(c['effects'])) for c in d.get('conditional',())))
