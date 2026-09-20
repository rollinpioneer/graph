"""Immutable SLG topology. Torch encoder is separately imported only when requested."""
from dataclasses import dataclass
from .facts import Truth
from .contracts import Atom,nominal_overlay
from .common import ContractError

FORWARD_RELATIONS=('PRE_POS','PRE_NEG','ADD','DEL','SOFT_SUPPORTS','SOFT_RELEVANT_TO_GOAL')
RELATIONS=FORWARD_RELATIONS+tuple('REV_'+x for x in FORWARD_RELATIONS)

@dataclass(frozen=True)
class Node:
    id:str
    kind:str
    schema:str
    arguments:tuple[str,...]
    argument_types:tuple[str,...]

@dataclass(frozen=True)
class Goal:
    fact_id:str
    sign:int
    def __post_init__(self):
        if self.sign not in (-1,1): raise ContractError('Goal sign must be -1/+1')

@dataclass(frozen=True)
class GraphTemplate:
    nodes:tuple[Node,...]
    edges:tuple[tuple[str,str,str],...]
    goals:tuple[Goal,...]
    contracts:tuple
    derived_rules:tuple=()
    exclusive_groups:tuple=()
    @property
    def node_ids(self): return tuple(n.id for n in self.nodes)
    @property
    def goal_refs(self):return tuple((self.node_ids.index(g.fact_id),g.sign) for g in self.goals)

@dataclass(frozen=True)
class GraphView:
    template:GraphTemplate
    values:tuple[tuple[str,Truth],...]
    soft_edges:tuple=()
    @property
    def edges(self):return tuple(sorted(set(self.template.edges+self.soft_edges)))

def reverse_edges(edges): return tuple(sorted(set(edges)|{(b,a,'REV_'+r) for a,b,r in edges}))

def build_template(contracts,goals,predicate_types,objects,extra_atoms=(),derived_rules=(),exclusive_groups=()):
    contracts=tuple(sorted(contracts,key=lambda c:c.id)); goals=tuple(goals)
    if len({c.id for c in contracts})!=len(contracts): raise ContractError('Duplicate action ID')
    if len({g.fact_id for g in goals})!=len(goals): raise ContractError('Duplicate/conflicting goal')
    atoms={a.id:a for c in contracts for a in c.atoms()}; atoms.update({a.id:a for a in extra_atoms})
    if any(g.fact_id not in atoms for g in goals): raise ContractError('Goal not registered')
    nodes=[Node(c.id,'ACTION',c.name,c.bound_arguments,tuple(a.type for a in c.arguments)) for c in contracts]
    for a in atoms.values():
        if a.predicate not in predicate_types or tuple(objects[v] for v in a.arguments)!=tuple(predicate_types[a.predicate]): raise ContractError('Invalid grounded fact types')
        nodes.append(Node(a.id,'PROPOSITION',a.predicate,a.arguments,tuple(predicate_types[a.predicate])))
    edges=[]
    for c in contracts:
        for group,rel in [(c.pre_pos,'PRE_POS'),(c.pre_neg,'PRE_NEG')]:edges.extend((a.id,c.id,rel) for a in group)
        for effects in (c.effects,)+tuple(x.effects for x in c.conditional):
            edges.extend((c.id,a.id,'ADD') for a in effects.add);edges.extend((c.id,a.id,'DEL') for a in effects.delete)
    return GraphTemplate(tuple(sorted(nodes,key=lambda n:n.id)),reverse_edges(edges),goals,contracts,tuple(derived_rules),tuple(tuple(g) for g in exclusive_groups))

def view(template,values,soft_edges=()):
    facts={n.id for n in template.nodes if n.kind=='PROPOSITION'}
    if set(values)!=facts: raise ContractError('Snapshot facts do not match template')
    for a,b,r in soft_edges:
        if r not in ('SOFT_SUPPORTS','SOFT_RELEVANT_TO_GOAL'):raise ContractError('Invalid soft relation')
        if a not in template.node_ids or b not in template.node_ids:raise ContractError('Unknown soft endpoint')
    return GraphView(template,tuple(sorted((k,Truth(v)) for k,v in values.items())),reverse_edges(soft_edges))

def successor(graph,contract,derived=None,exclusive_groups=None):
    overlay=nominal_overlay(contract,dict(graph.values),graph.template.derived_rules if derived is None else derived,graph.template.exclusive_groups if exclusive_groups is None else exclusive_groups)
    return GraphView(graph.template,tuple(sorted(overlay.items())),graph.soft_edges)

def four_views(template,facts,prior_edges,contract,derived=None,exclusive_groups=None):
    k=view(template,facts); ki=successor(k,contract,derived,exclusive_groups)
    if not prior_edges:return k,k,ki,ki
    h=view(template,facts,prior_edges)
    return k,h,ki,GraphView(template,ki.values,h.soft_edges)
