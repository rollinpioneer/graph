"""Canonical capability candidates and mask independent of VLM/nominal successors."""
from .contracts import precondition_value
from .facts import Truth
from .common import ContractError

def candidate_mask(contracts,facts,execution_checks):
    ordered=tuple(sorted(contracts,key=lambda c:c.id))
    if len({c.id for c in ordered})!=len(ordered):raise ContractError('Duplicate grounded candidate')
    if not {c.id for c in ordered}<=set(execution_checks):raise ContractError('Missing independent safety/parameter checks')
    return tuple(c.id for c in ordered),tuple(precondition_value(c,facts)==Truth.TRUE and execution_checks[c.id] is True for c in ordered)
