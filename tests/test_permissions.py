import dataclasses,inspect
import pytest
from cp_disr.adapters import EvaluationInput
from cp_disr.contracts import precondition_value
from cp_disr.common import digest
from cp_disr.vlm import validate_relations
from cp_disr.skills import candidate_mask

@pytest.mark.pure
def test_T20_data_permissions(fixture):
    before=digest(fixture['facts']);mask=[precondition_value(c,fixture['facts'].values) for c in fixture['contracts']]
    validate_relations(fixture['relations'],fixture['template'])
    assert digest(fixture['facts'])==before
    assert mask==[precondition_value(c,fixture['facts'].values) for c in fixture['contracts']]
    fields={f.name for f in dataclasses.fields(EvaluationInput)}
    assert not fields&{'prior','dk','dp','nominal_facts','graph'}
    assert 'prior' not in inspect.signature(precondition_value).parameters
    checks={c.id:True for c in fixture['contracts']}
    ids,mask=candidate_mask(fixture['contracts'],fixture['facts'].values,checks)
    assert all(mask) and 'prior' not in inspect.signature(candidate_mask).parameters
    checks[ids[0]]=False
    assert candidate_mask(fixture['contracts'],fixture['facts'].values,checks)[1][0] is False
