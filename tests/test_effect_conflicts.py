import pytest
from dataclasses import replace
from cp_disr.contracts import Atom,Effects,nominal_overlay
from cp_disr.common import ContractError

@pytest.mark.pure
def test_T07_conflicts(fixture):
    atom=Atom('Held',('red',))
    with pytest.raises(ContractError):Effects(add=(atom,),delete=(atom,))
    with pytest.raises(ContractError):Effects(add=(atom,),unknown=(atom,))
    c=next(c for c in fixture['contracts'] if c.name=='PICK')
    values=dict(fixture['facts'].values);values['p:Held:blue']='TRUE'
    with pytest.raises(ContractError):nominal_overlay(c,values,exclusive_groups=[('p:Held:red','p:Held:blue')])
    assert nominal_overlay(c,fixture['facts'].values)['p:Held:red']=='TRUE'
