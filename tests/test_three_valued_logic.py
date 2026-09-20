from dataclasses import replace
import pytest
from cp_disr.facts import Truth,negate,conjunction,DerivedRule,derive
from cp_disr.contracts import Atom,Effects,ConditionalEffect,precondition_value,nominal_overlay

@pytest.mark.pure
def test_T06_three_values_and_unknown_guard(fixture):
    c=next(c for c in fixture['contracts'] if c.name=='OPEN');key=c.pre_pos[0].id
    for value,positive,negative in [(Truth.TRUE,Truth.TRUE,Truth.FALSE),(Truth.FALSE,Truth.FALSE,Truth.TRUE),(Truth.UNKNOWN,Truth.UNKNOWN,Truth.UNKNOWN)]:
        facts=dict(fixture['facts'].values);facts[key]=value
        assert precondition_value(c,facts)==positive
        assert precondition_value(replace(c,pre_pos=(),pre_neg=c.pre_pos),facts)==negative
    assert negate(Truth.UNKNOWN)==Truth.UNKNOWN
    assert conjunction([Truth.TRUE,Truth.UNKNOWN])==Truth.UNKNOWN
    facts=dict(fixture['facts'].values);facts['p:AtBuffer:red:buffer']=Truth.UNKNOWN
    conditional=ConditionalEffect((Atom('AtBuffer',('red','buffer')),),(),Effects(add=(Atom('Held',('red',)),)))
    patched=nominal_overlay(replace(c,conditional=(conditional,)),facts)
    assert patched['p:Held:red']==Truth.UNKNOWN
    assert patched['p:Open:box']==Truth.TRUE
    facts['p:Held:red']=Truth.TRUE
    assert nominal_overlay(replace(c,conditional=(conditional,)),facts)['p:Held:red']==Truth.TRUE
    assert derive({'a':Truth.TRUE,'b':Truth.UNKNOWN},[DerivedRule('b',('a',))])['b']==Truth.TRUE
