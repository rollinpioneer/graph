import copy,json
from pathlib import Path
import pytest
from cp_disr.vlm import validate_relations
from cp_disr.graph import RELATIONS

@pytest.mark.pure
def test_T08_effect_ref(fixture):
    p=fixture['relations'];valid=validate_relations(p,fixture['template']);assert len(valid.final_edges)==1
    for key,value in [('source_ref','missing'),('target_ref','p:Held:red'),('effect_fact_ref','p:Open:box'),('effect_fact_ref','unknown')]:
        bad=copy.deepcopy(p);bad['relations'][0][key]=value;assert len(validate_relations(bad,fixture['template']).rejected)==1
    deletion=copy.deepcopy(p);deletion['relations'][0]['effect_fact_ref']='p:OnTable:red'
    assert len(validate_relations(deletion,fixture['template']).final_edges)==1
    from dataclasses import replace
    from cp_disr.contracts import Atom,Effects
    from cp_disr.graph import build_template
    move=next(c for c in fixture['contracts'] if c.name=='MOVE')
    modified=replace(move,effects=Effects(unknown=(Atom('AtBuffer',('red','buffer')),)))
    cs=tuple(modified if c.name=='MOVE' else c for c in fixture['contracts'])
    unknown_template=build_template(cs,fixture['template'].goals,fixture['registry'].predicate_types,fixture['objects'])
    assert validate_relations(p,unknown_template).rejected[0][1]=='UNREGISTERED_EFFECT_ANCHOR'
    redundant=copy.deepcopy(p);redundant['relations'][0].update(source_ref='a:PICK:red:v1',target_ref='a:OPEN:box:v1',effect_fact_ref='p:GripperEmpty')
    # DEL GripperEmpty duplicates a negative requirement only; change target precondition explicitly.
    opened=next(c for c in fixture['contracts'] if c.name=='OPEN')
    neg_open=replace(opened,pre_pos=(),pre_neg=(Atom('GripperEmpty'),))
    nt=build_template(tuple(neg_open if c.name=='OPEN' else c for c in fixture['contracts']),fixture['template'].goals,fixture['registry'].predicate_types,fixture['objects'])
    assert validate_relations(redundant,nt).rejected[0][1]=='CONTRACT_REDUNDANCY'
    dup=copy.deepcopy(p);row=dict(dup['relations'][0],relation_id='r2');dup['relations'].append(row)
    result=validate_relations(dup,fixture['template']);assert len(result.final_edges)==1 and len(result.dedup_log)==1
    goal=copy.deepcopy(p);goal['relations'][0].update(type='SOFT_RELEVANT_TO_GOAL',target_ref='goal:open')
    assert len(validate_relations(goal,fixture['template'],{'goal:open':'p:Open:box'}).final_edges)==1

@pytest.mark.pure
def test_T09_no_order(fixture):
    from jsonschema import Draft202012Validator
    schema=json.loads((Path(__file__).parents[1]/'schemas/relation_schema.json').read_text());validator=Draft202012Validator(schema)
    bad=copy.deepcopy(fixture['relations']);bad['relations'][0]['type']='SOFT_ORDER'
    assert list(validator.iter_errors(bad))
    assert validate_relations(bad,fixture['template']).rejected
    bad=copy.deepcopy(fixture['relations']);del bad['relations'][0]['effect_fact_ref']
    assert list(validator.iter_errors(bad)) and validate_relations(bad,fixture['template']).rejected
    assert all('ORDER' not in r for r in RELATIONS)
