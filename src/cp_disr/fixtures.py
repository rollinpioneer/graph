"""Explicit synthetic diagnostics; never used by production training entrypoints."""
import json
from pathlib import Path
from .contracts import Registry,from_dict
from .facts import FactRecord,FactStore,Truth
from .graph import Goal,build_template
from .vlm import validate_relations
from .rl import Snapshot
from .common import digest

def build_fixture(root):
    root=Path(root)
    def load(name):
        value=json.loads((root/name).read_text())
        if value.get('synthetic_unit_fixture') is not True or value.get('paper_performance_eligible') is not False:raise ValueError('Fixture labeling required')
        return value
    c=load('contracts_v1.json');f=load('facts_tfu.json');g=load('nonredundant_support_graph.json')
    registry=Registry(c['predicate_types'])
    for item in c['contracts']:registry.register(from_dict(item))
    contracts=registry.ground(c['objects']);template=build_template(contracts,tuple(Goal(**x) for x in g['goals']),c['predicate_types'],c['objects'])
    facts=FactStore(tuple(FactRecord(k,Truth(v),f['capture_time'],f['available_time'],('frame0',),Truth.TRUE,0.,'MEASURED_SYNTHETIC',.8) for k,v in f['values'].items()))
    relations=validate_relations(g['payload'],template)
    return {'synthetic_unit_fixture':True,'template':template,'facts':facts,'contracts':contracts,'relations':g['payload'],'edges':relations.final_edges,'registry':registry,'objects':c['objects']}

def snapshot(f,env='e0',episode='ep0',decision=0,prior=True,base=(1.,0.,0.,0.)):
    edges=f['edges'] if prior else ()
    return Snapshot(env,episode,decision,f['template'],f['facts'],tuple(c.id for c in f['contracts']),tuple(True for _ in f['contracts']),edges,digest(edges),base,tuple((float(i%2),float(i%3),0.) for i,c in enumerate(f['contracts'])),'synthetic:frame0',float(decision),synthetic_unit_fixture=True)
