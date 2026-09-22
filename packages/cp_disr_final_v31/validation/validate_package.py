"""Validate deliverable definitions and arithmetic, not production CP-DISR."""
from pathlib import Path
import json, yaml, re, zipfile, hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 checks=[]
 def ck(name,ok):
  checks.append({'name':name,'passed':bool(ok)})
 for p in ROOT.rglob('*.yaml'):
  yaml.safe_load(p.read_text())
 for p in ROOT.rglob('*.json'):
  if p.name not in ['package_validation.json','package_checksums.json']:
   json.loads(p.read_text())
 ck('json_yaml_parse',True)
 m=yaml.safe_load((ROOT/'interfaces/paper_method_manifest.yaml').read_text())
 plan=yaml.safe_load((ROOT/'experiments/experiment_manifest.v1.1.yaml').read_text())
 ck('versions',m['document']['version']=='3.1' and m['method']['version']=='2.1.1' and plan['plan_version']=='1.1')
 # Find the actual configuration fields without relying on serialization order.
 def walk(x):
  if isinstance(x,dict):
   for k,v in x.items():
    yield k,v
    yield from walk(v)
  elif isinstance(x,list):
   for v in x:yield from walk(v)
 fields=list(walk(m))
 pref=[v for k,v in fields if k in ('actor_episode_discount_weight','actor_prefix_discount_weight')]
 ck('actor_prefix_off',bool(pref) and all(v is False for v in pref))
 ck('pure_interaction',m['policy']['static_prior'] is False and m['policy']['direct_current_GH_context'] is False)
 ck('graph_unchanged',m['graph']['hidden_dim']==128 and m['graph']['layers']==4 and m['graph']['dropout']==0)
 ck('future_disabled',m['method']['future_m1_plus_enabled'] is False)
 ck('H_runtime_bound_not_point99',plan['duration_discount']['half_life_seconds']=='MUST_BIND' and plan['duration_discount']['use_test_for_H'] is False)
 ck('no_random_five_percent_gate',plan['task_gate']['hard_success_rate_threshold'] is None)
 report=(ROOT/'review/CP_DISR_Independent_Adjudication_v1.md').read_text()
 ck('25_review_sections',all(re.search(r'^# '+str(i)+r'\. ',report,re.M) for i in range(1,26)))
 spec=(ROOT/'research/CP_DISR_Paper_Oriented_Research_Specification_v3.1.md').read_text()
 ck('41_research_sections',all(re.search(r'^# '+str(i)+r'\. ',spec,re.M) for i in range(1,42)))
 ck('no_stale_active_appendix_version','method_version为2.1、document_version为3.0' not in spec)
 for num,name in [(35,'Abstract'),(36,'Introduction'),(37,'Problem_Formulation'),(38,'Method'),(39,'Experimental_Setup'),(40,'Discussion')]:
  a=spec.index(f'# {num}. ');b=spec.index(f'# {num+1}. ',a)
  ck('draft_'+name,(ROOT/f'paper_drafts/{name}_v1.md').read_text()==spec[a:b].rstrip()+'\n')
 ck('18_stage_cards',len(list((ROOT/'experiments/stages').glob('*.md')))==18)
 for f in (ROOT/'experiments/stages').glob('*.md'):
  t=f.read_text();c=len(re.findall(r'^\d+\. ',t.split('## Agent Execution Template')[1],re.M))
  ck(f.stem+'_steps',8<=c<=15)
 for f in (ROOT/'experiments/status').glob('*.json'):
  st=json.loads(f.read_text());ck(f.stem+'_not_started',st['status']=='NOT_STARTED')
 rows=json.loads((ROOT/'experiments/planned_runs.json').read_text())
 ids={r['planned_id'] for r in rows}
 ck('unique35',len(ids)==len(rows)==35)
 first=[r for r in rows if r['phase']=='first_pause']
 ck('first11',len(first)==11 and sum(r['N_cap'] for r in first)==622592)
 ck('no_executed_runs',all(r['status']=='NOT_STARTED' and r['actual_run_id'] is None for r in rows))
 ck('control_refs_resolve',all(not r['control_reference'] or r['control_reference'] in ids for r in rows))
 ck('updates_consistent',all(r['N_cap']//1024==r['updates_cap'] and r['N_cap']%1024==0 for r in rows))
 ck('max35_budget',sum(r['N_cap'] for r in rows)==2195456)
 im=json.loads((ROOT/'sources/input_manifest.json').read_text())
 for v in im['inputs']:
  p=ROOT/v['path'];ck('source_'+v['name'],p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'])
 zpath=ROOT/'sources/CP_DISR_Paper_Oriented_Research_Package_v3.0(1).zip'
 with zipfile.ZipFile(zpath) as z:
  for name in ['relation_schema.json','system_prompt.txt','place_contract.example.yaml']:
   candidates=[n for n in z.namelist() if n.endswith('/interfaces/'+name)]
   ck('unchanged_'+name,len(candidates)==1 and z.read(candidates[0])==(ROOT/'interfaces'/name).read_bytes())
 for folder in ['review','research','paper_drafts','experiments','interfaces','figures']:
  for p in (ROOT/folder).rglob('*.md'):
   t=p.read_text()
   ck('text_'+str(p.relative_to(ROOT)),not any(ord(c)<32 and c not in '\n\r\t' for c in t) and t.count('$$')%2==0)
 algebra=json.loads((ROOT/'validation/algebra_report.json').read_text())
 ck('algebra_witnesses',algebra['passed'] and not algebra['production_RGCN_run'] and not algebra['RL_training_run'])
 failed=[x for x in checks if not x['passed']]
 result={'scope':'document_configuration_source_and_algebra_checks_only','passed':not failed,'checks':checks,'failures':failed,'production_tests_executed':False,'RL_training_runs':0,'VLM_requests':0,'robot_commands':0}
 (ROOT/'validation/package_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'passed':not failed,'checks':len(checks),'failures':failed,'production_tests_executed':False},ensure_ascii=False,indent=2))
 return 1 if failed else 0
if __name__=='__main__':raise SystemExit(main())
