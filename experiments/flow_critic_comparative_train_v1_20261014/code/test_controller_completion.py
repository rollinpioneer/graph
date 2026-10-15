"""Synthetic filesystem receipts only; optimizer updates=0 and no live processes."""
from pathlib import Path
import json,tempfile,copy,time
from controller_completion import EXPECTED,completed_work
ROOT=Path(__file__).resolve().parents[1]
results=[]
def record(name,value):
 assert value,name
 results.append({'name':name,'status':'PASS'})
with tempfile.TemporaryDirectory(prefix='controller_completion_fixture_') as td:
 root=Path(td)
 def write(rel,value):
  p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
 current={'runs':{},'queue':{'completed':sorted(EXPECTED),'active':{},'failed':[]},'evaluation_queue':{'completed':sorted(EXPECTED),'active':None,'failed':[],'skipped':[]}}
 for job in EXPECTED:
  phase='T1' if job.startswith(('G0','D0')) else 'T3'
  current['runs'][job]={'status':{'status':'COMPLETED','updates':1000000,'exit_code':0,'phase':phase}}
  write('runs/'+job+'/'+phase+'_exit.json',{'status':'PASS','exit_code':0})
 write('receipts/manager_finished.json',{'completed':sorted(EXPECTED),'failed':[]})
 record('Completed training controller absence accepted',completed_work('manager_started.json',current,root))
 record('Evaluation absence before completion rejected',not completed_work('evaluation_manager_started.json',current,root))
 for field,bad in [('status','RUNNING'),('updates',999999),('exit_code',1)]:
  altered=copy.deepcopy(current);altered['runs']['G1_seed0']['status'][field]=bad
  record('Incomplete or failed native '+field+' rejected',not completed_work('manager_started.json',altered,root))
 missing=root/'runs/G1_seed0/T3_exit.json';missing.unlink()
 record('Missing authoritative native exit rejected',not completed_work('manager_started.json',current,root))
 write('runs/G1_seed0/T3_exit.json',{'status':'PASS','exit_code':0})
 altered=copy.deepcopy(current);altered['queue']['active']={'G1_seed0':'T3'}
 record('Active queue rejected',not completed_work('manager_started.json',altered,root))
 for job in EXPECTED:
  write('runs/'+job+'/matched_verified/completed.json',{'status':'PASS','optimizer_updates':0})
  write('runs/'+job+'/matched_verified_exit.json',{'exit_code':0,'optimizer_updates':0})
  for step in [800000,900000,1000000]:write('runs/'+job+'/matched_verified/'+str(step)+'.json',{'status':'PASS','optimizer_updates':0,'step':step,'episodes':[{}]*100})
 write('receipts/evaluation_manager_finished.json',{'completed':sorted(EXPECTED),'failed':[],'skipped':[]})
 record('All evaluation terminal receipts accepted',completed_work('evaluation_manager_started.json',current,root))
 write('runs/G1_seed0/matched_verified/1000000.json',{'status':'PASS','optimizer_updates':1,'step':1000000,'episodes':[{}]*100})
 record('Nonzero evaluation update rejected',not completed_work('evaluation_manager_started.json',current,root))
 write('runs/G1_seed0/matched_verified/1000000.json',{'status':'PASS','optimizer_updates':0,'step':1000000,'episodes':[{}]*99})
 record('Incomplete evaluation episodes rejected',not completed_work('evaluation_manager_started.json',current,root))
write_to=ROOT/'receipts/monitor/controller_completion_tests_20261014_1458.json'
write_to.write_text(json.dumps({'epoch':time.time(),'status':'PASS','optimizer_updates':0,'live_processes_touched':0,'fixtures_only':True,'tests':results},indent=2))
print('COMPLETION_FIXTURE_TESTS_PASS',len(results),'OPTIMIZER_UPDATES',0)
