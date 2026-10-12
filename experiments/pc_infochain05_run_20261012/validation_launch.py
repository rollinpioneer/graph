"""Execute frozen matched pairs sequentially; no outcome-based sampling or tuning."""
import os,pathlib,json,hashlib,subprocess,datetime,sys
r=pathlib.Path(__file__).resolve().parent;py=sys.executable;freeze=json.loads((r/'receipts/validation_freeze.json').read_text())
for f,digest in freeze['freeze_files_sha256'].items():assert hashlib.sha256((r/f).read_bytes()).hexdigest()==digest,f
plans=[]
for pair in freeze['pairs']:
 for arm in ['permuted','blocked']:
  plans.append({'n':pair['n'],'scene_seed':pair['scene_seed'],'policy_seed':pair['policy_seed'],'arm':arm,'gpu':int(sys.argv[1])})
receipt={'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plans':plans,'parallel_workers':1,'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'freeze_sha256':hashlib.sha256((r/'receipts/validation_freeze.json').read_bytes()).hexdigest()}
(r/'receipts/validation_launch.json').write_text(json.dumps(receipt,indent=2))
done=[];failed_n=set()
for plan in plans:
 name='validate%d_%s'%(plan['n'],plan['arm'])
 if plan['n'] in failed_n:
  done.append({'run':name,'status':'not_executed_after_matched_pair_technical_failure'});continue
 prior=sum(json.loads(f.read_text()).get('charged_gpu_seconds',0) for f in (r/'runs').glob('*/guard_result.json'))
 valsum=sum(json.loads(f.read_text()).get('charged_gpu_seconds',0) for f in (r/'runs').glob('validate*/guard_result.json'))
 limit=min(freeze['guard_s_each'],freeze['phase_max_gpu_s']-valsum,7000-prior)
 if limit<50:done.append({'run':name,'status':'not_executed_budget_exhausted'});continue
 cmd=[py,'-u',str(r/'guard.py'),'--out',str(r/'runs'/name),'--seconds',str(limit),'--gpu',str(plan['gpu']),'--',py,'-u',str(r/'rollout.py'),'--out',str(r/'runs'/name/'data'),'--n',str(plan['n']),'--scene-seed',str(plan['scene_seed']),'--policy-seed',str(plan['policy_seed']),'--role','validation','--arm',plan['arm']]
 print('LAUNCH',name,'limitGPU_s',limit,flush=True)
 with (r/'logs'/(name+'_guard.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=False)
 result=json.loads((r/'runs'/name/'guard_result.json').read_text());complete=r/'runs'/name/'data/completed.json'
 success=result['exit_code']==0 and complete.exists() and json.loads(complete.read_text())['full_episodes']==1
 done.append({'run':name,'status':'executed_complete' if success else 'failed_or_stopped','exit_code':result['exit_code'],'charged_gpu_seconds':result['charged_gpu_seconds'],'stop_reason':result['stop_reason']})
 if not success:failed_n.add(plan['n'])
 (r/'receipts/validation_execution_status.json').write_text(json.dumps(done,indent=2))
 print('JOB_STATUS',name,done[-1]['status'],flush=True)
(r/'receipts/validation_execution_status.json').write_text(json.dumps(done,indent=2));print('VALIDATION_SEQUENCE_COMPLETE',flush=True)

