from pathlib import Path
import json
R=Path(__file__).resolve().parents[1];D=R/'data_recovery';p=D/'execution_pipeline.py'
old=json.loads((D/'PIPELINE_DIRECTORY_RACE_FAILURE.json').read_text())
assert old['status']=='FAILED_STOPPED_AT_GATE' and old['stage']=='FULL_ECONOMIC_SUPERVISION'
assert json.loads((D/'ECONOMIC_DIRECTORY_RACE.json').read_text())['source_algorithm_changed'] is False
for name in ['economic_full_shard_1','economic_full_shard_0_attempt2']:
 assert json.loads((R/'receipts'/(name+'.json')).read_text())['status']=='PASS'
assert len(list((R/'data/graspnet/economic_grasp_label_300views').glob('*_labels.npz')))==100
scope={'__name__':'eg_validation_resume','__file__':str(p)}
exec(compile(p.read_text().split('try:\n save()\n while True:')[0],str(p),'exec'),scope)
state=dict(old);state.pop('error',None);state.update(status='RUNNING',directory_race_recovery='economic_full_shard_0_attempt2');scope['state']=state
state['completed_stages'].append('FULL_ECONOMIC_SUPERVISION')
try:
 scope['stage']('FULL_SUPERVISION_VALIDATION',lambda:scope['script']('validate_supervision.py'));scope['gate']('SUPERVISION_VALIDATION.json')
 state.update(status='PREPARATION_COMPLETE_PENDING_REAL_SMOKE',stage='REAL_SMOKE_WAITING_FOR_AVAILABLE_GPU');scope['save']()
except Exception as e:
 state.update(status='FAILED_STOPPED_AT_GATE',error=repr(e));scope['save']();raise
print(json.dumps(state),flush=True)
