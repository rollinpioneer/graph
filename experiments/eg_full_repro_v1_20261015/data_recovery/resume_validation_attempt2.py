from pathlib import Path
import sys,json
R=Path(__file__).resolve().parents[1];D=R/'data_recovery';p=D/'execution_pipeline.py'
old=json.loads((D/'PIPELINE_STATUS.json').read_text());assert old['status']=='FAILED_STOPPED_AT_GATE' and old['stage']=='FULL_SUPERVISION_VALIDATION'
scope={'__name__':'eg_validation_attempt2','__file__':str(p)}
exec(compile(p.read_text().split('try:\n save()\n while True:')[0],str(p),'exec'),scope)
state=dict(old);state.pop('error',None);state.update(status='RUNNING',supervision_validator_attempt=2);scope['state']=state
try:
 scope['stage']('FULL_SUPERVISION_VALIDATION',lambda:scope['run']('validate_supervision_attempt2',[sys.executable,str(D/'validate_supervision.py')],cwd=R/'runtime/EconomicGrasp'));scope['gate']('SUPERVISION_VALIDATION.json')
 state.update(status='PREPARATION_COMPLETE_PENDING_REAL_SMOKE',stage='REAL_SMOKE_WAITING_FOR_AVAILABLE_GPU');scope['save']()
except Exception as e:
 state.update(status='FAILED_STOPPED_AT_GATE',error=repr(e));scope['save']();raise
print(json.dumps(state),flush=True)
