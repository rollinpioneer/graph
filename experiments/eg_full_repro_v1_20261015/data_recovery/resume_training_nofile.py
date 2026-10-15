from pathlib import Path
import json,time
R=Path(__file__).resolve().parents[1];D=R/'data_recovery';p=D/'execution_pipeline.py'
old=json.loads((D/'PIPELINE_TRAINING_NOFILE_FAILURE.json').read_text());assert old['status']=='FAILED_STOPPED_AT_GATE' and old['stage']=='FORMAL_TRAINING'
scope={'__name__':'eg_training_nofile_resume','__file__':str(p)}
exec(compile(p.read_text().split('try:\n save()\n while True:')[0],str(p),'exec'),scope)
state=dict(old);state.pop('error',None);state.update(status='RUNNING',training_attempt=2,active_training_receipt='formal_training_attempt2',training_attempt2_admitted_unix=time.time());scope['state']=state
run=scope['run']
def retry_run(name,*args,**kwargs):return run(name+'_attempt2' if name=='formal_training' else name,*args,**kwargs)
scope['run']=retry_run
try:
 scope['stage']('FORMAL_TRAINING',scope['train'])
 state.update(status='TRAINING_EXITED_PENDING_FINAL_CHECKPOINT',stage='FINAL_CHECKPOINT_PENDING_FOREGROUND');scope['save']()
except Exception as e:
 state.update(status='FAILED_STOPPED_AT_GATE',error=repr(e));scope['save']();raise
print(json.dumps(state),flush=True)
