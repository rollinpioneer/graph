from pathlib import Path
import json
R=Path(__file__).resolve().parents[1];D=R/'data_recovery';p=D/'execution_pipeline.py'
old=json.loads((D/'PIPELINE_STATUS.json').read_text());assert old['status']=='TRAINING_EXITED_PENDING_FINAL_CHECKPOINT',old
assert json.loads((R/'receipts/formal_training_attempt2.json').read_text())['status']=='PASS'
scope={'__name__':'eg_evaluation_resume','__file__':str(p)}
exec(compile(p.read_text().split('try:\n save()\n while True:')[0],str(p),'exec'),scope)
state=dict(old);state.update(status='RUNNING');scope['state']=state
try:
 scope['stage']('FINAL_CHECKPOINT_AND_64000_UPDATES',lambda:scope['script']('summarize_training.py',True))
 scope['stage']('OFFICIAL_TESTS',scope['evaluate'])
 scope['stage']('EVALUATION_SUMMARY',lambda:scope['script']('summarize_evaluation.py'))
 state.update(status='EVALUATED_PENDING_BAD_CASE_AND_FINAL_ARCHIVE',stage='WAIT_DIAGNOSTIC_REVIEW');scope['save']()
except Exception as e:
 state.update(status='FAILED_STOPPED_AT_GATE',error=repr(e));scope['save']();raise
print(json.dumps(state),flush=True)
