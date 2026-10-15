from pathlib import Path
import os,json,time,sys
from resource_jobs import run
R=Path(__file__).resolve().parents[1];D=R/'data_recovery';S=R/'runtime/EconomicGrasp';P=os.environ['CONDA_PREFIX']+'/bin/python3';target=D/'EVALUATION_POSTPROCESS_STATUS.json'
def save(status,stage,**extra):target.write_text(json.dumps({'status':status,'stage':stage,'updated_unix':time.time(),**extra},indent=2))
assert not target.exists()
save('RUNNING','WAITING_FOR_NATIVE_AP')
try:
 while True:
  state=json.loads((D/'EVALUATION_STATUS.json').read_text())
  if state['status'].startswith('FAILED'):raise RuntimeError(state)
  if state['status']=='EVALUATION_FINISHED_PENDING_SUMMARY_AND_BAD_CASE':break
  time.sleep(10)
 for script,gpu in [('summarize_evaluation',False),('replay_bad_cases',True),('analyze_bad_cases',False)]:
  save('RUNNING',script);run(script,[P,D/(script+'.py')],cwd=S,gpu=gpu,preferred=0 if gpu else None)
 save('RUNNING','finalize_evaluation_report');run('finalize_evaluation_report',[P,D/'finalize_evaluation_report.py'],cwd=S)
 save('COMPLETE','COMPLETE');print('EVALUATION_POSTPROCESS_COMPLETE',flush=True)
except Exception as e:
 save('FAILED_STOPPED_AT_GATE',locals().get('script','WAITING_FOR_NATIVE_AP'),error=repr(e));raise
