"""Continue this card through the real smoke gate and its single formal training."""
import sys,json,time,hashlib,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'data_recovery';S=R/'runtime/EconomicGrasp'
old=json.loads((D/'PIPELINE_STATUS.json').read_text())
assert old['status']=='PREPARATION_COMPLETE_PENDING_REAL_SMOKE',old
p=D/'execution_pipeline.py';scope={'__name__':'eg_training_resume','__file__':str(p)}
exec(compile(p.read_text().split('try:\n save()\n while True:')[0],str(p),'exec'),scope)
state=dict(old);state.update(status='RUNNING',stage='REAL_DATA_SMOKE');scope['state']=state
stage,gate,save=scope['stage'],scope['gate'],scope['save']
try:
 for name in ['RAW_DATA_VALIDATION.json','GRASPNESS_EQUIVALENCE.json','CPU_KNN_CROSS_SCENE_CHECK.json','SUPERVISION_VALIDATION.json']:gate(name)
 patch=subprocess.check_output(['git','-C',str(S),'diff','--binary'])
 assert hashlib.sha256(patch).hexdigest()==state['runtime_patch_sha256']
 stage('REAL_DATA_SMOKE',lambda:scope['script']('real_smoke.py',True));gate('SMOKE_STATUS.json')
 state['formal_training_started']=True;state['formal_training_admitted_unix']=time.time();save()
 stage('FORMAL_TRAINING',scope['train'])
 state.update(status='TRAINING_EXITED_PENDING_FINAL_CHECKPOINT',stage='FINAL_CHECKPOINT_PENDING_FOREGROUND');save()
except Exception as e:
 state.update(status='FAILED_STOPPED_AT_GATE',error=repr(e));save();raise
print(json.dumps(state),flush=True)
