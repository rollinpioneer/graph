from pathlib import Path
import sys,json,time,hashlib,subprocess,os
from resource_jobs import run,R
D=R/'data_recovery';S=R/'runtime/EconomicGrasp';OUT=R/'runs/EG_SUB25_V1';P=D/'PIPELINE_STATUS.json'
assert not P.exists(),'Do not duplicate this run'
state={'experiment_id':'EG-SUB25-V1','status':'RUNNING','stage':'REAL_80FRAME_SMOKE','seed':20261015,'training_replicates':1,'expected_updates':16000,'started_unix':time.time(),'completed_stages':[]}
def save():
 tmp=P.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(P)
try:
 assert json.loads((D/'SUPERVISION_VALIDATION.json').read_text())['status']=='PASS'
 config=json.loads((OUT/'train_config.json').read_text())
 assert hashlib.sha256((D/'frozen_split.json').read_bytes()).hexdigest()==config['split_sha256']
 assert hashlib.sha256(subprocess.check_output(['git','-C',str(S),'diff','--binary'])).hexdigest()==config['runtime_patch_sha256']
 save();run('real_smoke',[sys.executable,str(D/'real_smoke.py')],cwd=S,gpu=True)
 smoke=json.loads((D/'SMOKE_STATUS.json').read_text());assert smoke['status']=='PASS' and smoke['dataset_images']==6400 and smoke['updates']==20 and len(smoke['indices'])==80
 state['completed_stages'].append('REAL_80FRAME_SMOKE');state['stage']='FORMAL_TRAINING';state['formal_training_admitted_unix']=time.time();save()
 config.update(preflight_passed=True,formal_training_started=True);(OUT/'train_config.json').write_text(json.dumps(config,indent=2))
 assert not (OUT/'train').exists()
 result=run('formal_training',[sys.executable,str(S/'train.py'),'--model','economicgrasp','--camera','kinect','--dataset_root',str(R/'data/graspnet'),'--log_dir',str(OUT/'train'),'--max_epoch','10','--batch_size','4','--num_point','20000','--voxel_size','0.005','--learning_rate','0.001','--split_manifest',str(D/'frozen_split.json')],cwd=S,gpu=True)
 state.update(status='TRAINING_EXITED_PENDING_FINAL_CHECKS',stage='FINAL_CHECKPOINT_AND_16000_UPDATES_PENDING',training_receipt=result);state['completed_stages'].append('FORMAL_TRAINING');save()
except Exception as e:
 state.update(status='FAILED_STOPPED_AT_GATE',error=repr(e));save();raise
print(json.dumps(state),flush=True)
