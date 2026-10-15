import sys,json,time,hashlib,subprocess
from pathlib import Path
from resource_jobs import run,R
D=R/'data_recovery';S=R/'runtime/EconomicGrasp';P=D/'EVALUATION_STATUS.json'
assert not P.exists(),'Do not duplicate the evaluation controller'
state={'status':'RUNNING','stage':'INFERENCE','started_unix':time.time(),'completed_jobs':[],'expected_model_frames':4608,'expected_scene_evaluations':18}
def save():
 tmp=P.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(P)
 main=json.loads((D/'PIPELINE_STATUS.json').read_text());main.update(status='EVALUATING',stage=state['stage'],evaluation_status=str(P));(D/'PIPELINE_STATUS.json').write_text(json.dumps(main,indent=2))
try:
 cfg=json.loads((D/'evaluation_run_config.json').read_text());assert hashlib.sha256(subprocess.check_output(['git','-C',str(S),'diff','--binary'])).hexdigest()==cfg['model_source_patch_sha256']
 for role,part in [('subset','dev'),('subset','diagnostic_seen'),('official','diagnostic_seen'),('subset','diagnostic_novel'),('official','diagnostic_novel')]:
  name='infer_'+role+'_'+part;state['stage']=name;save();run(name,[sys.executable,str(D/'infer_evaluation.py'),'--role',role,'--partition',part],cwd=S,gpu=True,preferred=0);state['completed_jobs'].append(name);save()
 for scope in ['dev','diagnostic']:
  name='official_api_'+scope;state['stage']=name;save();run(name,[sys.executable,str(D/'evaluate_many.py'),'--scope',scope],cwd=S);state['completed_jobs'].append(name);save()
 state.update(status='EVALUATION_FINISHED_PENDING_SUMMARY_AND_BAD_CASE',stage='SUMMARY_AND_BAD_CASE');save()
except Exception as e:
 state.update(status='FAILED_STOPPED_AT_GATE',error=repr(e));save();raise
print(json.dumps(state),flush=True)
