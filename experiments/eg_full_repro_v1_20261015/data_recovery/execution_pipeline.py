"""One authorized baseline only; every stage requires completed upstream gates."""
import os,sys,json,time,subprocess,concurrent.futures,fcntl,hashlib,shutil
from pathlib import Path
from resource_jobs import run,R
S=R/'runtime/EconomicGrasp';D=R/'data_recovery';OUT=R/'runs/EG_FULL_V1';STATUS=D/'PIPELINE_STATUS.json';lock=(D/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'experiment_id':'EG-FULL-REPRO-V1','status':'RUNNING','stage':'WAIT_RAW_VALIDATION','started_unix':time.time(),'completed_stages':[],'training_replicates':1,'seed':20261015}
def save():
 state['updated_unix']=time.time();p=STATUS.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2));p.replace(STATUS)
def gate(name):
 result=json.loads((D/name).read_text());assert result['status']=='PASS',(name,result.get('status'));return result
def stage(name,fn):
 state['stage']=name;save();print('STAGE',name,flush=True);fn();state['completed_stages'].append(name);save()
def script(name,gpu=False):return run(name.removesuffix('.py'),[sys.executable,str(D/name)],cwd=S,gpu=gpu)
def full_graspness():
 gate('GRASPNESS_EQUIVALENCE.json')
 def shard(i):
  return run(f'graspness_full_shard_{i}',[sys.executable,str(S/'dataset/generate_graspness.py'),'--dataset_root',str(R/'data/graspnet'),'--camera_type','kinect','--scene_start',str(i),'--scene_end','100','--scene_stride','3','--cache_scene_graspness','--resume_verified'],cwd=S,gpu=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(shard,range(3)))
 assert len(list((R/'data/graspnet/graspness').glob('scene_*/kinect/*.npy')))==25600
 state['graspness_files']=25600

def economic():
 (R/'data/graspnet/economic_grasp_label_300views').mkdir(parents=True,exist_ok=True)
 def shard(i):
  return run(f'economic_full_shard_{i}',[sys.executable,str(S/'dataset/generate_economic.py'),'--dataset_root',str(R/'data/graspnet'),'--camera_type','kinect','--scene_start',str(i),'--scene_end','100','--scene_stride','2','--resume_verified'],cwd=S)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(shard,range(2)))
 assert len(list((R/'data/graspnet/economic_grasp_label_300views').glob('*_labels.npz')))==100

def train():
 gate('RAW_DATA_VALIDATION.json');gate('GRASPNESS_EQUIVALENCE.json');gate('SUPERVISION_VALIDATION.json');gate('SMOKE_STATUS.json')
 assert shutil.disk_usage(R).free>24*1024**3
 train_dir=OUT/'train';train_dir.mkdir(exist_ok=True)
 assert not list(train_dir.glob('economicgrasp_epoch*.tar')) and not (train_dir/'batch_metrics.jsonl').exists(),'Do not start a second training; inspect/resume same experiment explicitly'
 command=[sys.executable,str(S/'train.py'),'--model','economicgrasp','--camera','kinect','--log_dir',str(train_dir),'--max_epoch','10','--batch_size','4','--dataset_root',str(R/'data/graspnet')]
 result=run('formal_training',command,cwd=S,gpu=True);state['formal_training_receipt']=result

def evaluate():
 gate('TRAINING_COMPLETE.json');gate('dex_cache_manifest.json');os.environ['EG_EVAL_PROCESSES']='4';os.environ['EG_SAVE_INFERENCE_INPUTS']='1'
 for split in ['seen','similar','novel']:
  state['stage']='OFFICIAL_TEST_'+split.upper();save();target=OUT/'test'/split;target.mkdir(parents=True,exist_ok=True)
  assert not list(target.glob('scene_*/kinect/*.npy')),'Inspect existing test results; do not overwrite'
  run('official_test_'+split,[sys.executable,str(S/'test.py'),'--model','economicgrasp','--camera','kinect','--batch_size','4','--checkpoint_path',str(OUT/'train/economicgrasp_epoch10.tar'),'--dataset_root',str(R/'data/graspnet'),'--save_dir',str(target),'--test_mode',split,'--inference'],cwd=S,gpu=True)
try:
 save()
 while True:
  p=D/'RAW_VALIDATION_DRIVER.json';raw=json.loads(p.read_text()) if p.exists() else {}
  if raw.get('status')=='FAILED':raise RuntimeError(raw)
  if raw.get('status')=='PASS':break
  time.sleep(20)
 gate('RAW_DATA_VALIDATION.json')
 expected='4119bdcd6bf5d3712a110f78ca87504dd359eec0';assert subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()==expected
 allowed={'train.py','test.py','dataset/generate_graspness.py','dataset/generate_economic.py'}
 changed=set(subprocess.check_output(['git','-C',str(S),'diff','--name-only'],text=True).splitlines());assert changed<=allowed
 patch=subprocess.check_output(['git','-C',str(S),'diff','--binary']);state['runtime_patch_sha256']=hashlib.sha256(patch).hexdigest();state['upstream_commit']=expected
 stage('GRASPNESS_EQUIVALENCE',lambda:script('calibrate_graspness.py'))
 stage('FULL_GRASPNESS',full_graspness)
 stage('FULL_ECONOMIC_SUPERVISION',economic)
 stage('FULL_SUPERVISION_VALIDATION',lambda:script('validate_supervision.py'))
 stage('REAL_DATA_SMOKE',lambda:script('real_smoke.py',True));gate('SMOKE_STATUS.json')
 stage('FORMAL_TRAINING',train)
 stage('FINAL_CHECKPOINT_AND_64000_UPDATES',lambda:script('summarize_training.py',True))
 stage('OFFICIAL_TESTS',evaluate)
 stage('EVALUATION_SUMMARY',lambda:script('summarize_evaluation.py'))
 state['status']='EVALUATED_PENDING_BAD_CASE_AND_FINAL_ARCHIVE';state['stage']='WAIT_DIAGNOSTIC_REVIEW';save()
except Exception as e:
 state.update(status='FAILED_STOPPED_AT_GATE',error=repr(e));save();raise
print(json.dumps(state),flush=True)
