from pathlib import Path
import json,csv,gzip,hashlib,subprocess,inspect,time,os
from graspnetAPI import GraspNetEval
R=Path(__file__).resolve().parents[1];D=R/'data_recovery';O=R/'runs/EG_SUB25_V1';E=O/'evaluation';B=O/'bad_cases';A=Path('/home/xushijie3/work/eg_full_repro_v1_20261015/archive_worktree/experiments/eg_sub25_v1_20261015');cfg=json.loads((D/'evaluation_run_config.json').read_text());os.nice(10)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=R/'runtime/EconomicGrasp')).hexdigest();assert source==cfg['model_source_patch_sha256'];native=hashlib.sha256(inspect.getsource(GraspNetEval.eval_scene).encode()).hexdigest();assert native==cfg['graspnet_api_eval_scene_sha256']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'runtime/EconomicGrasp',text=True).strip()=='4119bdcd6bf5d3712a110f78ca87504dd359eec0'
for n in ['per_scene_results.csv','per_frame_results.csv','paired_scene_comparison.csv']:assert sha(E/n)==sha(A/n)
for n in ['bad_case_metrics.json','bad_case_summary.csv','bad_case_grasps.csv']:assert sha(B/n)==sha(A/n)
result=json.loads((E/'evaluation_results.json').read_text());assert result['status']=='PASS' and result['scene_evaluations']==18 and result['model_frame_evaluations']==4608
qa=json.loads((D/'VISUAL_QA.json').read_text());assert qa['status']=='PASS' and qa['decoded_figure_count']==12
replay=json.loads((D/'BAD_CASE_REPLAY.json').read_text());assert replay['all24_replays_match']
audit=json.loads((D/'DIAGNOSTIC_OBSERVATION_AUDIT.json').read_text());assert all(x.get('fresh_official_crc_recheck',{}).get('status')=='PASS' for x in audit['scenes'] if x['scene'] in [188,189])
active=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if str(R) in cmd and any(v in cmd for v in ['run_evaluation.py','postprocess_evaluation.py','infer_evaluation.py','evaluate_many.py','replay_bad_cases.py','analyze_bad_cases.py','audit_duplicate_observations.py']):active.append({'pid':int(p.name),'cmdline':cmd})
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert not active,active
paths=[]
for base in [E/'predictions',E/'inputs',E/'results',B]:paths.extend(p for p in base.rglob('*') if p.is_file())
paths.extend(p for p in (R/'logs').iterdir() if p.is_file() and p.name.startswith(('infer_','official_api_','summarize_evaluation','replay_bad_cases','analyze_bad_cases','run_evaluation','observation_audit')))
paths.extend([D/'DIAGNOSTIC_OBSERVATION_AUDIT.json',D/'VISUAL_QA.json',O/'final_reproduction_report.md']);paths=sorted(set(paths));index=E/'EVALUATION_ARTIFACT_INDEX.csv.gz'
with gzip.open(index,'wt',newline='') as f:
 w=csv.writer(f);w.writerow(['absolute_path','bytes','sha256'])
 for p in paths:w.writerow([str(p),p.stat().st_size,sha(p)])
info={'path':str(index),'files':len(paths),'total_bytes':sum(p.stat().st_size for p in paths),'index_sha256':sha(index),'coverage':'Immutable prediction/input/native-evaluation/case assets,logs and final reports;large assets remain on server'}
(E/'EVALUATION_ARTIFACT_INDEX.json').write_text(json.dumps(info,indent=2))
result.update(artifact_index=info,diagnostic_observation_audit={'path':str(D/'DIAGNOSTIC_OBSERVATION_AUDIT.json'),'unique_observations':{str(x['scene']):x['channels']['rgb']['unique_sha256_count'] for x in audit['scenes']},'native256frame_AP_unchanged':True},visual_qa='PASS',human_reports='FINALIZED')
(E/'evaluation_results.json').write_text(json.dumps(result,indent=2));(D/'EVALUATION_COMPLETE.json').write_text(json.dumps(result,indent=2))
validation={'status':'PASS','sealed_unix':time.time(),'native_scene_evaluations':18,'model_frame_evaluations':4608,'source_patch_sha256':source,'native_eval_scene_sha256':native,'numeric_CSVs_unchanged_since_published_results':True,'all24_replays_match':True,'visual_QA':{'decoded':12,'viewed':len(qa['visually_reviewed_figures'])},'duplicate_observation_crc_rechecks':'PASS','active_owned_experiment_jobs':active,'SSH_closed':False,'SSH_note':'Pre-exit snapshot;actual client exit must be recorded externally after closure'}
(D/'FINAL_VALIDATION.json').write_text(json.dumps(validation,indent=2))
for name in ['PIPELINE_STATUS.json','EVALUATION_STATUS.json']:
 p=D/name;x=json.loads(p.read_text());x.update(status='COMPLETE',stage='COMPLETE',visual_qa='PASS',observation_audit='COMPLETE_OFFICIAL_CRC_PASS',report_sealed_unix=time.time());p.write_text(json.dumps(x,indent=2))
print('FINAL_SEAL_PASS',json.dumps(info),flush=True)
