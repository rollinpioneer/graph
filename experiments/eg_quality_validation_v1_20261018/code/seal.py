import os,json,hashlib,inspect,shutil,csv,gzip,subprocess,time
from pathlib import Path
import psutil
import graspnetAPI.utils.eval_utils as EU
from graspnetAPI import GraspNetEval
N=Path('/home/xushijie3/work/eg_quality_validation_v1_20261018')
W=Path('/home/xushijie3/work/eg_full_repro_v1_20261015/archive_worktree')
A=W/'experiments/eg_quality_validation_v1_20261018'
branch='codex/eg-quality-validation-v1-20261018'
assert subprocess.check_output(['git','-C',str(W),'branch','--show-current'],text=True).strip()==branch
q=json.loads((N/'QUEUE_COMPLETE.json').read_text());assert q['all_pass'] and len(q['jobs'])==18
s=json.loads((N/'summary.json').read_text());assert s['top1_model_frames']==1152 and s['pool_model_frames']==72
assert json.loads((N/'geometry_grid_check.json').read_text())['status']=='PASS'
assert json.loads((N/'decoder_control_summary.json').read_text())['failed_top1']==187
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
native=json.loads((N/'native_source_hashes.json').read_text())
for name,f in [('collision_detection',EU.collision_detection),('get_grasp_score',EU.get_grasp_score),('eval_scene',GraspNetEval.eval_scene)]:
    assert hashlib.sha256(inspect.getsource(f).encode()).hexdigest()==native[name]
sources={}
for p in (N/'results').glob('*/sources.json'):sources.update(json.loads(p.read_text()))
for p,h in sources.items():assert sha(Path(p))==h,p
prior=json.loads((A/'source_provenance.json').read_text())
for p,h in prior.items():assert sha(Path(p))==h,p
active=[]
for p in psutil.process_iter(['pid','cmdline']):
    if p.pid==os.getpid():continue
    try:
        cmd=' '.join(p.info['cmdline'] or [])
        if str(N/'code') in cmd and 'python' in cmd:active.append({'pid':p.pid,'cmd':cmd})
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
assert not active,active
env=json.loads((N/'environment_report.json').read_text())
env['runtime_sources']={p.name:sha(p) for p in (N/'code').glob('*.py')}
(N/'environment_report.json').write_text(json.dumps(env,indent=2))
validation=dict(status='PASS',source_predictions_rehashed=len(sources),native_function_hashes_unchanged=True,prior_sources_unchanged=True,completed_scene_model_groups=18,top1_frames=1152,pool_frames=72,verified_original_candidates=s['total_verified'],all_jobs_exit_zero=True,owned_experiment_processes=[],ssh_closed=False,ssh_note='Connection still open for publication; actual exit is recorded after closing in local receipt.',time=time.time())
(N/'FINAL_VALIDATION.json').write_text(json.dumps(validation,indent=2))
with gzip.open(N/'RAW_ARTIFACT_INDEX.csv.gz','wt',newline='') as f:
    wr=csv.writer(f);wr.writerow(['relative_path','bytes','sha256'])
    for p in sorted(N.rglob('*')):
        if p.is_file() and p.name not in ('RAW_ARTIFACT_INDEX.csv.gz','git_stage_paths.txt') and p != N/'logs/seal.log' and '__pycache__' not in p.parts:
            wr.writerow([str(p.relative_to(N)),p.stat().st_size,sha(p)])
files=['config.json','EXPERIMENT_CARD.md','summary.json','validation_report.md','validation_figure.png','candidate_diagnostics.csv','per_scene_oracle.csv','exploratory_cases.json','cost_results.json','cost_official_results.json','cost_actual_quality.json','storage_projection.json','environment_report.json','native_source_hashes.json','geometry_grid_check.json','METHOD_ADAPTATION.json','QUEUE_INITIAL.json','QUEUE_COMPLETE.json','RECOVERY_COMPLETE.json','FINAL_VALIDATION.json','RAW_ARTIFACT_INDEX.csv.gz','resource_snapshots.jsonl']
files+=['decoder_control_summary.json','DECISION.json']
for name in files:shutil.copy2(N/name,A/name)
for p in (N/'code').glob('*.py'):shutil.copy2(p,A/'code'/p.name)
paths=sorted(p for p in A.rglob('*') if p.is_file() and p.name!='LIGHTWEIGHT_MANIFEST.json')
manifest={'branch':branch,'source_root':str(N),'files':[{'path':str(p.relative_to(A)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths]}
assert sum(x['bytes'] for x in manifest['files'])<15*1024**2
(A/'LIGHTWEIGHT_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
paths.append(A/'LIGHTWEIGHT_MANIFEST.json')
(N/'git_stage_paths.txt').write_text('\n'.join(str(p.relative_to(W)) for p in paths)+'\n')
print(json.dumps(dict(validation=validation,light_files=len(paths),light_bytes=sum(p.stat().st_size for p in paths)),indent=2))
