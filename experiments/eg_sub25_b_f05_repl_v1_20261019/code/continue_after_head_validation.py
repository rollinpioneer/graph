"""Continue only post-training validation and untouched C3 stages."""
from repl_common import *
prefix=(R/'code/pipeline.py').read_text().split("try:\n status('WAITING_FOR_BACKBONE')")[0]
assert 'def export_stage' in prefix and 'while True:' not in prefix
exec(compile(prefix,'existing_pipeline_functions','exec'),globals())
try:
 for p,h in json.loads((R/'results/execution_code_manifest.json').read_text()).items():assert sha256(Path(p))==h
 assert json.loads((R/'results/backbone_status.json').read_text())['status']=='PASS'
 assert json.loads((R/'results/label_validation.json').read_text())['status']=='PASS'
 assert not (R/'C3_method_lock.json').exists(),'Preserve any existing C3 lock; inspect before continuation'
 status('P2_VALIDATE_EXISTING_HEADS_NO_TRAINING');py('validate_saved_heads','finish_saved_heads.py')
 status('P3_METHOD_LOCK');py('lock_C3_recovery','lock_C3.py')
 for bone in ['A','B']:
  status('P3_C3_EXPORT_'+bone,GPU_workers_max=8);export_stage('C3',bone)
 tasks=[(f'eval_{bone}_{s:04d}','evaluate.py',['--phase','confirmation','--scene',str(s)],'CPU',dict(REPL_EVAL_ROOT=str(R/bone))) for s in sum(P['C3'].values(),[]) for bone in ['A','B']]
 ne=cpu_workers('evaluation',24);status('P3_C3_EVALUATION',workers=ne);batch(tasks,ne)
 status('P4_FINALIZE');py('finalize','finalize.py')
 u=usage();size=disk_bytes();assert u['GPU']<=28800 and u['CPU']<=180000 and size<=25*1024**3
 nl=json.loads((R/'results/label_validation.json').read_text())['workers_max']
 write(R/'results/resource_summary.json',dict(status='PASS',usage=u,new_runtime_bytes=size,archive_worktree_excluded=True,cpu_workers_initial=nl,cpu_workers_max=max(nl,ne),evaluation_workers=ne,GPU_export_workers_max=8,gpu_training_replicates=1,head_replicates=3,head_training_updates=3000,post_save_validation_recovery_additional_updates=0,continuation_wall_seconds=time.time()-start))
 write(R/'results/pipeline_status.json',dict(status='COMPLETED_PENDING_GIT_ARCHIVE',continuation_wall_seconds=time.time()-start,usage=u,final_report=str(R/'results/final_report.md'),all_experiment_subprocesses_exited=True))
except Exception as e:
 write(R/'results/pipeline_status.json',dict(status='TECHNICAL_STOP',error=repr(e),traceback=traceback.format_exc(),wall_seconds=time.time()-start,usage=usage()));raise
