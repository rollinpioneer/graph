#!/usr/bin/env python3
import csv,json,pathlib,shutil,hashlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]; RUNS=ROOT/'runs/v13_r3_recovery'; REPORT=ROOT/'reports/v13_r3_recovery'; JOB=RUNS/'D0/B2/seed_0/20260928T032333Z_4e75091f'; FULL=RUNS/'full_original_dev10_reused.json'; ABS=RUNS/'full_absent_dev10.json'
def write(p,x): pathlib.Path(p).parent.mkdir(parents=True,exist_ok=True); pathlib.Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')
def main():
 # Required run-level aliases, with actual stop explicitly recorded.
 write(RUNS/'b2_eval_n_000000.json',json.loads((JOB/'eval_n_000000.json').read_text()))
 d=json.loads((JOB/'eval_final.json').read_text()); d['requested_label']='N=2048_or_actual_stop'; d['actual_transition_count']=1935; d['actual_stop_reason']='Tcap_first'; write(RUNS/'b2_eval_n_002048.json',d)
 rows=[]; fo=json.loads(FULL.read_text()); fa=json.loads(ABS.read_text());
 for r in fo['rows']: rows.append({'condition':'Full@Original','scope':'dev10_reused',**r})
 for r in fa['rows']: rows.append({'condition':'Full@Absent','scope':'dev10_new',**r})
 with (REPORT/'full_original_absent_dev10.csv').open('w',newline='') as f:
  fields=sorted({k for r in rows for k in r}); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
 shutil.copyfile(REPORT/'full_prior_conditional.csv',REPORT/'full_prior_conditional_train.csv')
 shutil.copyfile(REPORT/'matched_comparison.csv',REPORT/'matched_n2048_comparison.csv')
 with (REPORT/'unresolved_items.csv').open('w',newline='') as f:
  w=csv.writer(f); w.writerow(['item','status']); w.writerow(['B2_N2048','NOT_REACHED_TCAP_FIRST_N1935']); w.writerow(['Full_shadow_forward','NOT_RUN_ACTION_TV_NA']); w.writerow(['Full_resume','NOT_AUTHORIZED']);
 source={'worktree':str(ROOT),'source_commit':'b11061f10ec6a07cec5681bf26436b8d2857a100','baseline_evidence_commit':'dbaf23ba7c8137b7cfcf31b7c9a1c28af7c4b447','python':'/home/xushijie2/envs/lerobotpi0-xfs/bin/python','account':'xushijie2','host':'gpu03','runtime_manifest':'experiments/manifests/runtime_manifest_v211.yaml','full_checkpoint_sha256':json.loads((RUNS/'full_checkpoint_verification.json').read_text())['checkpoint_sha256'],'full_resume_sha256':json.loads((RUNS/'full_checkpoint_verification.json').read_text())['resume_sha256'],'b2_job_dir':str(JOB),'b2_checkpoint_files_not_committed':True}
 write(RUNS/'source_input_checkpoint_hashes.json',source)
 stamp=RUNS/'20260928T032333Z_4e75091f'; stamp.mkdir(parents=True,exist_ok=True); write(stamp/'recovery_manifest.json',{'status':'COMPLETE_TCAP_FIRST','canonical_outputs':str(RUNS),'b2_actual_N':1935,'b2_actual_T':9319.000000000771,'full_boundary_N':2048,'full_boundary_T':9771.400000000713,'new_eval_episodes':38,'pt_bytes_present_on_XFS_not_committed':True}); (stamp/'README.md').write_text('# R3 Recovery attempt 20260928T032333Z_4e75091f\n\nCanonical recovery outputs are one directory above and in `D0/B2`; this stamp directory contains metadata only. B2 stopped Tcap-first at N=1935.\n',encoding='utf-8')
if __name__=='__main__': main()
